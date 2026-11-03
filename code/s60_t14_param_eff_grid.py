"""s60: T-14 参数效率网格（两臂不同可训练范围 × r 扫描）。

问题（用户定义）：同一技能，两模型各需多少**训练参数**才能学会。

与 T-11 的关键差异：
  1) 任务已通过 pilot 验证（hop2 基础技能 + 深度外推），非饱和任务；
  2) 两臂可训练范围**有意不同**：稠密臂覆盖全 24 层投影，循环臂只覆盖
     M 区 [4,16) —— 这样曲线才可能在参数预算轴上交叉；
  3) 训练/评估/早停协议两侧完全一致（同题、同 lr、同 bps、同评估集）。

臂定义：
  dense: 原始 E/M/D 单遍（bypass，depth=1），LoRA 覆盖全 24 层投影；
  loop : S08 armB 权重（旧 LoRA+门冻结）+ 新 LoRA 覆盖 M 区，depth=2。

网格：r ∈ {2,4,8,16} × {dense, loop} = 8 格；每格 600 步上限、
dev_random 达 0.90 早停；每 25 步评估 dev(160)/depth(80)。

判读：对每臂拟合"r → 首次达阈步数"，并用 boot95 比较达到 0.90 所需
的最小 r。若 loop 在更小 r 达阈 → 参数效率优势；若两臂同 r → 无差异；
若稠密在小 r 更好 → 反方向证据。

输出：runs/t14_param_eff/{hist_<arm>_r<r>.jsonl, ckpt_*.pt}
      + results/t14_param_eff.json
"""
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from peft import LoraConfig, inject_adapter_in_model  # noqa: E402
from s08_arch import (EOS_ID, LO, HI, LoopUS08, call_layer,  # noqa: E402
                      inject_lora08, load_s08_bf16, load_tok)
from s48_t5_eos_patch_train import append_jsonl, gpu_free  # noqa: E402
from s59_t13_gpu_pilot import (CTX, N_LABELS, last_logits,  # noqa: E402
                               load_rows)

DATA = ROOT / 'data' / 'synth_multihop'
RUN = ROOT / 'runs' / 't14_param_eff'
OUT = ROOT / 'results' / 't14_param_eff.json'

SEED = 2029
LR = 5e-5
BPS = 4
STEPS = 200
EVAL_EVERY = 50
THRESH = 0.90
R_GRID = (1, 2)
ALL_LAYERS = list(range(24))
M_LAYERS = list(range(LO, HI))


def layer_targets(backbone, layers):
    """选定物理层范围内的 LoRA target 模块名。"""
    names = []
    for name, _ in backbone.named_modules():
        parts = name.split('.')
        if (len(parts) >= 3 and parts[0] == 'model'
                and parts[1] == 'layers' and parts[2].isdigit()
                and int(parts[2]) in layers
                and name.endswith(('q_proj', 'k_proj', 'v_proj', 'o_proj',
                                   'gate_proj', 'up_proj', 'down_proj',
                                   'in_proj_qkv', 'in_proj_z', 'in_proj_b',
                                   'out_proj'))):
            names.append(name)
    return names


def _label_index(token_id, digit_ids):
    return digit_ids.index(token_id)


@torch.no_grad()
def evaluate_arm(model, tok, rows, depth, bypass, digit_ids, bs=8):
    """批量评估（本脚本自带，不依赖 s59 的模块级全局）。"""
    hits = []
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        pids, ans = [], []
        for row in chunk:
            p_ids = tok.encode(row['question'])
            a = tok.encode(' ' + row['answer'])[-1]
            pids.append(p_ids[-CTX:])
            ans.append(_label_index(a, digit_ids))
        L = max(len(p) for p in pids)
        xb = torch.zeros((len(pids), L), dtype=torch.long, device='cuda')
        for i, p in enumerate(pids):
            xb[i, L - len(p):] = torch.tensor(p, device='cuda')
        lg = last_logits(model, xb, depth, bypass)
        pred = lg[:, list(digit_ids)].argmax(-1).tolist()
        hits.extend(int(pred[i] == g) for i, g in enumerate(ans))
    return sum(hits) / len(hits), {}


def build_batch(tok, rows, digit_ids):
    xs, ys = [], []
    for row in rows:
        p_ids = tok.encode(row['question'])
        a = tok.encode(' ' + row['answer'])[-1]
        xs.append(p_ids[-CTX:])
        ys.append(_label_index(a, digit_ids))
    L = max(len(x) for x in xs)
    xb = torch.zeros((len(xs), L), dtype=torch.long, device='cuda')
    for i, x in enumerate(xs):
        xb[i, L - len(x):] = torch.tensor(x, device='cuda')
    return xb, torch.tensor(ys, device='cuda')


def build_arm(arm, r):
    """构造 (model, trainables, depth, bypass, n_trainable, layer_span)。"""
    backbone = load_s08_bf16()
    for p in backbone.parameters():
        p.requires_grad_(False)
    model = LoopUS08(backbone, LO, HI)
    if arm == 'loop':
        ck = torch.load(ROOT / 'runs/s08_armB/ckpt_latest.pt',
                        map_location='cuda', weights_only=False)
        prior, _ = inject_lora08(backbone, LO, HI, ck['args']['lora_r'],
                                 dropout=0.0)
        with torch.no_grad():
            for p, saved in zip(prior, ck['model']['lora']):
                p.data.copy_(saved.to(p.device))
        model.l1gate.load_state_dict(ck['model']['gate'])
        for p in model.l1gate.parameters():
            p.requires_grad_(False)
        for p in prior:
            p.requires_grad_(False)
        span = M_LAYERS
    else:
        span = ALL_LAYERS
    targets = layer_targets(backbone, span)
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias='none',
                     target_modules=targets)
    inject_adapter_in_model(cfg, backbone, adapter_name='fresh')
    fresh = [p for n, p in backbone.named_parameters()
             if 'lora_' in n and 'fresh' in n]
    for p in fresh:
        p.requires_grad_(True)
    for mod in backbone.modules():
        if hasattr(mod, 'set_adapter'):
            try:
                mod.set_adapter(['prior', 'fresh'] if arm == 'loop'
                                else 'fresh')
            except Exception:
                pass
    depth, bypass = (2, False) if arm == 'loop' else (1, True)
    return (model, fresh, depth, bypass,
            sum(p.numel() for p in fresh),
            {'layers': [span[0], span[-1] + 1], 'n_layers': len(span),
             'n_target_modules': len(targets)})


def train_cell(tok, arm, r, digit_ids, train_rows, evals):
    model, fresh, depth, bypass, n_train, span = build_arm(arm, r)
    opt = torch.optim.AdamW(fresh, lr=LR)
    hp = RUN / f'hist_{arm}_r{r}.jsonl'
    ck_path = RUN / f'ckpt_{arm}_r{r}.pt'
    hist = []
    step0 = 0
    if ck_path.exists():
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        with torch.no_grad():
            for p, saved in zip(fresh, ck['fresh']):
                p.data.copy_(saved.to(p.device))
        opt.load_state_dict(ck['opt'])
        hist = ck['hist']
        step0 = ck['step']
        print(f'  [{arm} r={r}] resume step {step0}', flush=True)

    append_jsonl(hp, {'event': 'start', 'arm': arm, 'r': r,
                      'n_trainable': n_train, 'span': span,
                      'depth': depth, 'bypass': bypass, 'lr': LR,
                      'bps': BPS, 'ctx': CTX, 'steps_max': STEPS,
                      'train_n': len(train_rows),
                      'ts': time.strftime('%F %T')})
    print(f'=== {arm} r={r}: {n_train:,} trainable, '
          f'{span["n_layers"]} layers, {span["n_target_modules"]} modules',
          flush=True)

    model.train()
    t0 = time.time()
    step = step0
    while step < STEPS:
        batch = random.sample(train_rows, BPS)
        xb, yb = build_batch(tok, batch, digit_ids)
        lg = last_logits(model, xb, depth, bypass)
        sl = lg.float().index_select(-1, torch.tensor(digit_ids,
                                                     device='cuda'))
        loss = F.cross_entropy(sl, yb)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(fresh, 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % EVAL_EVERY == 0:
            model.eval()
            rec = {'step': step, 'loss': round(float(loss.detach()), 4),
                   'grad_norm': round(gn, 3),
                   'sec': round(time.time() - t0, 1)}
            for name, rows in evals.items():
                acc, per_hop = evaluate_arm(model, tok, rows, depth, bypass,
                                            digit_ids)
                rec[name] = round(acc, 4)
                rec[name + '_by_hop'] = per_hop
            append_jsonl(hp, {'event': 'eval', **rec,
                              'ts': time.strftime('%F %T')})
            hist.append(rec)
            print(f'  [{arm} r={r}] step {step} loss={loss.item():.3f} '
                  f'dev={rec["dev_random"]:.3f} '
                  f'depth={rec["confirm_depth"]:.3f} '
                  f'({rec["sec"]}s)', flush=True)
            torch.save({'arm': arm, 'r': r, 'step': step,
                        'fresh': [p.detach().cpu() for p in fresh],
                        'opt': opt.state_dict(), 'hist': hist,
                        'n_trainable': n_train, 'span': span},
                       ck_path)
            model.train()
            if rec['dev_random'] >= 0.95:
                print(f'  [{arm} r={r}] early stop at {step}', flush=True)
                break
    first90 = next((h['step'] for h in hist if h['dev_random'] >= THRESH),
                   None)
    append_jsonl(hp, {'event': 'done', 'first_90': first90,
                      'final_dev': hist[-1]['dev_random'] if hist else None,
                      'ts': time.strftime('%F %T')})
    del model, fresh
    torch.cuda.empty_cache()
    return {'arm': arm, 'r': r, 'n_trainable': n_train, 'span': span,
            'depth': depth, 'hist': hist, 'first_90': first90,
            'final_dev': hist[-1]['dev_random'] if hist else None,
            'final_depth': hist[-1]['confirm_depth'] if hist else None}


def main():
    ok, utils = gpu_free()
    RUN.mkdir(parents=True, exist_ok=True)
    hp = RUN / 'history.jsonl'
    if ok is False:
        append_jsonl(hp, {'event': 'abort', 'reason': 'external gpu load',
                          'utils': utils, 'ts': time.strftime('%F %T')})
        print('ABORT: external GPU load', utils, flush=True)
        return 2
    torch.manual_seed(SEED)
    random.seed(SEED)
    tok = load_tok()
    digit_ids = [tok.encode(' ' + str(i + 1))[-1] for i in range(N_LABELS)]
    train_rows = load_rows('train')
    evals = {'dev_random': load_rows('dev_random', 100),
             'confirm_depth': load_rows('confirm_depth', 40)}
    append_jsonl(hp, {'event': 'start', 'phase': 'T-14 param efficiency',
                      'grid': [f'{a}_r{r}' for a in ('dense', 'loop')
                               for r in R_GRID],
                      'task': 'hop2 base skill + depth extrapolation',
                      'train_n': len(train_rows),
                      'gpu_utils': utils, 'ts': time.strftime('%F %T')})

    res = []
    for r in R_GRID:
        for arm in ('dense', 'loop'):
            res.append(train_cell(tok, arm, r, digit_ids, train_rows, evals))
            OUT.write_text(json.dumps({'curves': res,
                                       'note': 'running'},
                                      ensure_ascii=False, indent=1),
                           encoding='utf-8')

    # 判读：每臂达到 THRESH 所需的最小 r
    summary = {}
    for arm in ('dense', 'loop'):
        cells = [c for c in res if c['arm'] == arm]
        reached = [c['r'] for c in cells if c['first_90'] is not None]
        summary[arm] = {
            'min_r_reached': min(reached) if reached else None,
            'cells': [{'r': c['r'], 'n_trainable': c['n_trainable'],
                       'first_90': c['first_90'],
                       'final_dev': c['final_dev'],
                       'final_depth': c['final_depth'],
                       'span': c['span']} for c in cells],
        }
    verdict = ('loop_more_param_efficient'
               if summary['loop']['min_r_reached'] is not None
               and (summary['dense']['min_r_reached'] is None
                    or summary['loop']['min_r_reached']
                    < summary['dense']['min_r_reached'])
               else 'no_difference'
               if summary['loop']['min_r_reached']
               == summary['dense']['min_r_reached']
               else 'dense_more_param_efficient')
    report = {'config': {'lr': LR, 'bps': BPS, 'ctx': CTX,
                         'steps_max': STEPS, 'threshold': THRESH,
                         'grid': list(R_GRID),
                         'task': 'synth_multihop hop2 (T-13 v2)',
                         'note': 'dense 臂覆盖全 24 层；loop 臂覆盖 M 区 12 层'
                                 '（可训练范围有意不同）'},
              'summary': summary, 'verdict': verdict,
              'curves': res, 'ts': time.strftime('%F %T')}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    append_jsonl(hp, {'event': 'final', 'verdict': verdict,
                      'summary': {k: v['min_r_reached']
                                  for k, v in summary.items()},
                      'ts': time.strftime('%F %T')})
    print('T-14 done verdict =', verdict, '->', OUT, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
