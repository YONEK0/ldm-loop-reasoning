"""s65: GPU 冒烟/诊断——过拟合定位 + 短程学习信号（正式实验前置）。

两步，均在 GPU（本机 RTX 5060 Laptop 8GB，模型 ~1.9M 参数，占用极小）：
  A. overfit 诊断：固定 32/64 条训练样本，看 loss 能否降到 ~0。
     不能 -> 结构性 bug（掩码/分词/梯度路径/初始化）。
  B. 短程学习曲线：全 train 子集 + dev_random，比较 loop r=1 / r=4 /
     dense mid=2 的学习信号强弱。

纪律：只用 train 与 dev_random；confirm_* 不参与。这是冒烟诊断，
不构成可行性或效率结论。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import s63_ldm_model as M

ROOT = Path(__file__).parent
OUT_OVERFIT = ROOT / 'results' / 'ldm_gpu_overfit.json'
OUT_CURVE = ROOT / 'results' / 'ldm_gpu_smoke.json'


def get_device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def to_dev(batch: dict, dev: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev, non_blocking=True)
        else:
            out[k] = v
    return out


def forward_loss(model, batch, n_rounds=None):
    out = model(batch['ids'], batch['pad_mask'],
                batch['cand_pos'], batch['cand_mask'],
                n_rounds=n_rounds)
    logits = out['final_logits'].masked_fill(~batch['cand_mask'], -1e9)
    loss = F.cross_entropy(logits, batch['label'])
    return loss, logits, out


def run_overfit(args, dev):
    """A: 在固定小样本上过拟合。若不能 -> 结构性 bug。"""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rows = M.load_jsonl(M.DATA_DIR / 'train.jsonl')
    rows = random.Random(args.seed).sample(rows, args.overfit_n)
    ds = M.EncodedDataset(rows, max_len=args.max_len, fused=args.fused)
    b = to_dev(ds.collate(list(range(len(ds)))), dev)
    results = []
    for variant, depth, zero_init in [
            ('loop', 4, True), ('loop', 4, False), ('dense', 2, True)]:
        torch.manual_seed(args.seed)
        model = M.build_model(variant, n_rounds=depth,
                              n_dense_mid=depth if variant == 'dense' else 2,
                              d=args.d, heads=args.heads, ffn=args.ffn,
                              zero_init_loop=zero_init).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=0.0)
        hist = []
        t0 = time.perf_counter()
        for s in range(1, args.overfit_steps + 1):
            model.train()
            loss, logits, _ = forward_loss(model, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if s % 20 == 0 or s == 1:
                acc = (logits.argmax(1) == b['label']).float().mean().item()
                hist.append({'step': s, 'loss': round(float(loss.detach()), 4),
                             'grad_norm': round(float(gn), 4),
                             'acc': round(acc, 3)})
        sec = time.perf_counter() - t0
        rec = {'variant': variant, 'depth': depth, 'zero_init': zero_init,
               'n': len(ds), 'params': model.param_report()['total'],
               'final_loss': hist[-1]['loss'], 'final_acc': hist[-1]['acc'],
               'sec': round(sec, 1), 'hist': hist}
        results.append(rec)
        print(json.dumps({'overfit': f'{variant}{depth}_zi{int(zero_init)}',
                          'loss': rec['final_loss'], 'acc': rec['final_acc'],
                          'grad': hist[-1]['grad_norm'],
                          'sec': rec['sec']}), flush=True)
    return results


def evaluate(model, ds, bs, dev, max_len, n_rounds=None):
    model.eval()
    order = list(range(len(ds)))
    correct = n = 0
    by_hop: dict[int, list[int]] = {}
    surv_hits: dict[int, list[float]] = {}
    with torch.no_grad():
        for i in range(0, len(order), bs):
            b = to_dev(ds.collate(order[i:i + bs]), dev)
            _, logits, out = forward_loss(model, b, n_rounds=n_rounds)
            ok = (logits.argmax(1) == b['label'])
            correct += int(ok.sum())
            n += len(ok)
            for h, o in zip(b['hop'].tolist(), ok.tolist()):
                by_hop.setdefault(h, []).append(int(o))
            for t, rd in enumerate(out.get('rounds', []), start=1):
                if 'surv' not in rd:
                    continue
                alive = ((b['chain_len'] >= t) & (b['chain_len'] >= 0))
                m = b['cand_mask']
                hit = (((rd['surv'] > 0) == alive) & m)
                surv_hits.setdefault(t, []).append(
                    float(hit.sum()) / max(1.0, float(m.sum())))
    res = {'acc': round(correct / max(n, 1), 4), 'n': n,
           'by_hop': {str(h): round(sum(v) / len(v), 3)
                      for h, v in sorted(by_hop.items())}}
    if surv_hits:
        res['surv_acc_by_round'] = {
            str(t): round(sum(v) / len(v), 3)
            for t, v in sorted(surv_hits.items())}
    return res


def run_curve(args, dev):
    """B: 全 train 子集短程曲线，比较深度与循环。"""
    train_rows = M.load_jsonl(M.DATA_DIR / 'train.jsonl')
    if args.train_subset:
        train_rows = random.Random(args.seed).sample(
            train_rows, min(args.train_subset, len(train_rows)))
    tr_ds = M.EncodedDataset(train_rows, max_len=args.max_len, fused=args.fused)
    remap_rng = random.Random(args.seed + 777)
    pool_n = 300
    dv_ds = M.EncodedDataset(
        M.load_jsonl(M.DATA_DIR / 'dev_random.jsonl'), max_len=args.max_len,
        fused=args.fused)
    print(json.dumps({'train_n': len(tr_ds), 'dev_n': len(dv_ds),
                      'truncated': [tr_ds.truncated, dv_ds.truncated]}),
          flush=True)

    arms = []
    for spec in args.arms.split(','):
        k, v = spec.split(':')
        arms.append((k, int(v)))

    results = []
    for variant, depth in arms:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        model = M.build_model(
            variant, n_rounds=depth,
            n_dense_mid=depth if variant == 'dense' else 2,
            d=args.d, heads=args.heads, ffn=args.ffn,
            n_prelude=args.n_prelude, n_core=args.n_core,
            n_coda=args.n_coda,
            round_in_core=args.round_in_core, pos=args.pos,
            attn_mode=args.attn_mode, local_w=args.local_w,
            bottleneck_dim=args.bottleneck_dim).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.wd, betas=(0.9, 0.95))
        if args.sched == 'oncycle':
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.15)
        else:   # 'const'：warmup 后恒定——grokking 类任务的文献配方
            warm = max(1, args.steps // 50)

            def _lr_fn(s):
                return min(1.0, s / warm)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_fn)
        order = list(range(len(tr_ds)))
        rng = random.Random(args.seed)
        rng.shuffle(order)
        batches = [order[i:i + args.bs]
                   for i in range(0, len(order), args.bs)]
        if args.exact_t:
            hop_of = [it['hop'] for it in tr_ds.items]
            hop_groups = {}
            for i in order:
                hop_groups.setdefault(hop_of[i], []).append(i)
            batches = []
            for h, idxs in sorted(hop_groups.items()):
                rng.shuffle(idxs)
                for i in range(0, len(idxs), args.bs):
                    batches.append((idxs[i:i + args.bs], h - 1))
        else:
            batches = [(b, None) for b in batches]
        hist = []
        step = 0
        epoch_len = max(1, len(batches))
        t0 = time.perf_counter()
        model.train()
        while step < args.steps:
            if args.remap_epoch and step % epoch_len == 0:
                perm = remap_rng.sample(range(pool_n), pool_n)
                tr_ds = M.EncodedDataset(train_rows, max_len=args.max_len,
                                         fused=args.fused, perm=perm)
            if args.curriculum:
                hops = [int(h) for h in args.curriculum.split(',')]
                stage = min(len(hops) - 1,
                            step * len(hops) // max(1, args.steps))
                if getattr(run_curve, '_cur_stage', None) != stage:
                    run_curve._cur_stage = stage
                    idxs = [i for i, r in enumerate(train_rows)
                            if r['hop'] <= hops[stage]]
                    rng.shuffle(idxs)
                    batches = [idxs[i:i + args.bs]
                               for i in range(0, len(idxs), args.bs)]
                    epoch_len = max(1, len(batches))
            rng.shuffle(batches)
            for idx, hop_R in batches:
                if step >= args.steps:
                    break
                b = to_dev(tr_ds.collate(idx), dev)
                R_dyn = None
                if args.exact_t and variant == 'loop' and hop_R:
                    R_dyn = min(hop_R, depth)
                elif args.dyn_depth and variant == 'loop':
                    R_dyn = rng.randint(args.dyn_min, depth)
                loss, logits, out = forward_loss(model, b, n_rounds=R_dyn)
                if args.process_sup and variant == 'loop':
                    cl = b['chain_len']
                    cm = b['cand_mask']
                    pl = torch.zeros((), device=loss.device)
                    n_t = 0
                    for t, rd in enumerate(out['rounds'], start=1):
                        alive = ((cl >= t) & (cl >= 0)).float()
                        bce = F.binary_cross_entropy_with_logits(
                            rd['surv'], alive, reduction='none')
                        m = cm.float()
                        pl = pl + (bce * m).sum() / m.sum().clamp_min(1.0)
                        n_t += 1
                    loss = loss + args.beta * pl / max(1, n_t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    args.clip)
                opt.step()
                sched.step()
                step += 1
                if step % args.log_every == 0 or step == args.steps:
                    rec = {'step': step,
                           'loss': round(float(loss.detach()), 4),
                           'grad_norm': round(float(gn), 4),
                           'sec': round(time.perf_counter() - t0, 1)}
                    if step % args.eval_every == 0 or step == args.steps:
                        rec['dev'] = evaluate(model, dv_ds, args.eval_bs,
                                              dev, args.max_len)
                    hist.append(rec)
                    print(json.dumps({'arm': f'{variant}:{depth}', **rec}),
                          flush=True)
        final = evaluate(model, dv_ds, args.eval_bs, dev, args.max_len)
        final_evals = {'dev_random': final}
        for name in args.eval_splits.split(','):
            rows = (M.load_jsonl(M.DATA_DIR / f'{name}.jsonl')
                    if name != 'dev_random' else None)
            ds2 = (dv_ds if rows is None else
                   M.EncodedDataset(rows, max_len=args.max_len,
                                    fused=args.fused))
            if variant == 'loop' and args.eval_rounds:
                for R in args.eval_rounds.split(','):
                    key = f'{name}_r{R}'
                    final_evals[key] = evaluate(model, ds2, args.eval_bs,
                                                dev, args.max_len,
                                                n_rounds=int(R))
            else:
                if name != 'dev_random':
                    final_evals[name] = evaluate(model, ds2, args.eval_bs,
                                                 dev, args.max_len)
        results.append({'variant': variant, 'depth': depth,
                        'params': model.param_report(), 'final_dev': final,
                        'final_evals': final_evals,
                        'hist': hist,
                        'train_sec': round(time.perf_counter() - t0, 1)})
        print(json.dumps({'arm_done': f'{variant}:{depth}',
                          'final_dev': final['acc'],
                          'final_evals': {k: v['acc']
                                          for k, v in final_evals.items()},
                          'params': model.param_report()['total'],
                          'sec': round(time.perf_counter() - t0, 1)}),
              flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='all',
                    choices=['overfit', 'curve', 'all'])
    ap.add_argument('--device', default='auto')
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--overfit-steps', type=int, default=120)
    ap.add_argument('--overfit-n', type=int, default=32)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--eval-bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--d', type=int, default=192)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--ffn', type=int, default=768)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--train-subset', type=int, default=1200)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--arms', default='loop:1,loop:4,dense:2')
    ap.add_argument('--sched', default='oncycle', choices=['oncycle', 'const'])
    ap.add_argument('--fused', action='store_true',
                    help='融合编码：每条事实一个位置（见 s63 encode_example_fused）')
    ap.add_argument('--dyn-depth', action='store_true',
                    help='D8：训练时每批均匀采样轮数 R∈[dyn_min, depth]')
    ap.add_argument('--dyn-min', type=int, default=1)
    ap.add_argument('--eval-rounds', default='',
                    help='loop 臂终评的固定轮数列表，如 1,2,4,6')
    ap.add_argument('--exact-t', action='store_true',
                    help='Fan 配方：按 hop 分批，每批轮数 R=hop-1（弃 D8）')
    ap.add_argument('--bottleneck-dim', type=int, default=0,
                    help='>0 启用轮间信息瓶颈（承重结构），跨轮只传该维数')
    ap.add_argument('--pos', default='sin', choices=['sin', 'rope', 'none'])
    ap.add_argument('--attn-mode', default='full',
                    choices=['full', 'causal', 'local'])
    ap.add_argument('--local-w', type=int, default=4)
    ap.add_argument('--n-core', type=int, default=2)
    ap.add_argument('--n-prelude', type=int, default=1)
    ap.add_argument('--n-coda', type=int, default=1)
    ap.add_argument('--process-sup', action='store_true',
                    help='过程监督：逐轮候选存活位 BCE（第二刀）')
    ap.add_argument('--beta', type=float, default=0.5,
                    help='过程监督损失权重')
    ap.add_argument('--round-in-core', action='store_true',
                    help='core 输入注入轮次嵌入（第三刀）')
    ap.add_argument('--remap-epoch', action='store_true',
                    help='记忆捷径消融：训练时每 epoch 重掷实体置换'
                         '（评测不置换）')
    ap.add_argument('--curriculum', default='',
                    help='课程：逗号分隔 hop 序列，按训练步数推进'
                         '（如 2,3,4 表示前1/3只hop2、中1/3 hop2-3、后1/3 hop2-4）')
    ap.add_argument('--eval-splits', default='dev_random',
                    help='终评切分，逗号分隔（如 dev_random,confirm_depth）')
    ap.add_argument('--data-dir', default='',
                    help='数据目录（默认 s63 内置 ldm_edge_action）')
    ap.add_argument('--out-overfit', default=str(OUT_OVERFIT))
    ap.add_argument('--out-curve', default=str(OUT_CURVE))
    args = ap.parse_args()

    dev = get_device(args.device)
    if args.data_dir:
        M.DATA_DIR = Path(args.data_dir)
    print(json.dumps({'device': str(dev),
                      'data_dir': str(M.DATA_DIR),
                      'gpu': torch.cuda.get_device_name(0)
                      if dev.type == 'cuda' else None}), flush=True)
    if dev.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True

    payload = {'args': vars(args), 'device': str(dev)}
    if args.mode in ('overfit', 'all'):
        payload['overfit'] = run_overfit(args, dev)
        Path(args.out_overfit).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
    if args.mode in ('curve', 'all'):
        payload['curve'] = run_curve(args, dev)
        Path(args.out_curve).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8')
    print('gpu smoke done', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
