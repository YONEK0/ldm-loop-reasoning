"""s74_ldm_residual_diag.py — 残差膨胀/梯度震荡诊断（D1，文献 arXiv:2605.18797）。

对照"成功解出全难度的 seed"与"失败 seed"的检查点，回答：
  1. 残差范数 ||h_raw|| 是否随轮次单调膨胀（residual explosion）？
  2. h0 重注入占比 ||h_raw||/||h0|| 是否随轮次线性增长？
  3. 训练时首层 FFN 梯度范数是否震荡（gradient oscillation）？

用法：
  python s74_ldm_residual_diag.py --ckpts results/iter4_v7_exactT_s41_ckpt.pt:seed41_OK \
      results/iter4_v7_exactT_s53_ckpt.pt:seed53_FAIL --out results/s74_residual.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import s63_ldm_model as M

ROOT = Path(__file__).parent
DEV_DIR = ROOT / 'data' / 'ldm_mvt5'


def load_model(ckpt, dev):
    m = M.build_model('loop', n_rounds=8, n_core=1, round_in_core=True,
                      pos='none').to(dev)
    sd = torch.load(ckpt, map_location=dev)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    m.load_state_dict(sd)
    m.eval()
    return m


def diag_forward(model, ds, dev, bs=256):
    """逐轮残差范数（评测模式，no_grad）。"""
    order = list(range(len(ds)))
    aggs = None
    n = 0
    with torch.no_grad():
        for i in range(0, len(order), bs):
            b = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in ds.collate(order[i:i + bs]).items()}
            out = model(b['ids'], b['pad_mask'], b['cand_pos'],
                        b['cand_mask'], n_rounds=8, return_diag=True)
            d = out['diag']
            if aggs is None:
                aggs = {'h0_norm': [], 'round_norm': [[] for _ in d['round_norm']],
                        'h0_share': [[] for _ in d['h0_share']]}
            aggs['h0_norm'].append(d['h0_norm'])
            for t, v in enumerate(d['round_norm']):
                aggs['round_norm'][t].append(v)
            for t, v in enumerate(d['h0_share']):
                aggs['h0_share'][t].append(v)
            n += 1
    res = {
        'h0_norm': sum(aggs['h0_norm']) / len(aggs['h0_norm']),
        'round_norm': [sum(x) / len(x) for x in aggs['round_norm']],
        'h0_share': [sum(x) / len(x) for x in aggs['h0_share']],
    }
    rn = res['round_norm']
    res['growth_ratio'] = rn[-1] / max(rn[0], 1e-9)
    res['monotone'] = bool(all(rn[i] <= rn[i + 1] * 1.02
                               for i in range(len(rn) - 1)))
    return res


def grad_probe(model, ds, dev, bs=32, n_batch=8):
    """训练态下首层 FFN 梯度 L2 范数（gradient oscillation 指标）。"""
    model.train()
    order = list(range(min(len(ds), bs * n_batch)))
    gnorms = []
    for i in range(0, len(order), bs):
        b = {k: (v.to(dev) if torch.is_tensor(v) else v)
             for k, v in ds.collate(order[i:i + bs]).items()}
        out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                    n_rounds=8)
        lg = out['final_logits'].masked_fill(~b['cand_mask'], -1e9)
        loss = F.cross_entropy(lg, b['label'])
        model.zero_grad(set_to_none=True)
        loss.backward()
        g = model.core[0].up.weight.grad     # Block.up = FFN 第一层
        if g is None:
            g = next(model.core.parameters()).grad
        gnorms.append(float(g.norm()))
        model.zero_grad(set_to_none=True)
    model.eval()
    mean = sum(gnorms) / len(gnorms)
    var = sum((x - mean) ** 2 for x in gnorms) / len(gnorms)
    return {'g_norms': gnorms, 'g_mean': mean,
            'g_std': var ** 0.5,
            'g_cv': (var ** 0.5) / max(mean, 1e-12)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True,
                    help='path:label 形式，如 results/x_ckpt.pt:seed41_OK')
    ap.add_argument('--out', default='results/s74_residual.json')
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = [json.loads(l) for l in
            (DEV_DIR / 'dev_random.jsonl').read_text(
                encoding='utf-8').splitlines() if l.strip()]
    ds = M.EncodedDataset(rows, max_len=256, fused=True)

    payload = {'n_dev': len(ds), 'runs': {}}
    for spec in args.ckpts:
        path, _, label = spec.rpartition(':')
        if not path:
            path, label = spec, Path(spec).stem
        p = Path(path)
        if not p.exists():
            payload['runs'][label] = {'missing': True, 'path': str(p)}
            continue
        m = load_model(p, dev)
        r = diag_forward(m, ds, dev)
        r.update(grad_probe(m, ds, dev))
        r['path'] = str(p)
        payload['runs'][label] = r
        print(json.dumps({'label': label,
                          'h0_norm': round(r['h0_norm'], 3),
                          'round_norm': [round(x, 2) for x in r['round_norm']],
                          'growth_ratio': round(r['growth_ratio'], 2),
                          'monotone': r['monotone'],
                          'g_mean': round(r['g_mean'], 4),
                          'g_cv': round(r['g_cv'], 3)}), flush=True)

    Path(args.out).write_text(json.dumps(payload, indent=2) + '\n',
                              encoding='utf-8')
    print(json.dumps({'done': args.out}), flush=True)


if __name__ == '__main__':
    main()
