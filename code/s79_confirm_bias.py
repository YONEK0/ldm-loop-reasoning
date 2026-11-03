"""s79_confirm_bias.py — hop7/8 外推失败的偏置诊断（H-015，只读）。

回答"外推失败是模型不会，还是 confirm 集有偏置"：

  1. 标签平衡：confirm 集 answer_index 分布
  2. 位置偏好：模型预测 index=0/1 的比例（与 50% 比）
  3. 候选互换：把两候选对调后重测
       - 预测跟着内容翻转 → content-following（正常）
       - 预测跟着位置不动 → position-following（位置偏置）
  4. 干扰项选择率：预测到非答案候选的比例（hop7 低于随机说明偏好干扰项）
  5. 对照：同模型在 dev_random（hop2-6）上的同样统计

用法：
  python s79_confirm_bias.py --ckpt results/R_Q1_extrap_s53_ckpt.pt --out results/bias_s53.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _pick_device():
    """设备选择：优先 NPU（昇腾 910B），其次 CUDA，最后 CPU。"""
    import torch
    if hasattr(torch, 'npu') and getattr(torch.npu, 'is_available',
                                          lambda: False)():
        return torch.device('npu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

import s63_ldm_model as M

ROOT = Path(__file__).parent
DEV = ROOT / 'data' / 'ldm_mvt5'


def load_model(ckpt, dev):
    m = M.build_model('loop', n_rounds=8, n_core=1, round_in_core=True,
                      pos='none', step_sup=True).to(dev)
    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    # 容错：checkpoint 可能不含 step_head（早期 run）
    missing, unexpected = m.load_state_dict(sd, strict=False)
    m.eval()
    return m, [k for k in missing if 'step_head' not in k]


def swap_rows(rows):
    """候选对调 + 标签翻转。"""
    out = []
    for r in rows:
        c = list(r['candidates'])
        assert len(c) == 2, 'only 2-candidate sets supported'
        r2 = dict(r)
        r2['candidates'] = [c[1], c[0]]
        r2['answer_index'] = 1 - int(r['answer_index'])
        out.append(r2)
    return out


@torch.no_grad()
def predict(model, rows, dev, bs=256, r=8):
    ds = M.EncodedDataset(rows, max_len=256, fused=True)
    preds = []
    for i in range(0, len(ds), bs):
        b = {k: (v.to(dev) if torch.is_tensor(v) else v)
             for k, v in ds.collate(list(range(i, min(i + bs, len(ds))))).items()}
        out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                    n_rounds=r)
        lg = out['final_logits'].masked_fill(~b['cand_mask'], -1e9)
        preds.extend(lg.argmax(1).tolist())
    return preds


def load_rows(name):
    p = DEV / name
    return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()
            if l.strip()]


def analyze(rows, model, dev, label):
    labels = [int(r['answer_index']) for r in rows]
    n = len(rows)
    pred = predict(model, rows, dev, r=8)
    pred_sw = predict(model, swap_rows(rows), dev, r=8)
    acc = sum(p == y for p, y in zip(pred, labels)) / n
    # 位置偏好
    pos0 = sum(1 for p in pred if p == 0) / n
    # 内容跟随 vs 位置跟随（互换后：内容跟随 = 预测翻转）
    flip = sum(1 for p, q in zip(pred, pred_sw) if q != p) / n
    # 干扰项选择率（预测 ≠ 标签）
    distractor = 1.0 - acc
    # 按 hop 分组
    by_hop = {}
    for r, p, q in zip(rows, pred, pred_sw):
        h = str(r['hop'])
        d = by_hop.setdefault(h, {'n': 0, 'ok': 0, 'pos0': 0, 'flip': 0})
        d['n'] += 1
        d['ok'] += int(p == int(r['answer_index']))
        d['pos0'] += int(p == 0)
        d['flip'] += int(q != p)
    hop_stats = {h: {'n': d['n'],
                     'acc': round(d['ok'] / d['n'], 3),
                     'pos0_rate': round(d['pos0'] / d['n'], 3),
                     'content_follow': round(d['flip'] / d['n'], 3)}
                 for h, d in sorted(by_hop.items())}
    return {
        'label': label,
        'n': n,
        'label_balance_idx1': round(sum(labels) / n, 3),
        'acc': round(acc, 4),
        'pos0_rate': round(pos0, 3),
        'content_follow_rate': round(flip, 3),
        'distractor_rate': round(distractor, 3),
        'by_hop': hop_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', required=True)
    ap.add_argument('--out', default='results/confirm_bias.json')
    ap.add_argument('--r', type=int, default=8)
    args = ap.parse_args()

    dev = _pick_device()
    confirm = load_rows('confirm_depth.jsonl')
    devrows = load_rows('dev_random.jsonl')

    payload = {'confirm_n': len(confirm), 'dev_n': len(devrows), 'runs': []}
    for ck in args.ckpt:
        p = Path(ck)
        if not p.exists():
            payload['runs'].append({'ckpt': ck, 'missing': True})
            continue
        model, missing = load_model(p, dev)
        rec = {'ckpt': str(p), 'missing_keys_nonstep': missing[:5]}
        rec['confirm'] = analyze(confirm, model, dev, 'confirm_hop7_8')
        rec['dev_ref'] = analyze(devrows, model, dev, 'dev_hop2_6')
        payload['runs'].append(rec)
        print(json.dumps({'ckpt': p.name,
                          'confirm_acc': rec['confirm']['acc'],
                          'confirm_pos0': rec['confirm']['pos0_rate'],
                          'confirm_content_follow':
                              rec['confirm']['content_follow_rate'],
                          'confirm_by_hop': rec['confirm']['by_hop'],
                          'dev_acc': rec['dev_ref']['acc']},
                         default=float), flush=True)

    Path(args.out).write_text(json.dumps(payload, indent=2) + '\n',
                              encoding='utf-8')
    print(json.dumps({'done': args.out}), flush=True)


if __name__ == '__main__':
    main()
