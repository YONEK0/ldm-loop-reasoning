"""s81_eval_rounds.py — 在"更多推理轮数"下评测（Kohli et al. 2026 的做法）。

文献依据（arXiv:2604.07822 "Loop, Think, & Generalize"）：
  固定轮数训练的循环模型，**在与训练相同的轮数下不会外推**；必须把推理轮数
  提高到训练轮数之上，外推才会出现（且需要训练轮数 R>4 作为前提）。

我们的评测一直只扫 r ∈ {1,2,3,4,5,6,8}，而训练 r_max=8 —— 即**从未在
超过训练轮数的推理预算下测过 hop7/8**。这个脚本补上：
  在 r ∈ {6,8,10,12,16} 上评测 dev(hop2-6) 与 confirm(hop7/8)，
  并记录逐 hop 精度与 logit margin（检验 overthinking：margin 应先升后降）。

用法：
  python s81_eval_rounds.py --ckpt results/R_Q1_extrap_s53_ckpt.pt \
      --out results/rounds_s53.json --rounds 6,8,10,12,16
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
import torch.nn.functional as F

import s63_ldm_model as M

ROOT = Path(__file__).parent
DEV = ROOT / 'data' / 'ldm_mvt5'


def load_model(ckpt, dev, step_sup=True):
    m = M.build_model('loop', n_rounds=16, n_core=1, round_in_core=True,
                      pos='none', step_sup=step_sup).to(dev)
    sd = torch.load(ckpt, map_location=dev, weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def load_rows(name):
    return [json.loads(l) for l in
            (DEV / name).read_text(encoding='utf-8').splitlines() if l.strip()]


@torch.no_grad()
def eval_at(model, rows, dev, r, bs=256):
    ds = M.EncodedDataset(rows, max_len=256, fused=True)
    ok_by_hop, n_by_hop = {}, {}
    margins = []
    for i in range(0, len(ds), bs):
        b = {k: (v.to(dev) if torch.is_tensor(v) else v)
             for k, v in ds.collate(
                 list(range(i, min(i + bs, len(ds))))).items()}
        out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                    n_rounds=r)
        lg = out['final_logits'].masked_fill(~b['cand_mask'], -1e9)
        pred = lg.argmax(1)
        top2 = lg.topk(2, dim=1).values
        margins.extend((top2[:, 0] - top2[:, 1]).tolist())
        for j in range(len(pred)):
            h = str(int(b['hop'][j]))
            n_by_hop[h] = n_by_hop.get(h, 0) + 1
            ok_by_hop[h] = ok_by_hop.get(h, 0) + int(pred[j] == b['label'][j])
    return {h: round(ok_by_hop[h] / n_by_hop[h], 3) for h in sorted(n_by_hop)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--rounds', default='6,8,10,12,16')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    dev = _pick_device()
    rounds = [int(x) for x in args.rounds.split(',')]
    model = load_model(args.ckpt, dev)
    confirm = load_rows('confirm_depth.jsonl')
    devrows = load_rows('dev_random.jsonl')

    payload = {'ckpt': args.ckpt, 'by_round': {}}
    for r in rounds:
        dc = eval_at(model, confirm, dev, r)
        dd = eval_at(model, devrows, dev, r)
        payload['by_round'][str(r)] = {'confirm': dc, 'dev': dd}
        print(json.dumps({'r': r, 'confirm': dc, 'dev': dd}), flush=True)

    Path(args.out).write_text(json.dumps(payload, indent=2) + '\n',
                              encoding='utf-8')
    print(json.dumps({'done': args.out}), flush=True)


if __name__ == '__main__':
    main()
