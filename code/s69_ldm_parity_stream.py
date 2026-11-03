"""s69: parity 流式大预算探针（方向 2 本机版，无限数据封死记忆）。

设计要点：
  - 边训边生成：每批调用 s68 的 gen() 现造样本（parity 无需求解器，
    单样本微秒级）→ 每个样本只见一次，等于无限数据，比文献"4M 固定
    样本"更严格——记忆路径彻底封死，只留算法解。
  - 复用 s63 模型栈（薄 core / D8 / round-in-core / 融合编码）。
  - 200k 步 × bs 64 ≈ 1.9 小时（5060 实测 0.0345 s/步）。
  - 每 10k 步评 dev（固定 64 槽分布）与 confirm（96 槽），并做
    轮数扫描 r∈{1,2,4,8,12,16}，捕捉 Fan 签名（精度随轮数上升）。

判读沿用 §14（dev ≥0.90 且随 r 上升 = 循环计算角色成立）。
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
from s68_ldm_parity_gen import gen as gen_parity

ROOT = Path(__file__).parent
DEV_DIR = ROOT / 'data' / 'ldm_parity'


def make_batch(rng: random.Random, bs: int, dev: torch.device,
               fused: bool = True):
    rows = [gen_parity(rng, 'train', 64, (1, 48)) for _ in range(bs)]
    ds = M.EncodedDataset(rows, max_len=256, fused=fused)
    out = {}
    for k, v in ds.collate(list(range(bs))).items():
        out[k] = v.to(dev) if torch.is_tensor(v) else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=200000)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--r-max', type=int, default=12)
    ap.add_argument('--eval-every', type=int, default=10000)
    ap.add_argument('--out', default='results/parity_stream_200k.json')
    args = ap.parse_args()

    dev_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    model = M.build_model('loop', n_rounds=args.r_max, n_core=1,
                          round_in_core=True).to(dev_)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.1, betas=(0.9, 0.95))
    warm = max(1, args.steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warm))

    from s65_ldm_gpu_smoke import evaluate as _ev  # 复用带存活头的评测
    dev_ds = M.EncodedDataset(
        [json.loads(l) for l in
         (DEV_DIR / 'dev_random.jsonl').read_text(encoding='utf-8')
         .splitlines() if l.strip()], max_len=256, fused=True)
    cf_ds = M.EncodedDataset(
        [json.loads(l) for l in
         (DEV_DIR / 'confirm_depth.jsonl').read_text(encoding='utf-8')
         .splitlines() if l.strip()], max_len=256, fused=True)

    hist = []
    t0 = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        b = make_batch(rng, args.bs, dev_)
        R = rng.randint(1, args.r_max)
        out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                    n_rounds=R)
        logits = out['final_logits'].masked_fill(~b['cand_mask'], -1e9)
        loss = F.cross_entropy(logits, b['label'])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            hist.append({'step': step,
                         'loss': round(float(loss.detach()), 4)})
        if step % args.eval_every == 0 or step == args.steps:
            ev = {}
            for name, ds, rs in (('dev', dev_ds, (1, 2, 4, 8, 12, 16)),
                                 ('confirm', cf_ds, (1, 2, 4, 8, 12, 16))):
                for r in rs:
                    key = f'{name}_r{r}'
                    ev[key] = _ev(model, ds, 256, dev_, 256,
                                  n_rounds=r)['acc']
            rec = {'step': step, 'sec': round(time.perf_counter() - t0, 1),
                   **ev}
            hist[-1].update({k: v for k, v in rec.items() if k != 'step'})
            print(json.dumps(rec), flush=True)
            Path(args.out).write_text(json.dumps(
                {'args': vars(args), 'hist': hist}, indent=2) + '\n',
                encoding='utf-8')
    print(json.dumps({'done': args.out, 'final': hist[-1]}), flush=True)


if __name__ == '__main__':
    main()
