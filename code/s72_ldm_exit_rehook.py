"""s72: V3a——对角线模型上训练退出控制器（承重后的"何时停"测试）。

宿主：V1 最优 ckpt（iter4_v1_none_s18，对角线 5/5）。冻结全部骨干，
只训 exit_mlp（12,673 参数）——干净检验"控制器能否学会在对角线上停轮"。

数据：MVT-5 流式 + per-sample T（rounds_mask 冻结已停样本），soft-exit
损失（w_t·CE_t + λ·E[R]，λ 0→0.02 线性）；骨干冻结 → 各轮 probe logits
固定，仅停轮策略在学习。

硬退出评测：τ∈{0.3,0.5,0.7,0.9} 扫描，逐样本停轮后取该轮读出；
指标：精度 vs oracle 对角线、mean_rounds vs E[T]=3、vs 固定 r8 节省、
停轮-需求一致率 P(stop=T)。

预注册判据（POST_DIAGONAL_TESTPLAN V3）：硬退出 ≥oracle−2pp；
mean_rounds ≤3.5；vs r8 节省 ≥50%；一致率 ≥0.9。
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
from s66_ldm_mvt_gen import gen_example
from s67_ldm_exit import hard_exit_eval

ROOT = Path(__file__).parent
DEV_DIR = ROOT / 'data' / 'ldm_mvt5'


def make_batch(rng, bs, cap, dev_):
    rows = []
    while len(rows) < bs:
        ex = gen_example(rng, 'train', rng.randint(2, cap), comp_mode='deep')
        if ex is not None:
            rows.append(ex)
    ds = M.EncodedDataset(rows, max_len=256, fused=True)
    return {k: (v.to(dev_) if torch.is_tensor(v) else v)
            for k, v in ds.collate(list(range(len(rows)))).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=16000)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--lam-max', type=float, default=0.002)
    ap.add_argument('--warmup-frac', type=float, default=0.6)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--ckpt', default='results/iter4_v1_none_s18_ckpt.pt')
    ap.add_argument('--taus', default='0.3,0.5,0.7,0.9')
    ap.add_argument('--out', default='results/iter4_v3_exit.json')
    args = ap.parse_args()

    dev_ = torch.device('cuda')
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = M.build_model('loop', n_rounds=8, n_core=1, round_in_core=True,
                          pos='none').to(dev_)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev_))
    for p in model.parameters():
        p.requires_grad = False
    for p in model.exit_mlp.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.exit_mlp.parameters(), lr=args.lr)

    hist = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        b = make_batch(rng, args.bs, 6, dev_)
        rm = torch.zeros(b['ids'].size(0), 8, dtype=torch.bool,
                         device=dev_)
        for i, h in enumerate(b['hop'].tolist()):
            rm[i, :min(max(1, h - 1), 8)] = True
        out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                    n_rounds=8, rounds_mask=rm)
        ces = [F.cross_entropy(r['logits'].masked_fill(~b['cand_mask'], -1e9),
                               b['label']) for r in out['rounds']]
        qs = [r['q'] for r in out['rounds']]
        B = b['label'].size(0)
        s = torch.ones(B, device=dev_)
        L = torch.zeros((), device=dev_)
        E = torch.zeros(B, device=dev_)
        for t in range(8):
            w = s * (1 - qs[t]) if t < 7 else s
            L = L + w.mean() * ces[t]
            E = E + s
            if t < 7:
                s = s * qs[t]
        lam = args.lam_max * min(1.0, step / (args.steps * 0.7)) \
            if step > args.steps * 0.3 else 0.0
        loss = L + lam * E.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            hist.append({'step': step, 'loss': round(float(loss.detach()), 4),
                         'lam': round(lam, 4),
                         'E': round(float(E.mean().detach()), 3),
                         'sec': round(time.perf_counter() - t0, 1)})
            print(json.dumps(hist[-1]), flush=True)

    dev_ds = M.EncodedDataset(
        [json.loads(l) for l in (DEV_DIR / 'dev_random.jsonl')
         .read_text(encoding='utf-8').splitlines() if l.strip()],
        max_len=256, fused=True)
    taus = [float(t) for t in args.taus.split(',')]
    res = [hard_exit_eval(model, dev_ds, 256, dev_, t, 1, 8) for t in taus]
    # 停轮-需求一致率
    for r in res:
        by = {}
        for rec in r['per_sample_rounds']:
            by.setdefault(rec['hop'], []).append(rec)
        r['stop_match_rate'] = round(sum(
            1 for h, v in by.items() if v
            for rec in [None]) / 1, 3) if False else round(sum(
            (rec['round'] == min(max(1, rec['hop'] - 1), 8))
            / len(v) for h, v in by.items()) / len(by), 3)
    payload = {'args': vars(args), 'hist': hist,
               'oracle_diag_ref': 0.99, 'fixed_r8_ref': 0.974,
               'tau_sweep': res}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False,
                                         indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'done': args.out,
                      'taus': [(r['tau'], r['acc'], r['mean_rounds'],
                                r.get('stop_match_rate'))
                               for r in res]}), flush=True)


if __name__ == '__main__':
    main()
