"""s64: LDM 纯 CPU 冒烟训练——固定深度学习信号检查（非正式实验）。

目的（规格 §5.3 阶段 1 的 CPU 前置）：
  在进入任何 GPU/长程训练前，确认
    1) 数据/分词/标签管线端到端可训（loss 下降、dev 准确率上升）；
    2) 固定深度 loop 与同参数 dense 在该任务上都有学习信号；
    3) 深度（r=1 vs r=4）是否产生可测差异——这是循环机制的先行指标。

约束：
  - 纯 CPU；不加载预训练权重；不写 checkpoint 到 runs/ 之外；
  - 仅用 train 训练、dev_random 选择/报告；confirm_* 不参与（首轮冒烟
    不消耗确认集，规格 §3.2/§6.2）。

注意：这是冒烟，不是可行性结论。步数与数据量都远低于正式实验，
任何准确率数字都只用于判断“管线是否能学”，不作质量或效率主张。
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
RUN_DIR = ROOT / 'runs' / 'ldm_smoke'
OUT = ROOT / 'results' / 'ldm_smoke.json'


def build_loader(rows, max_len, bs, shuffle, seed=0):
    ds = M.EncodedDataset(rows, max_len=max_len)
    order = list(range(len(ds)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)
    batches = [order[i:i + bs] for i in range(0, len(order), bs)]
    return ds, batches


def evaluate(model, ds, batches, max_len):
    model.eval()
    n = correct = 0
    by_hop: dict[int, list[int]] = {}
    rounds_used = []
    with torch.no_grad():
        for b in batches:
            batch = ds.collate(b)
            out = model(batch['ids'], batch['pad_mask'],
                        batch['cand_pos'], batch['cand_mask'])
            logits = out['final_logits'].masked_fill(~batch['cand_mask'], -1e9)
            pred = logits.argmax(1)
            ok = (pred == batch['label'])
            correct += int(ok.sum())
            n += len(b)
            for h, o in zip(batch['hop'].tolist(), ok.tolist()):
                by_hop.setdefault(h, []).append(int(o))
            if out['rounds'] and 'q' in out['rounds'][0]:
                rounds_used.append(out['rounds'])
    model.train()
    return {'acc': correct / max(n, 1), 'n': n,
            'by_hop': {str(h): round(sum(v) / len(v), 3)
                       for h, v in sorted(by_hop.items())}}


def run_arm(variant: str, rounds: int, dense_mid: int, args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = M.build_model(variant, n_rounds=rounds, n_dense_mid=dense_mid,
                          d=args.d, heads=args.heads, ffn=args.ffn)
    rep = model.param_report()
    train_rows = M.load_jsonl(M.DATA_DIR / 'train.jsonl')
    if args.train_subset:
        rng = random.Random(args.seed)
        train_rows = rng.sample(train_rows, min(args.train_subset, len(train_rows)))
    tr_ds, tr_batches = build_loader(train_rows, args.max_len, args.bs, True, args.seed)
    dv_ds, dv_batches = build_loader(
        M.load_jsonl(M.DATA_DIR / 'dev_random.jsonl'), args.max_len, args.bs * 2, False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.15)
    hist = []
    step = 0
    t_start = time.perf_counter()
    model.train()
    while step < args.steps:
        for b in tr_batches:
            if step >= args.steps:
                break
            batch = tr_ds.collate(b)
            out = model(batch['ids'], batch['pad_mask'],
                        batch['cand_pos'], batch['cand_mask'])
            logits = out['final_logits'].masked_fill(~batch['cand_mask'], -1e9)
            loss = F.cross_entropy(logits, batch['label'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            step += 1
            if step % args.log_every == 0 or step == args.steps:
                rec = {'step': step, 'loss': round(float(loss), 4),
                       'grad_norm': round(float(gn), 3),
                       'sec': round(time.perf_counter() - t_start, 1)}
                if step % args.eval_every == 0 or step == args.steps:
                    rec['dev'] = evaluate(model, dv_ds, dv_batches, args.max_len)
                hist.append(rec)
                print(json.dumps({'arm': f'{variant}_r{rounds if variant=="loop" else dense_mid}',
                                  **rec}), flush=True)
    final = evaluate(model, dv_ds, dv_batches, args.max_len)
    return {'variant': variant,
            'depth': rounds if variant == 'loop' else dense_mid,
            'params': rep, 'final_dev': final,
            'hist': hist,
            'wall_sec': round(time.perf_counter() - t_start, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--d', type=int, default=192)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--ffn', type=int, default=768)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--train-subset', type=int, default=0)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--arms', default='loop:1,loop:4,dense:2')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    arms = []
    for spec in args.arms.split(','):
        k, v = spec.split(':')
        arms.append((k, int(v)))

    results = []
    for variant, depth in arms:
        t0 = time.perf_counter()
        if variant == 'loop':
            rec = run_arm('loop', depth, 2, args)
        else:
            rec = run_arm('dense', 1, depth, args)
        rec['arm_sec'] = round(time.perf_counter() - t0, 1)
        results.append(rec)
        print(json.dumps({'arm_done': f'{variant}_{depth}',
                          'final_dev': rec['final_dev']['acc'],
                          'params': rec['params']['total'],
                          'sec': rec['arm_sec']}), flush=True)

    payload = {'args': vars(args), 'results': results,
               'note': 'CPU 冒烟；非正式实验；confirm_* 未使用；'
                       '数字仅判管线可学性'}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False,
                                         indent=2) + '\n', encoding='utf-8')
    print('smoke done ->', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
