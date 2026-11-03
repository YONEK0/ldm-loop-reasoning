"""s75_bench_workload.py — 测量当前 LDM 任务到底是 GPU 受限还是 CPU/launch 受限。

判据：若 bs 从 64 增到 256（4×）时每样本耗时基本不变 → 不是算力受限，
     换更强 GPU 无收益，应看并行路数与显存容量。
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import s63_ldm_model as M
from s66_ldm_mvt_gen import gen_example

ROOT = Path(__file__).parent
DEV_DIR = ROOT / 'data' / 'ldm_mvt5'


def make_batch(rng, bs, cap, dev):
    rows = []
    while len(rows) < bs:
        ex = gen_example(rng, 'train', rng.randint(2, cap), comp_mode='deep')
        if ex is not None:
            rows.append(ex)
    ds = M.EncodedDataset(rows, max_len=256, fused=True)
    return {k: (v.to(dev) if torch.is_tensor(v) else v)
            for k, v in ds.collate(list(range(bs))).items()}


def bench(bs, steps, dev, grad=True):
    torch.manual_seed(0)
    rng = random.Random(0)
    model = M.build_model('loop', n_rounds=8, n_core=1, round_in_core=True,
                          pos='none').to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    # warmup
    for _ in range(5):
        b = make_batch(rng, bs, 6, dev)
        out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                    n_rounds=8)
        loss = F.cross_entropy(
            out['final_logits'].masked_fill(~b['cand_mask'], -1e9), b['label'])
        if grad:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        b = make_batch(rng, bs, 6, dev)
        out = model(b['ids'], b['pad_mask'], b['cand_pos'], b['cand_mask'],
                    n_rounds=8)
        loss = F.cross_entropy(
            out['final_logits'].masked_fill(~b['cand_mask'], -1e9), b['label'])
        if grad:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {'bs': bs, 'steps': steps, 'sec': round(dt, 3),
            'samples_per_sec': round(bs * steps / dt, 1),
            'ms_per_sample': round(dt * 1000 / (bs * steps), 3)}


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info = {'device': str(dev),
            'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info['vram_gb'] = round(p.total_memory / 1e9, 1)
        info['sm'] = f'{p.major}.{p.minor}'
    print(json.dumps(info), flush=True)

    # 峰值显存（单路）
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    res = []
    for bs in (32, 64, 128, 256):
        try:
            r = bench(bs, 40, dev)
            res.append(r)
            print(json.dumps(r), flush=True)
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({'bs': bs, 'oom': True}), flush=True)
            torch.cuda.empty_cache()
            break
    if torch.cuda.is_available():
        info['peak_vram_gb'] = round(
            torch.cuda.max_memory_allocated() / 1e9, 2)

    # 判据：ms_per_sample 随 bs 的变化
    if len(res) >= 2:
        a, b = res[0], res[-1]
        info['scaling_ratio'] = round(a['ms_per_sample'] / b['ms_per_sample'], 2)
        info['verdict'] = ('GPU-bound（大批量更高效，强卡有用）'
                           if info['scaling_ratio'] > 1.5
                           else 'NOT GPU-bound（启动开销主导，强卡收益小）')
    Path('results/s75_bench.json').write_text(
        json.dumps({'info': info, 'sweep': res}, indent=2) + '\n',
        encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
