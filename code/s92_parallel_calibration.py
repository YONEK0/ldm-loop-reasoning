"""s92_parallel_calibration.py — 实测并行度拐点，再决定开几路。

背景：这个任务的瓶颈**不是显存**。模型 1.47M、每 run 约 650 MB 显存，
而数据生成是单线程 Python（实测每进程 pcpu 恒 100%）。所以在有 192 GB
显存的机器上，真正限制并行度的是 CPU 核数。

本脚本在同一台机器上依次跑 1 / 3 / 6 / 9 路并行，每路 300 步，
报告：单 run 速度、总吞吐、加速比。据此选并行度——取吞吐曲线的拐点，
不是拍脑袋。

用法：
  python s92_parallel_calibration.py --slots 1,3,6,9 --steps 300
  python s92_parallel_calibration.py --slots 6 --resume-ok   # 只测一个档

判读：
  - 吞吐随并行度线性增长 → 还能加
  - 吞吐增长明显低于线性（<0.7×） → 已过拐点，停在前一档
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _find_dirs() -> tuple[Path, Path]:
    """定位 (代码目录, 数据目录)。

    仓库结构有两种：代码在仓库根（本地），或在 code/ 子目录（GitHub 版）。
    数据相应在 data/ldm_mvtd2 或 ../data/ldm_mvtd2。逐个探测，别写死。
    """
    code = ROOT
    for cand in (ROOT, ROOT / 'code'):
        if (cand / 's71_ldm_iter4.py').exists():
            code = cand
            break
    data = None
    for cand in (ROOT / 'data/ldm_mvtd2', ROOT.parent / 'data/ldm_mvtd2',
                 code.parent / 'data/ldm_mvtd2'):
        if (cand / 'train.jsonl').exists():
            data = cand
            break
    return code, (data or ROOT / 'data/ldm_mvtd2')


def run_one(tmp: Path, idx: int, steps: int, warmup: int) -> float:
    """跑一个 run，返回每秒步数。"""
    out = tmp / f'cal_{idx}.json'
    cmd = [
        sys.executable, 's71_ldm_iter4.py',
        '--task', 'mvtd', '--data-dir', str(ROOT / 'data/ldm_mvtd2'),
        '--pos', 'none', '--ouro', '--step-sup',
        '--curriculum', '2', '--curriculum-step', '8000',
        '--steps', str(steps), '--warmup', str(warmup),
        '--bs', '64', '--lr', '1e-3', '--r-max', '8',
        '--eval-every', str(steps), '--seed', str(100 + idx),
        '--out', str(out),
    ]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        print(f'  run{idx} 失败：{r.stderr.strip().splitlines()[-1][:160]}')
        return 0.0
    return steps / dt


def measure(slots: int, steps: int, warmup: int) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f'cal{slots}_'))
    code_dir, data_dir = _find_dirs()
    try:
        procs = [subprocess.Popen(
            [sys.executable, 's71_ldm_iter4.py',
             '--task', 'mvtd', '--data-dir', str(data_dir),
             '--pos', 'none', '--ouro', '--step-sup',
             '--curriculum', '2', '--curriculum-step', '8000',
             '--steps', str(steps), '--warmup', str(warmup),
             '--bs', '64', '--lr', '1e-3', '--r-max', '8',
             '--eval-every', str(steps), '--seed', str(200 + slots * 10 + i),
             '--out', str(tmp / f'cal_{i}.json')],
            cwd=str(code_dir), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
            for i in range(slots)]
        t0 = time.perf_counter()
        for p in procs:
            p.wait()
        dt = time.perf_counter() - t0
        ok = sum(1 for p in procs if p.returncode == 0)
        if ok < slots:
            # 子进程失败（路径/依赖问题）时数据无意义，必须显式报出来，
            # 否则效率算成 0 会被误读成"并行度没用"。
            print(f'  !! {slots}-way: 只有 {ok}/{slots} 个子进程成功返回 0，'
                  f'测量无效（检查代码/数据目录）', flush=True)
        return {'slots': slots, 'wall': dt, 'ok': ok,
                'per_run_sps': steps / dt,
                'total_sps': steps * ok / dt}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slots', default='1,3,6,9')
    ap.add_argument('--steps', type=int, default=600,
                    help='每个 run 的步数。太小会被进程启动开销污染'
                         '（实测 100 步时启动占了一半以上）')
    ap.add_argument('--warmup', type=int, default=60)
    args = ap.parse_args()
    slots = [int(x) for x in args.slots.split(',')]

    # 预热：先跑一个小 run，把 torch import / 磁盘缓存 / 内核选择的热身
    # 成本排除在计时之外，否则第一个档位会被系统性低估。
    print('预热中（不计入计时）...', flush=True)
    measure(1, 100, 20)

    print(f'校准：steps={args.steps} warmup={args.warmup}')
    print(f"{'并行':>4s} {'墙上(s)':>9s} {'单run步/秒':>11s} "
          f"{'总吞吐步/秒':>12s} {'加速比':>7s} {'相对效率':>9s}")
    base = None
    for s in slots:
        m = measure(s, args.steps, args.warmup)
        if base is None:
            base = m['total_sps'] or 1.0
        speedup = m['total_sps'] / base
        eff = speedup / s
        print(f"{s:>4d} {m['wall']:>9.1f} {m['per_run_sps']:>11.2f} "
              f"{m['total_sps']:>12.2f} {speedup:>7.2f} {eff:>9.2f}")
    print('\n判读：相对效率 = 加速比 / 并行度。'
          '\n  接近 1.0 → 仍线性，可继续加；'
          '\n  掉到 0.7 以下 → 已过拐点，取前一档并行度。')


if __name__ == '__main__':
    main()
