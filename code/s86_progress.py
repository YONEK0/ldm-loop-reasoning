"""s86_progress.py — 一次性查看所有实验进度与关键指标（在服务器上运行）。

用法：
  python s86_progress.py                    # 看所有 P0*/P01*/P02*/MVTD*/OURO*
  python s86_progress.py --pattern 'P01_*'  # 只看某一批
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='*.json')
    ap.add_argument('--show-scan', action='store_true',
                    help='显示完整的 r 扫描')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RES, args.pattern)))
    files = [f for f in files if f.endswith('.json')
             and '_ckpt' not in f and not f.endswith('.json.tmp')]
    if not files:
        print('(no result files yet)')
        return
    print(f'{"run":<28} {"step":>6} {"loss":>7} {"r1":>6} {"r2":>6} '
          f'{"r3":>6} {"r8":>6} {"h7":>6} {"h8":>6}')
    print('-' * 88)
    for f in files:
        name = os.path.basename(f)[:-len('.json')]
        try:
            j = json.load(open(f, encoding='utf-8'))
        except Exception:
            print(f'{name:<28} (incomplete)')
            continue
        h = j.get('hist') or []
        last_h = h[-1] if h else {}
        ev = j.get('evals') or {}
        if not ev:
            hist_ev = j.get('evals_hist') or []
            ev = hist_ev[-1] if hist_ev else {}
        conf = ev.get('confirm_r6_by_hop') or {}
        step = ev.get('step') or last_h.get('step') or 0
        loss = last_h.get('loss')
        print(f'{name:<28} {step:>6} '
              f'{(f"{loss:.3f}" if loss is not None else "-"):>7} '
              f'{_fmt(ev.get("dev_r1")):>6} {_fmt(ev.get("dev_r2")):>6} '
              f'{_fmt(ev.get("dev_r3")):>6} {_fmt(ev.get("dev_r8")):>6} '
              f'{_fmt(conf.get("7")):>6} {_fmt(conf.get("8")):>6}')
        if args.show_scan:
            scan = {r: ev.get(f'dev_r{r}') for r in (1, 2, 3, 4, 5, 6, 8)}
            print(f'{"":28} scan={scan}  by_hop={ev.get("dev_r8_by_hop")}')
    print(f'\n[{time.strftime("%H:%M:%S")}] {len(files)} result files')


def _fmt(v):
    return f'{v:.3f}' if isinstance(v, (int, float)) else '-'


if __name__ == '__main__':
    main()
