"""s77_watchdog.py — 训练监督守护进程（无需 API，确定性逻辑）。

每 --interval 秒扫描 results/*.json，检测四类异常并按预设动作处理：

  A. 进程死亡但未完成  → 记录 + 可选重启
  B. loss 爆炸（NaN/inf/突增 10×） → 记录 + 标记，等待上层决策
  C. 精度长期不涨（连续 N 次 eval 峰值无提升）→ 记录为停滞
  D. 全部 hop 归零（catastrophe）→ 回滚到最近的 MILESTONE

所有动作写入 results/watchdog.log 与 results/watchdog_state.json，
供上层（人工或 dsh）读取。设计原则：**只做保守动作**，破坏性操作需人工确认。

用法：
  python s77_watchdog.py --interval 300              # 前台
  python s77_watchdog.py --interval 300 --auto-stop  # 允许自动停掉灾变run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
LOG = RES / 'watchdog.log'
STATE = RES / 'watchdog_state.json'

# 关注的文件模式（服务器与本机通用）
PATTERNS = ('C_persist_*.json', 'D_long_*.json', 'A_film_*.json',
            'h0*_*.json', 'base_*.json', 'iter4_v*.json')

STALL_WINDOW = 5          # 连续多少次 eval 无提升视为停滞
CATASTROPHE_ZERO = 3      # 多少个 hop 同时 <0.2 视为灾变


def log(msg: str):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'runs': {}}


def save_state(st: dict):
    STATE.write_text(json.dumps(st, indent=2), encoding='utf-8')


def discover():
    seen = {}
    for pat in PATTERNS:
        for p in RES.glob(pat):
            if p.name.startswith('remote_'):
                continue
            seen[str(p)] = p
    return seen


def analyze(path: Path):
    """返回当前 run 的诊断字典；无法读取返回 None。"""
    try:
        d = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    hist = d.get('hist') or []
    ev = d.get('evals_hist') or []
    args = d.get('args') or {}
    target = args.get('steps')

    losses = [h.get('loss') for h in hist if isinstance(h.get('loss'), (int, float))]
    bad_loss = any((x is None or x != x or x == float('inf')) for x in losses)
    spike = False
    if len(losses) >= 4:
        recent = losses[-1]
        prev_med = sorted(losses[:-1])[len(losses[:-1]) // 2]
        if prev_med > 0 and recent > prev_med * 10 and recent > 1.0:
            spike = True

    peak = 0.0
    peak_at = None
    stall = 0
    for e in ev:
        v = e.get('dev_r8') or 0
        if v > peak + 1e-4:
            peak, peak_at, stall = v, e.get('step'), 0
        else:
            stall += 1

    zeros = 0
    if ev:
        for k, v in (ev[-1].get('dev_r8_by_hop') or {}).items():
            if v < 0.2:
                zeros += 1

    return {
        'path': str(path), 'name': path.stem,
        'step': ev[-1].get('step') if ev else (hist[-1].get('step') if hist else None),
        'target': target,
        'peak': round(peak, 4), 'peak_at': peak_at, 'stall': stall,
        'n_eval': len(ev),
        'zeros_last': zeros,
        'bad_loss': bad_loss, 'spike': spike,
        'done': bool(target and ev and ev[-1].get('step', 0) >= target),
        'early_stop': d.get('early_stop'),
        'mtime': path.stat().st_mtime,
        'age': time.time() - path.stat().st_mtime,
    }


def process_alive(name: str) -> bool:
    """用 pgrep 判断该 run 的训练进程是否还在（Linux）。"""
    try:
        r = subprocess.run(['pgrep', '-f', name], capture_output=True,
                           text=True, timeout=10)
        return r.returncode == 0
    except Exception:
        return True      # 无法判断时保守假设还活着


def find_milestone(name: str):
    """找该 run 最新的里程碑存档。"""
    ms = sorted(RES.glob(f'{name}_MILESTONE_*.pt'))
    return ms[-1] if ms else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=int, default=300)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--auto-stop', action='store_true',
                    help='灾变时自动停止该 run（kill）')
    ap.add_argument('--max-cycles', type=int, default=0,
                    help='0=无限')
    args = ap.parse_args()

    RES.mkdir(exist_ok=True)
    log(f'watchdog start interval={args.interval}s auto_stop={args.auto_stop}')
    st = load_state()
    cycles = 0

    while True:
        cycles += 1
        files = discover()
        log(f'--- cycle {cycles}: {len(files)} runs ---')

        for path_str, path in sorted(files.items()):
            info = analyze(path)
            if info is None:
                continue
            name = info['name']
            prev = st['runs'].get(name, {})
            events = []

            if info['done'] or info['early_stop']:
                events.append('DONE')

            # A. 进程死亡但未完成（早停档存在或峰值≥0.98 = 实质完成）
            _has_earlystop = any(RES.glob(f'{name}_EARLYSTOP.pt'))
            _effectively_done = info['peak'] >= 0.98
            if (not info['done'] and not info['early_stop']
                    and not _has_earlystop and not _effectively_done
                    and info['age'] > 900):
                if not process_alive(name):
                    events.append('DEAD_NOT_DONE')

            # B. loss 异常
            if info['bad_loss']:
                events.append('BAD_LOSS')
            if info['spike']:
                events.append('LOSS_SPIKE')

            # C. 停滞
            if info['stall'] >= STALL_WINDOW:
                events.append(f'STALL_{info["stall"]}')

            # D. 灾变（多 hop 同时归零）
            if info['zeros_last'] >= CATASTROPHE_ZERO:
                events.append('CATASTROPHE')
                ms = find_milestone(name)
                if ms:
                    events.append(f'MILESTONE_AVAIL:{ms.name}')

            new_events = [e for e in events if e not in prev.get('events', [])]
            if new_events:
                log(f'  {name}: step={info["step"]}/{info["target"]} '
                    f'peak={info["peak"]} new={new_events}')
                if 'CATASTROPHE' in new_events and args.auto_stop:
                    subprocess.run(['pkill', '-f', name], timeout=15)
                    log(f'  → AUTO-STOP {name}')

            st['runs'][name] = {**info, 'events': events}

        # 汇总
        alive = [k for k, v in st['runs'].items()
                 if not v.get('done') and not v.get('early_stop')]
        finished = [k for k, v in st['runs'].items()
                    if v.get('done') or v.get('early_stop')]
        log(f'summary: {len(alive)} running, {len(finished)} finished')
        save_state(st)

        if args.once or (args.max_cycles and cycles >= args.max_cycles):
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
