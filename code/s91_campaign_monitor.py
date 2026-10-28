"""s91_campaign_monitor.py — 正式训练的翻车监测器（确定性，无 API）。

与 s77_watchdog 的分工：s77 是"进程级"监护（死了没、loss 炸没），
本脚本是"指标级"监护，盯的是本项目实际出现过的四种翻车形态：

  1. COLLAPSE  升档后遗忘：某跳准确率曾 ≥0.85，现在 <0.60
                （实测：cap 升档后 hop4 从 0.915 崩到 0.1）
  2. SEESAW    跷跷板：同一轮里最高与最低跳相差 ≥0.40 且最低 <0.60
                （实测：hop5=0.0 同时 hop6=1.0，H-007）
  3. STALL     收敛停滞：连续 N 次 eval 的 hop6 峰值无提升，且 hop7 <0.80
                （实证：task loss 卡在 0.21+ 的 seed 全部 fail，H-029）
  4. FLAT      评测失效：dev_r1..r8 首次与末次相差 <0.02（全随机）
                （实测：评测漏传 adj_bias_list 导致恒 0.50，H-026）

判据阈值在此处**写死为默认值**，可在命令行覆盖；跑训练前先定下来，
不要在看到结果后再调。

用法：
  python s91_campaign_monitor.py --once                    # 打一张表
  python s91_campaign_monitor.py --watch 300               # 每 5 分钟刷一次
  python s91_campaign_monitor.py --once --alert-exit 2     # 有 ALERT 时退出码 2

输出：
  results/campaign_status.md    人读的看板
  results/campaign_status.json  机读状态（供上层自动化判断）
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'

# ---- 判据默认值（跑前定稿） ----
COLLAPSE_HIGH = 0.85      # 曾达到过高水位
COLLAPSE_LOW = 0.60       # 现在低于此值即报
SEESAW_SPREAD = 0.40      # 同轮最高-最低
SEESAW_FLOOR = 0.60       # 且最低低于此值
STALL_WINDOW = 4          # 连续多少次 eval 无新峰值
STALL_H7 = 0.80           # 且 hop7 未达标
FLAT_SPREAD = 0.02        # r1..r8 跨度小于此值视为全随机
LATE_FRAC = 0.75          # 训练进度超过此比例后
LATE_TASK = 0.15          # task loss 仍高于此值 → 收敛停滞
DEAD_AGE = 2400           # 文件 40 分钟无更新且未完成 → 疑似进程死亡


def _by_hop_series(ev: list[dict]) -> dict[str, list[float | None]]:
    """把 evals_hist 里的 by-hop 指标抽成时间序列（键: 'r5h6' 等）。"""
    out: dict[str, list[float | None]] = {}
    for e in ev:
        for src, prefix in (('dev_r8_by_hop', 'r8'),
                            ('dev_r5_by_hop', 'r5'),
                            ('confirm_r6_by_hop', 'c6')):
            for h, v in (e.get(src) or {}).items():
                out.setdefault(f'{prefix}h{h}', []).append(v)
    return out


def analyze(path: Path, target_override: int | None = None,
            dead_check: bool = False) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None
    if not isinstance(d, dict) or 'evals_hist' not in d:
        return None
    ev = d.get('evals_hist') or []
    hist = d.get('hist') or []
    args = d.get('args') or {}
    if not ev:
        return None

    target = target_override or args.get('steps') or 0
    last, prev = ev[-1], (ev[-2] if len(ev) > 1 else {})
    step = last.get('step') or 0
    task = None
    for h in reversed(hist):
        if isinstance(h.get('task'), (int, float)):
            task = h['task']
            break

    series = _by_hop_series(ev)
    flags: list[str] = []

    # 1. COLLAPSE：任何跳的水位从 ≥0.85 掉到 <0.60
    for key, vals in series.items():
        clean = [v for v in vals if isinstance(v, (int, float))]
        if len(clean) < 2:
            continue
        if max(clean) >= COLLAPSE_HIGH and clean[-1] < COLLAPSE_LOW:
            flags.append(f'COLLAPSE:{key}:{max(clean):.2f}->{clean[-1]:.2f}')

    # 2. SEESAW：同一 eval 内的极差（只看课程已覆盖的跳）
    cur = {k: v for k, v in (last.get('dev_r8_by_hop') or {}).items()}
    if len(cur) >= 2:
        hi, lo = max(cur.values()), min(cur.values())
        if hi - lo >= SEESAW_SPREAD and lo < SEESAW_FLOOR:
            flags.append(f'SEESAW:spread={hi - lo:.2f}:min={lo:.2f}')

    # 3. STALL：hop6 峰值长时间不动且 hop7 未达标
    h6 = [(v, i) for i, v in enumerate(series.get('r5h6', []))
          if isinstance(v, (int, float))]
    h7 = (last.get('confirm_r6_by_hop') or {}).get('7')
    if h6:
        peak, peak_i = max((v, i) for v, i in h6)
        quiet = len(h6) - 1 - peak_i
        if quiet >= STALL_WINDOW and (h7 is None or h7 < STALL_H7):
            flags.append(f'STALL:h6peak={peak:.2f}@eval{peak_i}:quiet={quiet}')

    # 4. FLAT：r1..r8 既全随机又彼此相等 —— 评测失效或完全没学。
    #    注意必须同时满足"贴近随机"，否则"各轮都饱和在 0.9"会被误报。
    rs = [last.get(f'dev_r{i}') for i in (1, 2, 3, 4, 5, 6, 8)]
    rs = [v for v in rs if isinstance(v, (int, float))]
    if len(rs) >= 3 and max(rs) - min(rs) < FLAT_SPREAD and max(rs) < 0.65:
        flags.append(f'FLAT:spread={max(rs) - min(rs):.3f}:level={max(rs):.2f}')

    # 5. LATE_STALL：训练后段 task loss 仍高（H-029 的签名）
    if target and step >= LATE_FRAC * target and task is not None \
            and task >= LATE_TASK:
        flags.append(f'LATE_STALL:task={task:.3f}')

    age = time.time() - path.stat().st_mtime
    done = bool(target and step >= target)
    early = bool(d.get('early_stop'))
    # DEAD 只在显式开启时判定（--dead-check）。历史 run 的 mtime 都很旧，
    # 默认开启会把归档数据全部误报成"进程死了"。
    if dead_check and not done and not early and age > DEAD_AGE:
        flags.append(f'DEAD_OR_IDLE:age={age / 60:.0f}min:step={step}')

    hard = [f for f in flags if f.split(':')[0] in
            ('COLLAPSE', 'SEESAW', 'FLAT', 'DEAD_OR_IDLE')]
    status = 'ALERT' if hard else ('WATCH' if flags else 'OK')
    return {
        'name': path.stem, 'step': step, 'target': target,
        'task': task, 'h6': (last.get('dev_r5_by_hop') or {}).get('6'),
        'h7': h7, 'h8': (last.get('confirm_r6_by_hop') or {}).get('8'),
        'r1': last.get('dev_r1'), 'r8': last.get('dev_r8'),
        'cap': last.get('cap'), 'n_eval': len(ev), 'age_min': round(age / 60),
        'done': done, 'early_stop': early,
        'status': status, 'flags': flags,
    }


def render(rows: list[dict]) -> str:
    order = {'ALERT': 0, 'WATCH': 1, 'OK': 2, 'DONE': 3}
    rows = sorted(rows, key=lambda r: (order.get(r['status'], 9),
                                       -(r['h7'] or 0)))
    head = (f"{'run':30s} {'st':5s} {'step':>6s} {'cap':>3s} "
            f"{'task':>6s} {'h6':>5s} {'h7':>5s} {'h8':>5s} {'flags'}")
    lines = [head, '-' * len(head)]
    for r in rows:
        lines.append(
            f"{r['name'][:30]:30s} {r['status']:5s} "
            f"{r['step']:>6d} {str(r['cap']):>3s} "
            f"{(r['task'] if r['task'] is not None else float('nan')):>6.3f} "
            f"{(r['h6'] if r['h6'] is not None else float('nan')):>5.2f} "
            f"{(r['h7'] if r['h7'] is not None else float('nan')):>5.2f} "
            f"{(r['h8'] if r['h8'] is not None else float('nan')):>5.2f} "
            f"{','.join(r['flags'])[:80]}")
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', action='append', default=None,
                    help='要监测的文件名模式，可多次给；默认扫描全部 run JSON')
    ap.add_argument('--watch', type=int, default=0, help='秒；0=只跑一次')
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--out-prefix', default=str(RES / 'campaign_status'))
    ap.add_argument('--alert-exit', type=int, default=0,
                    help='有 ALERT 时使用的退出码（0=不设）')
    ap.add_argument('--dead-check', action='store_true',
                    help='启用"进程疑似死亡"判定。历史归档请勿开启')
    args = ap.parse_args()

    pats = args.glob or ['*.json']

    def one_pass() -> list[dict]:
        seen: dict[str, Path] = {}
        for pat in pats:
            for p in RES.rglob(pat):
                if p.name.startswith('remote_') or p.name.startswith('campaign_status'):
                    continue
                if p.name in ('auto_loop_state.json', 'watchdog_state.json'):
                    continue
                seen[str(p)] = p
        rows = [r for r in (analyze(p, dead_check=args.dead_check)
                            for p in seen.values()) if r]
        return rows

    while True:
        rows = one_pass()
        table = render(rows)
        n_alert = sum(1 for r in rows if r['status'] == 'ALERT')
        n_watch = sum(1 for r in rows if r['status'] == 'WATCH')
        n_ok = sum(1 for r in rows if r['status'] == 'OK')
        stamp = time.strftime('%Y-%m-%d %H:%M:%S')
        text = f'# 训练监测看板 {stamp}\n\n```\n{table}\n```\n'
        Path(args.out_prefix + '.md').write_text(text, encoding='utf-8')
        Path(args.out_prefix + '.json').write_text(
            json.dumps({'ts': stamp, 'n': len(rows), 'n_alert': n_alert,
                        'runs': rows}, ensure_ascii=False, indent=2),
            encoding='utf-8')
        print(text)
        if args.alert_exit and n_alert:
            raise SystemExit(args.alert_exit)
        if args.once or not args.watch:
            break
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
