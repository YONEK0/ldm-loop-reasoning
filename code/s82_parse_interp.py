"""s82: H-022 深度插值实验的机械判读器。

读 results/remote_R_INT_even_s{53,17,71}.json，提取未训练深度
hop3/hop5（dev_r8_by_hop）与 hop7（confirm_r6_by_hop / confirm_r8_by_hop）
的准确率，按 H-022 预注册判据给出每个 seed 与总体的判决。

预注册判据（H-022-PENDING-depth-interpolation.md，不得事后修改）：
  插值成立  未训练深度均 >=0.9
  部分      未训练深度 0.6-0.9
  不成立    未训练深度 ≈0.5（随机）

对照：同 run 的训练深度 hop2/4/6/8 应当高（>0.9），否则 run 本身无效。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
SEEDS = (53, 17, 71)
UNSEEN_DEV = ('3', '5')
UNSEEN_CONF = ('7',)
TRAINED = ('2', '4', '6', '8')


def load(seed):
    p = ROOT / 'results' / f'remote_R_INT_even_s{seed}.json'
    if not p.exists():
        return None, p
    return json.loads(p.read_text(encoding='utf-8')), p


def pick_ev(j):
    return j.get('last_evals') or j.get('evals') or {}


def fmt_hop(d):
    if not d:
        return '-'
    return ' '.join(f'h{k}={v:.3f}' for k, v in sorted(
        d.items(), key=lambda kv: int(kv[0])))


def main():
    rows = []
    for s in SEEDS:
        j, p = load(s)
        if j is None:
            print(f'[missing] {p.name}')
            continue
        ev = pick_ev(j)
        dev8 = ev.get('dev_r8_by_hop') or {}
        conf6 = ev.get('confirm_r6_by_hop') or {}
        conf8 = ev.get('confirm_r8_by_hop') or {}
        unseen = {h: dev8.get(h) for h in UNSEEN_DEV}
        unseen.update({h: conf6.get(h) for h in UNSEEN_CONF})
        missing = [h for h, v in unseen.items() if v is None]
        vals = [v for v in unseen.values() if v is not None]
        if missing or not vals:
            verdict = f'INCOMPLETE(missing {missing})'
        elif all(v >= 0.9 for v in vals):
            verdict = 'CONFIRMED'
        elif all(v >= 0.6 for v in vals):
            verdict = 'PARTIAL'
        elif max(vals) < 0.6:
            verdict = 'REFUTED'
        else:
            verdict = 'MIXED'
        trained_ok = {h: dev8.get(h) for h in TRAINED}
        rows.append({'seed': s, 'step': ev.get('step'),
                     'dev_r8': ev.get('dev_r8'), 'dev_r6': ev.get('dev_r6'),
                     'dev_r8_by_hop': dev8, 'confirm_r8': ev.get('confirm_r8'),
                     'confirm_r6_by_hop': conf6, 'confirm_r8_by_hop': conf8,
                     'unseen': unseen, 'verdict': verdict,
                     'trained_ok': trained_ok,
                     'early_stop': j.get('early_stop')})
        print(f'--- s{s}  step={ev.get("step")}  dev_r8={ev.get("dev_r8")} '
              f'confirm_r8={ev.get("confirm_r8")}')
        print(f'  dev_r8_by_hop   : {fmt_hop(dev8)}')
        print(f'  confirm_r6_byhop: {fmt_hop(conf6)}')
        print(f'  confirm_r8_byhop: {fmt_hop(conf8)}')
        print(f'  unseen(3,5,7)   : {unseen}  -> {verdict}')
        print(f'  trained(2,4,6,8): {trained_ok}')
        if j.get('early_stop'):
            print(f'  early_stop: {j["early_stop"]}')

    if len(rows) == 3:
        agg = {}
        for h in UNSEEN_DEV + UNSEEN_CONF:
            vs = [r['unseen'].get(h) for r in rows]
            agg[h] = None if any(v is None for v in vs) else \
                round(sum(vs) / len(vs), 4)
        print('--- 3-seed mean unseen:', agg)
        means = [v for v in agg.values() if v is not None]
        if means and all(v >= 0.9 for v in means):
            print('=== H-022 VERDICT: CONFIRMED (可插值，课程可稀疏)')
        elif means and all(v >= 0.6 for v in means):
            print('=== H-022 VERDICT: PARTIAL (0.6-0.9，插值不稳定)')
        elif means and max(means) < 0.6:
            print('=== H-022 VERDICT: REFUTED (未训练深度≈随机，需逐级覆盖)')
        else:
            print('=== H-022 VERDICT: MIXED/INCOMPLETE')

    out = ROOT / 'results' / 's82_interp_verdict.json'
    out.write_text(json.dumps({'seeds': rows}, indent=2) + '\n',
                   encoding='utf-8')
    print('[written]', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
