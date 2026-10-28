"""s89_auto_loop.py — 自主"测试→判读→优化"迭代链（可接 dsh 智能体）。

## 架构

    spec.json -> [scan] -> [evaluate] -> [decide] -> [act] -> [record]
                     ^                                  |
                     +------------- loop ----------------+

## 决策后端

- `--agent rules`：按 spec 的 `on_done` 规则（确定性、可复现）
- `--agent dsh`：用 `dsh headless "<prompt>"` 让智能体决定，
  失败自动回退到规则。密钥从 /root/ldm/.dsh_env 读取（600 权限）。

## 用法

    python s89_auto_loop.py --spec docs/plans/auto_loop_spec.json --once
    nohup python s89_auto_loop.py --spec ... --interval 180 --agent dsh &
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
STATE = RES / 'auto_loop_state.json'
LOG = RES / 'auto_loop_log.jsonl'
REPORT = RES / 'auto_loop_report.md'
DSH_ENV = Path('/root/ldm/.dsh_env')
PY = sys.executable
NL = chr(10)


# ---------- 结果扫描 ----------
def _get(d: dict, path: str):
    cur = d
    for part in path.split('.'):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def scan(pattern: str) -> list[dict]:
    runs = []
    for p in sorted(RES.glob(pattern + '.json')):
        try:
            j = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        ev = j.get('evals') or {}
        runs.append({'name': p.stem, 'step': ev.get('step') or 0,
                     'evals': ev, 'hist': j.get('evals_hist') or []})
    return runs


def stage_metrics(runs: list[dict], metric_map: dict) -> list[dict]:
    out = []
    for r in runs:
        rec = {'name': r['name'], 'step': r['step']}
        for k, path in metric_map.items():
            rec[k] = _get({'evals': r['evals']}, 'evals.' + path)
        out.append(rec)
    return out


def safe_eval(expr: str, env: dict) -> bool:
    allowed = {k: v for k, v in env.items() if isinstance(v, (int, float))}
    try:
        return bool(eval(expr, {'__builtins__': {}}, allowed))  # noqa: S307
    except Exception:
        return False


def summarize(rows: list[dict], pass_expr: str) -> dict:
    ok, vals = 0, {}
    for r in rows:
        if safe_eval(pass_expr, {k: v for k, v in r.items()
                                 if isinstance(v, (int, float))}):
            ok += 1
        for k, v in r.items():
            if isinstance(v, (int, float)):
                vals.setdefault(k, []).append(v)
    n = len(rows)
    agg = {k: round(sum(v) / len(v), 4) for k, v in vals.items() if v}
    agg.update({'n': n, 'passed': ok,
                'pass_rate': round(ok / n, 4) if n else 0.0,
                'max_h7': max(vals.get('h7', [0]) or [0]),
                'max_h6': max(vals.get('h6', [0]) or [0])})
    return agg


# ---------- 动作 ----------
def launch(tag: str, seeds: list[int], base_args: str, extra: str) -> list[str]:
    started = []
    for s in seeds:
        out = f'results/{tag}_s{s}.json'
        if (RES / Path(out).name).exists():
            continue
        cmd = (f'cd {ROOT} && nohup {PY} s71_ldm_iter4.py {base_args} {extra} '
               f'--seed {s} --out {out} > {out[:-5]}.log 2>&1 < /dev/null &')
        subprocess.Popen(['bash', '-c', cmd], start_new_session=True)
        started.append(out)
    return started


def write_report(stage: str, summ: dict, action: str, extra: str = ''):
    lines = [f'# 自动迭代报告（{time.strftime("%Y-%m-%d %H:%M:%S")}）', '',
             f'**当前阶段**：`{stage}`', '', '## 判读摘要', '',
             '| 指标 | 值 |', '|---|---|']
    for k, v in sorted(summ.items()):
        lines.append(f'| {k} | {v:.4f} |' if isinstance(v, float)
                     else f'| {k} | {v} |')
    lines += ['', f'**决策**：{action}', '']
    if extra:
        lines += [extra, '']
    REPORT.write_text(NL.join(lines) + NL, encoding='utf-8')


def record(rec: dict):
    with LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + NL)


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'stage_idx': 0, 'launched': {}, 'history': []}


def save_state(st: dict):
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2) + NL,
                     encoding='utf-8')


# ---------- 智能体决策（dsh headless） ----------
def _load_env() -> dict:
    env = dict(os.environ)
    if DSH_ENV.exists():
        for line in DSH_ENV.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    env['PATH'] = '/usr/local/node22/bin:' + env.get('PATH', '')
    return env


def agent_decide(stage: dict, summ: dict, spec: dict,
                 timeout: int = 180) -> dict | None:
    acts = json.dumps(stage.get('on_done', []), ensure_ascii=False)[:1000]
    lines = [
        "先读 /root/ldm/docs/plans/AGENT_TASK_BRIEF.md（背景、已确立结论、当前状态、第八节闭环契约）。你有 bash/文件/联网工具：遇到行不通的局面，先检索文献、写假说与预注册判据，再启动实验。",
        f"你是 LDM 循环模型实验的自主监督者。阶段「{stage['name']}」完成。",
        f"判读摘要（JSON）：{json.dumps(summ, ensure_ascii=False)}",
        f"预注册规则（可参考）：{acts}",
        f"base_args：{spec['base_args'][:350]}",
        "决定下一步。只选一个动作，只输出一行纯 JSON，不要解释、不要代码块：",
        '{"type":"launch","tag":"<名字>","seeds":[<整数>],"extra":"<额外参数或空>","next":"<下一阶段名或空>","reason":"<20字内>"}',
        '{"type":"stage","next":"<下一阶段名>","reason":"<20字内>"}',
        '{"type":"stop","reason":"<30字内>"}',
        "约束：seeds 取 1-4 个整数；不要重复已跑过的 tag。",
    ]
    prompt = NL.join(lines)
    try:
        r = subprocess.run(['dsh', 'headless', prompt], env=_load_env(),
                           capture_output=True, text=True, timeout=timeout,
                           cwd=str(ROOT))
        out = (r.stdout or '').strip()
        i, j = out.find('{'), out.rfind('}')
        if i < 0 or j <= i:
            return None
        act = json.loads(out[i:j + 1])
        if act.get('type') not in ('launch', 'stage', 'stop'):
            return None
        act['_agent'] = True
        return act
    except Exception:
        return None


# ---------- 主循环 ----------
def tick(spec: dict, st: dict, dry: bool = False,
         agent: str = 'rules') -> dict:
    stages = spec['stages']
    idx = st['stage_idx']
    if idx >= len(stages):
        return {'done': True, 'action': 'all stages complete'}
    stage = stages[idx]
    runs = scan(stage['glob'])
    target_step = stage.get('target_step')
    ready = [r for r in runs if target_step is None
             or r['step'] >= target_step]
    expect = stage.get('expect', len(runs))
    rows = stage_metrics(ready, stage['metric'])
    summ = summarize(rows, stage.get('pass_if', 'False'))
    summ['n_ready'] = len(ready)
    summ['expect'] = expect

    if len(ready) < expect:
        action = f"wait: {len(ready)}/{expect} runs reached step>={target_step}"
        write_report(stage['name'], summ, action)
        return {'done': False, 'action': action, 'summary': summ}

    # 智能体优先（失败回退规则）
    if agent == 'dsh' and not dry:
        act = agent_decide(stage, summ, spec)
        if act is not None:
            act.setdefault('next', '')
            if act['type'] == 'stop':
                action = f"[dsh] stop: {act.get('reason', '')}"
                write_report(stage['name'], summ, action)
                return {'done': True, 'action': action, 'summary': summ}
            if act['type'] == 'stage':
                nxt = act.get('next')
                st['stage_idx'] = next(
                    (i for i, s in enumerate(stages) if s['name'] == nxt),
                    idx + 1)
                save_state(st)
                action = f"[dsh] stage -> {nxt} ({act.get('reason', '')})"
                write_report(stage['name'], summ, action)
                return {'done': False, 'action': action, 'summary': summ}
            if act['type'] == 'launch':
                started = launch(act.get('tag', 'AGENT'),
                                 act.get('seeds') or [71],
                                 spec['base_args'], act.get('extra', ''))
                if act.get('next'):
                    st['stage_idx'] = next(
                        (i for i, s in enumerate(stages)
                         if s['name'] == act['next']), idx + 1)
                st['launched'].setdefault(act.get('tag', 'AGENT'), []).extend(
                    started)
                save_state(st)
                action = (f"[dsh] launch {act.get('tag')} "
                          f"seeds={act.get('seeds')} ({len(started)} started)"
                          f" - {act.get('reason', '')}")
                write_report(stage['name'], summ, action,
                             NL.join(['新启动：'] + [f'- {s}' for s in started]))
                return {'done': False, 'action': action, 'summary': summ}

    for rule in stage.get('on_done', []):
        if safe_eval(rule['if'], summ):
            act = rule['then']
            if act['type'] == 'stage':
                nxt = act.get('next')
                st['stage_idx'] = next(
                    (i for i, s in enumerate(stages) if s['name'] == nxt),
                    idx + 1)
                if not dry:
                    save_state(st)
                action = f'advance stage -> {nxt}'
                write_report(stage['name'], summ, action)
                return {'done': False, 'action': action, 'summary': summ}
            if act['type'] == 'launch':
                started = (launch(act['tag'], act['seeds'], spec['base_args'],
                                  act.get('extra', '')) if not dry
                           else ['(dry)'] * len(act['seeds']))
                if act.get('next') and not dry:
                    st['stage_idx'] = next(
                        (i for i, s in enumerate(stages)
                         if s['name'] == act['next']), idx + 1)
                if not dry:
                    st['launched'].setdefault(act['tag'], []).extend(started)
                    save_state(st)
                action = (f"launch {act['tag']} seeds={act['seeds']} "
                          f"({len(started)} started)")
                write_report(stage['name'], summ, action,
                             NL.join(['新启动：'] + [f'- {s}' for s in started]))
                return {'done': False, 'action': action, 'summary': summ}
            if act['type'] == 'stop':
                action = f"stop: {act.get('reason', '')}"
                write_report(stage['name'], summ, action)
                return {'done': True, 'action': action, 'summary': summ}

    action = 'no rule matched (hold)'
    write_report(stage['name'], summ, action)
    return {'done': False, 'action': action, 'summary': summ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--interval', type=int, default=300)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--agent', default='rules', choices=['rules', 'dsh'])
    args = ap.parse_args()

    st = load_state()
    print(f'[auto_loop] spec={args.spec} stage={st["stage_idx"]} '
          f'agent={args.agent}', flush=True)
    # PID 文件：便于用数字 PID 精确清理（pkill -f 对 setsid 脱离的进程不可靠）
    try:
        (RES / 'auto_loop.pid').write_text(str(os.getpid()))
    except Exception:
        pass
    while True:
        # 每轮重读 spec：改配置后无需重启（此前只在启动时读一次，改了就失效）
        try:
            spec = json.loads(Path(args.spec).read_text(encoding='utf-8'))
        except Exception as e:
            print(f'[auto_loop] spec reload failed: {e}', flush=True)
            time.sleep(args.interval)
            continue
        st = load_state()      # 状态也重读，避免与外部编辑冲突
        try:
            r = tick(spec, st, dry=args.dry_run, agent=args.agent)
        except Exception as e:
            r = {'done': False, 'action': f'error: {type(e).__name__}: {e}'}
        rec = {'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
               'agent': args.agent,
               **{k: v for k, v in r.items() if k != 'summary'},
               'summary': r.get('summary')}
        record(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        if r['done'] or args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
