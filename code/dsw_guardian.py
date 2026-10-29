"""dsw_guardian.py — DSW 实例上的训练守护进程（保守型）。

定位：**观察者 + 最小干预者**，不是训练调度器，也不是 LLM。
阈值与动作边界写死在 `docs/plans/PREREG_GUARDIAN_20261029.md`，
部署前定稿，跑起来后不改。

核心约束（对应预注册 §1）：
  1. 默认不动手；任何自动动作都必须可逆或不丢数据。
  2. 只读 results/**/*.json；不删不改任何 run 产物。
  3. 不修改任何训练参数。
  4. 同一异常 60 分钟冷却，避免重复动作。
  5. 致命级动作要求**两条独立证据**。

动作（全部满足预注册 §3 的边界）：
  - 停坏 run：从 MILESTONE 重启到**新 --out**（原档保留）。
  - kill：仅 NaN/SPIKE 双证据命中时，按数字 PID 停**单个** run。
  - 绝不 pkill -f（会匹配到本进程的命令行）。

用法：
  python dsw_guardian.py --results /mnt/workspace/results/main \
      --guard /mnt/workspace/results/guardian --interval 60
  python dsw_guardian.py --once            # 跑一轮检测就退出（演练用）
  python dsw_guardian.py --drill kill      # 注入一次故障做演练
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------- 预注册阈值
NAN_VALUES = (float('nan'), float('inf'), float('-inf'))
SPIKE_FACTOR = 10.0          # loss > 中位数 × 10 且 > 1.0
COLLAPSE_HIGH = 0.85         # 曾达到
COLLAPSE_LOW = 0.60          # 现在低于
SEESAW_SPREAD = 0.40
SEESAW_FLOOR = 0.60
STALL_EVALS = 5              # 连续无提升的 eval 次数
STALL_PROGRESS = 0.60        # 训练进度超过此比例才算停滞
FLAT_SPREAD = 0.02
FLAT_LEVEL = 0.65
FLAT_MIN_EVALS = 3           # 至少 3 个 eval 点才判 FLAT（刚启动的 run 无历史）
DEAD_MIN_AGE = 300           # 进程消失后至少等这么久再判定
QUEUE_STALL_SEC = 1800       # 30 分钟没启动新 run
DISK_MIN_GB = 20.0           # 可用空间低于此值则停止启动新 run
GPU_IDLE_SEC = 1200          # 20 分钟无训练进程
COOLDOWN_SEC = 3600          # 同一 run 同一异常的冷却（可由 --cooldown 覆盖）

# 异常 → 预置检索词（预注册 §4）
LITERATURE_QUERIES = {
    'NAN_LOSS': ['all:"gradient explosion" AND all:"transformer"',
                 'all:"loss spike" AND all:"fine-tuning"'],
    'LOSS_SPIKE': ['all:"loss spike" AND all:"fine-tuning"',
                   'all:"gradient clipping" AND all:"instability"'],
    'COLLAPSE': ['all:"catastrophic forgetting" AND all:"curriculum"',
                 'all:"continual learning" AND all:"task interference"'],
    'SEESAW': ['all:"interference" AND all:"multi-task" AND all:"representation"'],
    'STALL': ['all:"optimization stagnation" AND all:"deep networks"',
              'all:"learning rate schedule" AND all:"plateau"'],
    'DEAD_RUN': ['all:"checkpoint" AND all:"resume" AND all:"training"'],
    'FLAT_ALL_RANDOM': ['all:"evaluation" AND all:"bug" AND all:"model"'],
    'DISK_PRESSURE': ['all:"out of memory" AND all:"training"'],
}


# ---------------------------------------------------------------- 工具
def _now() -> float:
    return time.time()


def _stamp(ts: float) -> str:
    return time.strftime('%Y%m%dT%H%M', time.localtime(ts))


def _atomic_write(path: Path, text: str) -> None:
    """写临时文件再 rename —— 避免守护进程被 kill 时留下半个文件。"""
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------- 文献检索
def search_arxiv(queries: list[str], max_results: int = 3) -> list[dict]:
    """查 arXiv。只返回事实（标题/ID/摘要片段），不做因果解释。"""
    out: list[dict] = []
    seen: set[str] = set()
    for q in queries:
        url = ('http://export.arxiv.org/api/query?search_query='
               + urllib.parse.quote(q) + f'&max_results={max_results}')
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ldm-guardian/1.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                xml = r.read().decode('utf-8', 'replace')
        except Exception:
            continue
        for entry in re.findall(r'<entry>(.*?)</entry>', xml, re.S):
            def _g(tag: str) -> str:
                m = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', entry, re.S)
                return re.sub(r'\s+', ' ', m.group(1)).strip() if m else ''
            aid = _g('id').rsplit('/', 1)[-1]
            title = _g('title')
            summary = _g('summary')
            if not aid or aid in seen:
                continue
            seen.add(aid)
            out.append({'arxiv_id': aid, 'title': title,
                        'summary': summary[:400], 'query': q})
            if len(out) >= max_results * len(queries):
                break
    return out


def write_literature(guard_dir: Path, event_id: str, event: str,
                      papers: list[dict]) -> str:
    """把检索结果写成 markdown。**只记录论文说了什么，不做因果归因。**"""
    lit_dir = guard_dir / 'literature'
    lit_dir.mkdir(parents=True, exist_ok=True)
    path = lit_dir / f'{event_id}.md'
    lines = [f'# 文献检索记录 · {event_id}',
             '',
             f'- 异常类型：`{event}`',
             f'- 检索时间：{time.strftime("%Y-%m-%d %H:%M:%S")}',
             f'- 数据源：arXiv API（export.arxiv.org）',
             '',
             '> 以下为检索到的文献内容摘要。**本文件不对异常成因做断言**——',
             '> 因果判断需由本项目的对照实验支撑。',
             '']
    if not papers:
        lines.append('_检索无结果。_')
    for p in papers:
        lines += [f'## {p["title"]}',
                  f'- arXiv: [{p["arxiv_id"]}](https://arxiv.org/abs/{p["arxiv_id"]})',
                  f'- 检索式：`{p["query"]}`',
                  '', p['summary'], '']
    _atomic_write(path, '\n'.join(lines))
    return f'literature/{path.name}'


# ---------------------------------------------------------------- run 读取
class RunState:
    """单个 run 的滚动状态。"""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem
        self.args: dict = {}
        self.evals: list = []
        self.hist: list = []
        self.target = 0
        self.mtime = 0.0
        self.early_stop: dict | None = None
        self.loaded_ok = False

    def load(self) -> bool:
        try:
            d = json.loads(self.path.read_text(encoding='utf-8'))
        except Exception:
            return False
        if not isinstance(d, dict):
            return False
        self.args = d.get('args') or {}
        self.evals = d.get('evals_hist') or []
        self.hist = d.get('hist') or []
        self.target = int(self.args.get('steps') or 0)
        self.early_stop = d.get('early_stop') or None
        self.mtime = self.path.stat().st_mtime
        self.loaded_ok = True
        return True

    # ---- 派生指标
    @property
    def step(self) -> int:
        if self.evals:
            return int(self.evals[-1].get('step') or 0)
        if self.hist:
            return int(self.hist[-1].get('step') or 0)
        return 0

    @property
    def done(self) -> bool:
        """完成 = 跑满目标步数 **或** 触发早停。

        早停是**正常结束**：课程覆盖的难度全部 ≥0.95 时训练主动停止并存档。
        漏判 early_stop 会把正常完成的 run 误报成 DEAD_RUN，
        而 DEAD_RUN 是致命级 → 会触发重启，毁掉已完成的实验。
        （实测踩到：s29/s67/s83 在 36000 步早停，被误报为 DEAD_RUN。）
        """
        if self.early_stop:
            return True
        return bool(self.target and self.step >= self.target)

    def is_nan(self) -> list[dict]:
        bad = []
        for h in self.hist:
            for k in ('loss', 'task'):
                v = h.get(k)
                if isinstance(v, float) and (v != v or v in (float('inf'), float('-inf'))):
                    bad.append({'step': h.get('step'), 'field': k, 'value': str(v)})
        return bad

    def is_spike(self) -> dict | None:
        losses = [h['loss'] for h in self.hist
                  if isinstance(h.get('loss'), (int, float))
                  and h['loss'] == h['loss'] and abs(h['loss']) != float('inf')]
        if len(losses) < 5:
            return None
        recent = losses[-1]
        med = sorted(losses[:-1])[len(losses[:-1]) // 2]
        if med > 0 and recent > med * SPIKE_FACTOR and recent > 1.0:
            return {'recent': recent, 'median': med,
                    'factor': round(recent / med, 2)}
        return None

    def collapse(self) -> dict | None:
        """某跳曾 ≥0.85，最近 <0.60。"""
        if len(self.evals) < 2:
            return None
        last = self.evals[-1]
        srcs = (('dev_r8_by_hop', 'r8h'), ('dev_r5_by_hop', 'r5h'))
        for src, pref in srcs:
            cur = last.get(src) or {}
            if not cur:
                continue
            for h, v in cur.items():
                if not isinstance(v, (int, float)):
                    continue
                best = v
                for e in self.evals:
                    b = (e.get(src) or {}).get(h)
                    if isinstance(b, (int, float)):
                        best = max(best, b)
                if best >= COLLAPSE_HIGH and v < COLLAPSE_LOW:
                    return {'hop': h, 'best': best, 'now': v, 'src': src}
        return None

    def seesaw(self) -> dict | None:
        if not self.evals:
            return None
        cur = self.evals[-1].get('dev_r8_by_hop') or {}
        vals = [v for v in cur.values() if isinstance(v, (int, float))]
        if len(vals) < 2:
            return None
        hi, lo = max(vals), min(vals)
        if hi - lo >= SEESAW_SPREAD and lo < SEESAW_FLOOR:
            return {'spread': round(hi - lo, 3), 'min': lo, 'max': hi}
        return None

    def stall(self) -> dict | None:
        if len(self.evals) < STALL_EVALS:
            return None
        if self.target and self.step < STALL_PROGRESS * self.target:
            return None
        r8 = [e.get('dev_r8') for e in self.evals
              if isinstance(e.get('dev_r8'), (int, float))]
        if len(r8) < STALL_EVALS:
            return None
        peak = max(r8)
        quiet = 0
        for v in reversed(r8):
            if v >= peak - 1e-9:
                break
            quiet += 1
        if quiet >= STALL_EVALS:
            return {'peak_dev_r8': peak, 'quiet_evals': quiet,
                    'step': self.step, 'target': self.target}
        return None

    def flat(self) -> dict | None:
        if len(self.evals) < FLAT_MIN_EVALS:
            return None
        last = self.evals[-1]
        rs = [last.get(f'dev_r{i}') for i in (1, 2, 3, 4, 5, 6, 8)]
        rs = [v for v in rs if isinstance(v, (int, float))]
        if len(rs) < 3:
            return None
        if max(rs) - min(rs) < FLAT_SPREAD and max(rs) < FLAT_LEVEL:
            return {'spread': round(max(rs) - min(rs), 4), 'level': max(rs)}
        return None

    def milestones(self) -> list[Path]:
        return sorted(self.path.parent.glob(self.name + '_MILESTONE_*.pt'))

    def latest_milestone(self) -> Path | None:
        ms = self.milestones()
        return ms[-1] if ms else None


# ---------------------------------------------------------------- 进程
def running_train_procs() -> list[tuple[int, str]]:
    """返回 [(pid, cmdline)]，只含 s71_ldm_iter4.py。"""
    out = []
    try:
        r = subprocess.run(['/bin/bash', '-c',
                            "ps -eo pid,args | grep '[s]71_ldm_iter4.py'"],
                           capture_output=True, text=True, timeout=20)
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            pid_s, _, rest = line.partition(' ')
            try:
                out.append((int(pid_s), rest))
            except ValueError:
                continue
    except Exception:
        pass
    return out


def pid_for_run(procs: list[tuple[int, str]], run_name: str) -> int | None:
    for pid, cmd in procs:
        if run_name + '.json' in cmd:
            return pid
    return None


def kill_pid(pid: int) -> bool:
    """按数字 PID 杀。**不用 pkill -f**（会匹配到自身命令行）。"""
    try:
        os.kill(pid, signal.SIGKILL)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- 检测
def detect(run: RunState, procs: list[tuple[int, str]],
           now: float) -> list[dict]:
    """返回本 run 的异常列表（不含动作）。"""
    ev: list[dict] = []

    nan = run.is_nan()
    spike = run.is_spike()
    # 致命级要求双证据：NaN 与 spike 互为佐证
    if nan and spike:
        ev.append({'event': 'NAN_LOSS', 'severity': 'fatal',
                   'evidence': {'nan': nan[:5], 'spike': spike}})
    elif nan:
        ev.append({'event': 'NAN_LOSS', 'severity': 'warn',
                   'evidence': {'nan': nan[:5], 'spike': None,
                                'note': 'single evidence; no action'}})

    if run.done:
        return ev

    # 只有「进程不在 + 未完成 + 文件有一阵子没更新」才判定为死亡。
    # 额外要求：进程表里确实没有这个 run 的 pid（而不是 pgrep 没匹配上）。
    pid = pid_for_run(procs, run.name)
    if pid is None and (now - run.mtime) > DEAD_MIN_AGE:
        ev.append({'event': 'DEAD_RUN', 'severity': 'fatal',
                   'evidence': {'step': run.step, 'target': run.target,
                                'age_sec': round(now - run.mtime),
                                'milestones': len(run.milestones())}})

    c = run.collapse()
    if c:
        ev.append({'event': 'COLLAPSE', 'severity': 'warn', 'evidence': c})
    s = run.seesaw()
    if s:
        ev.append({'event': 'SEESAW', 'severity': 'warn', 'evidence': s})
    st = run.stall()
    if st:
        ev.append({'event': 'STALL', 'severity': 'warn', 'evidence': st})
    fl = run.flat()
    if fl:
        ev.append({'event': 'FLAT_ALL_RANDOM', 'severity': 'warn',
                   'evidence': fl})
    return ev


# ---------------------------------------------------------------- 动作
def act_on(event: dict, run: RunState, procs: list[tuple[int, str]],
           results_dir: Path, guard_dir: Path, now: float,
           stop_new: list[bool]) -> dict:
    """执行（若允许）自动动作，返回动作描述。"""
    ev = event['event']
    name = run.name
    detail: dict = {}

    if ev == 'NAN_LOSS' and event['severity'] == 'fatal':
        pid = pid_for_run(procs, name)
        if pid:
            detail['killed_pid'] = pid
            detail['killed'] = kill_pid(pid)
        detail['action'] = 'kill_run'

    elif ev == 'DEAD_RUN':
        ms = run.latest_milestone()
        if ms is None:
            detail['action'] = 'record_only_no_milestone'
            return detail
        # 从 MILESTONE 重启到**新 --out**，原档保留
        m = re.search(r'_MILESTONE_s(\d+)\.pt$', ms.name)
        step = int(m.group(1)) if m else 0
        new_out = results_dir / f'{name}_RESUMED_s{step}.json'
        if new_out.exists():
            detail['action'] = 'already_resumed'
            detail['out'] = str(new_out)
            return detail
        cmd = run.args.get('_cmdline')
        if not cmd:
            detail['action'] = 'record_only_no_cmdline'
            return detail
        newcmd = re.sub(r'--out\s+\S+', f'--out {new_out}', cmd)
        newcmd = re.sub(r'--resume\s+\S+', '', newcmd)
        newcmd += f' --resume {ms}'
        try:
            with open(results_dir / f'{name}_RESUME.sh', 'w', encoding='utf-8') as f:
                f.write('#!/bin/bash\n' + newcmd + '\n')
            log = results_dir.parent / 'logs' / f'{name}_RESUMED.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, 'w') as lf:
                p = subprocess.Popen(['/bin/bash', '-c', newcmd],
                                     stdout=lf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            detail['action'] = 'resume_from_milestone'
            detail['milestone'] = str(ms)
            detail['new_out'] = str(new_out)
            detail['new_pid'] = p.pid
        except Exception as e:
            detail['action'] = 'resume_failed'
            detail['error'] = f'{type(e).__name__}: {e}'

    elif ev == 'DISK_PRESSURE':
        stop_new[0] = True
        detail['action'] = 'stop_new_launches'
    else:
        detail['action'] = 'record_only'
    return detail


# ---------------------------------------------------------------- 主循环
def scan_once(results_dir: Path, guard_dir: Path, apply_actions: bool,
              now: float, cooldown_sec: int = COOLDOWN_SEC,
              force: bool = False) -> dict:
    guard_dir.mkdir(parents=True, exist_ok=True)
    events_file = guard_dir / 'events.jsonl'
    state_file = guard_dir / 'state.json'

    runs: dict[str, RunState] = {}
    for p in sorted(results_dir.glob('*.json')):
        if p.name.startswith(('auto_loop_state', 'watchdog_state',
                              'campaign_status')):
            continue
        r = RunState(p)
        if r.load():
            runs[r.name] = r

    # 命令行持久化：进程死后 /proc 里就没了，必须提前存下来才能重启。
    # 优先用本守护进程之前记下的（cmds.json），其次用当前进程表。
    cmds_file = guard_dir / 'cmds.json'
    try:
        saved_cmds = json.loads(cmds_file.read_text(encoding='utf-8'))
    except Exception:
        saved_cmds = {}
    procs = running_train_procs()
    for pid, cmd in procs:
        m = re.search(r'--out\s+(\S+\.json)', cmd)
        if m:
            nm = Path(m.group(1)).stem
            saved_cmds[nm] = cmd
            if nm in runs:
                runs[nm].args['_cmdline'] = cmd
    # 之前记下的、现在进程已消失的 run，也能拿到命令行
    for nm, cmd in saved_cmds.items():
        if nm in runs and not runs[nm].args.get('_cmdline'):
            runs[nm].args['_cmdline'] = cmd
    # 兜底：从队列的 jobs.txt 重建命令行。
    # 关键：run 死掉时 /proc 里已经没有它，而守护进程可能是**之后**才启动的，
    # 此时 cmds.json 里也没有——只有队列的 job 定义还在。
    # jobs.txt 可能在 results/ 的同级目录，也可能在更上层（队列的工作目录）
    jobs_cands = [results_dir.parent / 'jobs.txt',
                  results_dir.parent.parent / 'jobs.txt',
                  Path('/mnt/workspace/jobs.txt')]
    for jobs in jobs_cands:
      if not jobs.exists():
        continue
      try:
            for line in jobs.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                m = re.search(r'--out\s+(\S+\.json)', line)
                if not m:
                    continue
                nm = Path(m.group(1)).stem
                if nm in runs and not runs[nm].args.get('_cmdline'):
                    runs[nm].args['_cmdline'] = line
            break
      except Exception:
            pass
    try:
        _atomic_write(cmds_file, json.dumps(saved_cmds, ensure_ascii=False, indent=2))
    except Exception:
        pass

    # 加载冷却状态
    try:
        state = json.loads(state_file.read_text(encoding='utf-8'))
    except Exception:
        state = {'cooldown': {}, 'seen': {}}
    cooldown = state.setdefault('cooldown', {})

    all_events: list[dict] = []
    stop_new = [False]

    for name, r in runs.items():
        for event in detect(r, procs, now):
            event['run'] = name
            key = f"{name}:{event['event']}"
            last = cooldown.get(key, 0)
            if not force and now - last < cooldown_sec:
                # 冷却期内：只更新最新证据，不重复写事件（避免刷屏）
                event['cooldown_suppressed'] = round(now - last)
                continue
            if event['severity'] == 'fatal' and apply_actions:
                detail = act_on(event, r, procs, results_dir, guard_dir,
                                now, stop_new)
            else:
                detail = {'action': 'record_only'}
            cooldown[key] = now
            event['action_detail'] = detail
            event['action'] = detail.get('action') or 'record_only'
            all_events.append(event)

    # 系统级检查
    try:
        st = os.statvfs(results_dir)
        free_gb = st.f_bavail * st.f_frsize / 1e9
    except Exception:
        free_gb = 999.0
    if free_gb < DISK_MIN_GB:
        ev = {'event': 'DISK_PRESSURE', 'severity': 'fatal',
              'evidence': {'free_gb': round(free_gb, 2),
                           'threshold': DISK_MIN_GB},
              'run': '(system)'}
        if apply_actions:
            ev['action_detail'] = act_on(ev, None, procs, results_dir,
                                         guard_dir, now, stop_new)
        all_events.append(ev)

    # 写事件（含文献）
    for event in all_events:
        name = event.get('run') or '(system)'
        event['ts'] = event.get('ts', now)
        event['run'] = name
        event_id = f"{_stamp(event['ts'])}_{event['event']}_{name}"
        event['event_id'] = event_id
        if event['severity'] == 'fatal':
            papers = search_arxiv(LITERATURE_QUERIES.get(event['event'], []))
            event['lit_file'] = write_literature(guard_dir, event_id,
                                                 event['event'], papers)
            event['lit_count'] = len(papers)
        event['note'] = 'observed only; no causal claim'
        _append_jsonl(events_file, event)

    # 状态
    per_run = {}
    for name, r in runs.items():
        per_run[name] = {
            'step': r.step, 'target': r.target, 'done': r.done,
            'task_loss': (r.hist[-1].get('task') if r.hist else None),
            'h6': ((r.evals[-1].get('dev_r5_by_hop') or {}).get('6')
                   if r.evals else None),
            'h7': ((r.evals[-1].get('confirm_r6_by_hop') or {}).get('7')
                   if r.evals else None),
            'alive': pid_for_run(procs, name) is not None,
        }
    state['cooldown'] = cooldown
    state['runs'] = per_run
    state['last_scan'] = now
    state['n_running'] = len(procs)
    state['free_gb'] = round(free_gb, 2)
    _atomic_write(state_file, json.dumps(state, ensure_ascii=False, indent=2))

    # 看板
    lines = [f'# 守护看板 {_stamp(now)}', '',
             f"- 运行中进程：{len(procs)}",
             f"- 可用空间：{free_gb:.1f} GB",
             f"- 停止启动新 run：{'是' if stop_new[0] else '否'}",
             '',
             f"{'run':34s} {'alive':>5s} {'step':>7s} {'target':>7s} "
             f"{'task':>7s} {'h6':>6s} {'h7':>6s}"]
    for name, v in sorted(per_run.items()):
        lines.append(f"{name[:34]:34s} {'Y' if v['alive'] else '-':>5s} "
                     f"{v['step']:>7d} {v['target']:>7d} "
                     f"{(v['task_loss'] if v['task_loss'] is not None else float('nan')):>7.4f} "
                     f"{(v['h6'] if v['h6'] is not None else float('nan')):>6.3f} "
                     f"{(v['h7'] if v['h7'] is not None else float('nan')):>6.3f}")
    _atomic_write(guard_dir / 'report.md', '\n'.join(lines) + '\n')

    return {'n_runs': len(runs), 'n_events': len(all_events),
            'n_running': len(procs), 'free_gb': round(free_gb, 1),
            'events': all_events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='/mnt/workspace/results/main')
    ap.add_argument('--guard', default='/mnt/workspace/results/guardian')
    ap.add_argument('--interval', type=int, default=60)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--no-actions', action='store_true',
                    help='只检测与记录，不做任何动作（演练用）')
    ap.add_argument('--drill', default='',
                    choices=['', 'kill', 'nan'],
                    help='注入一次故障做演练')
    ap.add_argument('--cooldown', type=int, default=COOLDOWN_SEC,
                    help='同一 run 同一异常的冷却秒数')
    ap.add_argument('--force', action='store_true',
                    help='忽略冷却，强制处理（演练用；生产勿开）')
    args = ap.parse_args()

    results_dir = Path(args.results)
    guard_dir = Path(args.guard)

    if args.drill == 'kill':
        procs = running_train_procs()
        if procs:
            pid = procs[0][0]
            print(f'[drill] killing pid {pid}: {procs[0][1][:90]}')
            kill_pid(pid)
        else:
            print('[drill] no training proc found')
        return

    if args.drill == 'nan':
        # 往某个 run 的 hist 里注入 NaN（只改副本，不动原档）
        cands = sorted(results_dir.glob('*.json'))
        if not cands:
            print('[drill] no run json')
            return
        target = cands[0]
        bak = target.with_suffix('.json.nandrill.bak')
        shutil.copy2(target, bak)
        d = json.loads(target.read_text(encoding='utf-8'))
        if d.get('hist'):
            d['hist'][-1]['loss'] = float('nan')
            d['hist'][-1]['task'] = float('nan')
        _atomic_write(target, json.dumps(d, ensure_ascii=False))
        print(f'[drill] injected NaN into {target.name} (backup at {bak.name})')
        return

    apply_actions = not args.no_actions
    print(f'[guardian] start results={results_dir} guard={guard_dir} '
          f'interval={args.interval} actions={"ON" if apply_actions else "OFF"}',
          flush=True)
    while True:
        try:
            s = scan_once(results_dir, guard_dir, apply_actions, _now(),
                         args.cooldown, args.force)
            if s['n_events']:
                print(f"[guardian] {s['n_events']} event(s), "
                      f"{s['n_running']} running, {s['free_gb']}GB free",
                      flush=True)
                for e in s['events']:
                    print(f"  - {e.get('event')} [{e.get('severity')}] "
                          f"{e.get('run','')} -> {e.get('action')}", flush=True)
        except Exception as e:
            import traceback
            print(f'[guardian] ERROR {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
