"""s78_agent_loop.py — 自主实验编排器（由 dsh headless 做判读与决策）。

设计原则：
  * **执行确定性**：实验队列、命令白名单、并发上限、预注册判据都在本脚本里，
    dsh 只负责"读摘要 → 给判读与下一步"。
  * **token 可控**：每次调用只送摘要（<2000 token）；有最小间隔与总调用上限。
  * **可中断/可恢复**：队列状态存 results/agent_queue_state.json，重启继续。

用法：
  python s78_agent_loop.py --init            # 生成队列状态（只做一次）
  python s78_agent_loop.py --run             # 主循环（后台跑）
  python s78_agent_loop.py --summary         # 看当前进度
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
STATE = RES / 'agent_queue_state.json'
REPORT = RES / 'agent_report.md'
PROTOCOL = ROOT / 'docs' / 'plans' / 'AGENT_PROTOCOL_DHES.md'
DSH_DIR = Path('/root/dsh-test')
DSH_SHIM = DSH_DIR / '_dsh.mjs'
DHOME = '/root/.dsh'
PY = '/root/miniconda3/bin/python'
DEADLINE = time.mktime(time.strptime('2026-10-27 16:30:00', '%Y-%m-%d %H:%M:%S'))

BASE = ('--pos none --persist-sup --step-sup --curriculum 2,3,4,5,6 '
        '--steps 24000 --bs 64 --lr 1e-3 --r-max 8 --eval-every 2000 '
        '--early-stop')

# ---------------- 实验队列（按协议 §3 的优先级） ----------------
QUEUE = [
    # Q1 hop7/8 外推（最高优先）
    ('Q1_extrap_s17', f'{BASE} --seed 17 --out results/R_Q1_extrap_s17.json', 'Q1'),
    ('Q1_extrap_s53', f'{BASE} --seed 53 --out results/R_Q1_extrap_s53.json', 'Q1'),
    # Q2 补种子做实成功率
    ('Q2_seed201', f'{BASE} --seed 201 --out results/R_Q2_seed201.json', 'Q2'),
    ('Q2_seed202', f'{BASE} --seed 202 --out results/R_Q2_seed202.json', 'Q2'),
    ('Q2_seed203', f'{BASE} --seed 203 --out results/R_Q2_seed203.json', 'Q2'),
    ('Q2_seed204', f'{BASE} --seed 204 --out results/R_Q2_seed204.json', 'Q2'),
    # Q3 s89 失败个案
    ('Q3_s89_w1', f'{BASE} --seed 89 --step-weight 1.0 --out results/R_Q3_s89_w1.json', 'Q3'),
    ('Q3_s89_long', f'--pos none --persist-sup --step-sup --curriculum 2,3,4,5,6,6,6 '
                    f'--steps 36000 --bs 64 --lr 1e-3 --r-max 8 --eval-every 2000 '
                    f'--early-stop --seed 89 --out results/R_Q3_s89_long.json', 'Q3'),
    ('Q3_seed189', f'{BASE} --seed 189 --out results/R_Q3_seed189.json', 'Q3'),
    # Q4 参数效率正式对照
    ('Q4_dense1_s17', f'{BASE} --variant dense --dense-mid 1 --seed 17 '
                      f'--out results/R_Q4_dense1_s17.json', 'Q4'),
    ('Q4_dense3_s17', f'{BASE} --variant dense --dense-mid 3 --seed 17 '
                      f'--out results/R_Q4_dense3_s17.json', 'Q4'),
    ('Q4_dense5_s17', f'{BASE} --variant dense --dense-mid 5 --seed 17 '
                      f'--out results/R_Q4_dense5_s17.json', 'Q4'),
    ('Q4_dense1_s53', f'{BASE} --variant dense --dense-mid 1 --seed 53 '
                      f'--out results/R_Q4_dense1_s53.json', 'Q4'),
    ('Q4_dense3_s53', f'{BASE} --variant dense --dense-mid 3 --seed 53 '
                      f'--out results/R_Q4_dense3_s53.json', 'Q4'),
    # Q5 消融（step-sup 的独立贡献）
    ('Q5_nostep_s17', f'--pos none --persist-sup --curriculum 2,3,4,5,6 '
                      f'--steps 24000 --bs 64 --lr 1e-3 --r-max 8 --eval-every 2000 '
                      f'--early-stop --seed 17 --out results/R_Q5_nostep_s17.json', 'Q5'),
    ('Q5_nostep_s71', f'--pos none --persist-sup --curriculum 2,3,4,5,6 '
                      f'--steps 24000 --bs 64 --lr 1e-3 --r-max 8 --eval-every 2000 '
                      f'--early-stop --seed 71 --out results/R_Q5_nostep_s71.json', 'Q5'),
    ('Q5_steponly_s17', f'--pos none --step-sup --curriculum 2,3,4,5,6 '
                        f'--steps 24000 --bs 64 --lr 1e-3 --r-max 8 --eval-every 2000 '
                        f'--early-stop --seed 17 --out results/R_Q5_steponly_s17.json', 'Q5'),
]

MAX_PARALLEL = 6
MAX_CALLS = 30
MIN_CALL_GAP = 600       # 秒
TOKEN_WARN = 2500

# 只读诊断脚本（同步执行，不占训练并发）
DIAG_QUEUE = [
    {'tag': 'DIAG_bias_s53', 'script': 's79_confirm_bias.py',
     'phase': 'Q1.5',
     'args': '--ckpt results/R_Q1_extrap_s53_ckpt.pt --out results/bias_s53.json'},
    {'tag': 'DIAG_bias_s17', 'script': 's79_confirm_bias.py',
     'phase': 'Q1.5',
     'args': '--ckpt results/R_Q1_extrap_s17_ckpt.pt --out results/bias_s17.json'},
]


# ---------------- 工具 ----------------
def stem_of(args: str, tag: str = '') -> str:
    """从 --out 解析文件名主干（tag 可能已带 R_ 前缀，不能盲目拼接）。"""
    m = re.search(r'--out\s+(\S+)', args or '')
    if m:
        return m.group(1).rsplit('/', 1)[-1].replace('.json', '')
    return tag


def now() -> str:
    return time.strftime('%H:%M:%S')


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding='utf-8'))
    items = [{'tag': t, 'args': a, 'phase': p, 'status': 'pending'}
             for t, a, p in QUEUE]
    items += [{**d, 'status': 'pending'} for d in DIAG_QUEUE]
    return {'items': items, 'calls': 0, 'last_call': 0, 'batch': 0}


def save_state(st: dict):
    STATE.write_text(json.dumps(st, indent=2), encoding='utf-8')


def report_append(text: str):
    with open(REPORT, 'a', encoding='utf-8') as f:
        f.write(text.rstrip() + '\n\n')


def procs_by_tag() -> dict:
    """返回 {tag: pid}，用于判断是否在跑。"""
    out = {}
    try:
        r = subprocess.run(['pgrep', '-af', 's71_ldm_iter4'],
                           capture_output=True, text=True, timeout=15)
        for line in r.stdout.splitlines():
            m = re.search(r'--out\s+(\S+?)(\.json)?\s', line + ' ')
            if m:
                base = m.group(1).rsplit('/', 1)[-1]
                base = base[:-5] if base.endswith('.json') else base
                out[base] = line.split()[0]      # 每个 run 只记一个 key
    except Exception:
        pass
    return out


def gpu_free_mb() -> int:
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                            '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().split('\n')[0])
    except Exception:
        return 0


def launch(args: str, tag: str = '') -> bool:
    # 若已有续跑状态，自动 --resume，避免重启丢进度
    _st = RES / f'{stem_of(args, tag)}_state.pt'
    if _st.exists() and '--resume' not in args:
        args = args + ' --resume'
    log = f'{RES}/{args.split("--out ")[1].split()[0].split("/")[-1]}'.replace(
        '.json', '.log')
    cmd = (f'cd {ROOT} && setsid nohup {PY} s71_ldm_iter4.py {args} '
           f'> {log} 2>&1 < /dev/null & disown')
    try:
        subprocess.run(['bash', '-c', cmd], timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def summarize_one(path: Path) -> dict:
    """紧凑摘要：只取判读需要的数字。"""
    try:
        d = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {'name': path.stem, 'error': 'unreadable'}
    ev = d.get('evals_hist') or []
    if not ev:
        return {'name': path.stem, 'error': 'no_evals'}
    best = max(ev, key=lambda e: e.get('dev_r8', 0))
    bh = best.get('dev_r8_by_hop') or {}
    n_solved = sum(1 for v in bh.values() if v >= 0.95)
    last = ev[-1]
    out = {
        'name': path.stem,
        'steps': last.get('step'),
        'peak_r8': round(best.get('dev_r8', 0), 3),
        'peak_at': best.get('step'),
        'solved_hops': f'{n_solved}/{len(bh)}' if bh else '?',
        'by_hop': {k: round(v, 2) for k, v in bh.items()},
        'final_r8': round(last.get('dev_r8', 0), 3),
        'early_stop': (d.get('early_stop')
                       or (ev[-1].get('early_stop') if ev else None)),
        'stopped_short': (bool((d.get('args') or {}).get('steps'))
                          and last.get('step', 0)
                          < (d.get('args') or {}).get('steps')
                          and not (d.get('early_stop')
                                   or (ev[-1].get('early_stop') if ev else None))),
        'confirm_r8': round(last.get('confirm_r8', 0), 3)
        if 'confirm_r8' in last else None,
        'confirm_by_hop': last.get('confirm_r6_by_hop'),
        'nv_milestones': len(list(RES.glob(f'{path.stem}_MILESTONE_*.pt'))),
    }
    if 'variant' in (d.get('args') or {}):
        out['variant'] = d['args']['variant']
    return out


def batch_summary(tags) -> str:
    lines = []
    for t in tags:
        p = RES / f'{t}.json'
        if p.exists():
            s = summarize_one(p)
            lines.append(json.dumps(s, ensure_ascii=False))
        else:
            lines.append(json.dumps({'name': t, 'error': 'missing'}))
    return '\n'.join(lines)


def call_dsh(prompt: str, st: dict) -> str:
    """调用 dsh headless；返回其输出。"""
    env = dict(os.environ)
    # 从 .env 读 key（不打印、不落盘到别处）
    envfile = DSH_DIR / '.env'
    if envfile.exists():
        for line in envfile.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    env['DSH_HOME'] = DHOME
    try:
        r = subprocess.run(['node', str(DSH_SHIM), 'headless', prompt],
                           cwd=str(DSH_DIR), env=env, capture_output=True,
                           text=True, timeout=600)
        out = (r.stdout or '').strip()
        err = (r.stderr or '').strip()
        if not out and err:
            out = f'[dsh stderr] {err[-800:]}'
        return out
    except subprocess.TimeoutExpired:
        return '[dsh timeout 600s]'
    except Exception as e:
        return f'[dsh error] {type(e).__name__}: {e}'


def build_prompt(batch_no: int, tags, summary: str) -> str:
    return (
        f'你是 LDM 循环模型实验的自主监督者。\n'
        f'先读 /root/ldm/docs/plans/LDM_BACKGROUND_FOR_AGENT.md（完整背景），'
        f'再读同目录 AGENT_PROTOCOL_DHES.md（执行手册）。\n'
        f'注意：不要读 S08_CAMPAIGN_LOG.md / TRAINING_PLAN_V6*.md / '
        f'pkg_hybridloop4b/（无关的另一条线）。\n'
        f'这是第 {batch_no} 批实验结果。\n'
        f'预注册判据：五级全解 = hop2-6 全部 >=0.95；参数效率 = loop 达标且 '
        f'dense 同预算 <0.7；外推 = confirm hop7/8 >=0.8。\n\n'
        f'结果摘要（JSONL）：\n{summary}\n\n'
        f'请只输出以下四项，简洁，不要复述数据：\n'
        f'1) 判读：每项对照判据给出 达标/未达标/部分\n'
        f'2) 新证据：是否需要新建/更新假说（给出 H-0XX 编号建议与状态）\n'
        f'3) 下一步：从队列继续即可，或给出具体调整建议\n'
        f'4) 风险：任何异常信号（如某 hop 归零、步数异常）\n\n'
        '如果你需要**新增实验**，在回答末尾附一个 JSON 代码块（否则不要附）：\n'
        '```json\n'
        '{"add": [{"tag": "R_unique_name", "phase": "D1", '
        '"args": "--pos none --persist-sup --step-sup --curriculum 2,3,4,5,6 '
        '--steps 24000 --bs 64 --lr 1e-3 --r-max 8 --eval-every 2000 '
        '--early-stop --seed 301 --out results/R_unique_name.json"}]}\n'
        '```\n'
        '约束：tag 全局唯一；args 只能用白名单 flag；--out 必须 results/R_*.json；'
        '队列上限 40 项；不得修改算法逻辑。\n'
    )


# ---------------- dsh 动作协议（自主增补实验） ----------------
ALLOWED_FLAGS = {
    '--pos', '--persist-sup', '--step-sup', '--curriculum', '--steps',
    '--bs', '--lr', '--r-max', '--seed', '--early-stop', '--eval-every',
    '--variant', '--dense-mid', '--film-rounds', '--n-heads-read',
    '--step-weight', '--cap-reach', '--ponder', '--lam-prior', '--beta-kl',
    '--taus', '--out', '--resume', '--init-step', '--ema-decay',
    '--bottleneck-dim', '--width', '--joint-exit',
}
MAX_ITEMS = 40
FORBIDDEN = (';', '&&', '||', '|', '`', '$', '\n', '>', '<')


def parse_actions(text: str):
    """从 dsh 输出中提取 JSON 动作块（```json ... ``` 或裸 {...}）。"""
    import re
    blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S)
    if not blocks:
        m = re.search(r'(\{\s*"add"\s*:.*\})', text, re.S)
        if m:
            blocks = [m.group(1)]
    for b in blocks:
        try:
            return json.loads(b)
        except Exception:
            continue
    return None


def validate_action(item):
    """返回 (ok, tag, args, reason)。"""
    tag = str(item.get('tag', '')).strip()
    a = str(item.get('args', '')).strip()
    if not tag or not a:
        return False, tag, a, 'tag/args 为空'
    if any(t in a for t in FORBIDDEN):
        return False, tag, a, 'args 含禁止字符'
    if '--out results/R_' not in a:
        return False, tag, a, 'out 必须为 results/R_*.json'
    # 每个 flag 必须在白名单
    toks = a.split()
    for i, t in enumerate(toks):
        if t.startswith('--') and t not in ALLOWED_FLAGS:
            return False, tag, a, f'flag 不在白名单: {t}'
    return True, tag, a, ''


def apply_actions(st, act, batch_no):
    """把 dsh 的动作写进队列；返回应用摘要。"""
    if not isinstance(act, dict):
        return 'no-action'
    adds = act.get('add') or []
    if not isinstance(adds, list):
        return 'bad-add'
    tags = {i['tag'] for i in st['items']}
    applied, rejected = [], []
    for it in adds:
        if len(st['items']) >= MAX_ITEMS:
            rejected.append(('', '队列已达上限'))
            break
        ok, tag, a, why = validate_action(it)
        if not ok:
            rejected.append((tag, why))
            continue
        if tag in tags:
            rejected.append((tag, '重复 tag'))
            continue
        st['items'].append({'tag': tag, 'args': a,
                            'phase': str(it.get('phase', f'D{batch_no}')),
                            'status': 'pending'})
        tags.add(tag)
        applied.append(tag)
    save_state(st)
    msg = f'应用 {len(applied)} 项: {", ".join(applied) or "无"}'
    if rejected:
        msg += ' | 拒绝: ' + '; '.join(f'{t}({w})' for t, w in rejected)
    return msg


def run_script_item(it: dict):
    """同步跑只读诊断脚本，返回 (rc, 输出尾部)。"""
    cmd = f'cd {ROOT} && {PY} {it["script"]} {it["args"]}'
    try:
        r = subprocess.run(['bash', '-c', cmd], capture_output=True,
                           text=True, timeout=1800)
        tail = ((r.stdout or '')[-1500:] + (r.stderr or '')[-800:]).strip()
        return r.returncode, tail
    except subprocess.TimeoutExpired:
        return -1, '[timeout 1800s]'
    except Exception as e:
        return -1, f'[error] {type(e).__name__}: {e}'


def pack_results(tag: str = 'final') -> str:
    """打包 results(不含 .pt) + docs + 代码，返回包路径。"""
    ts = time.strftime('%m%d_%H%M')
    out = ROOT / f'package_{tag}_{ts}.tgz'
    try:
        subprocess.run(
            ['tar', 'czf', str(out), '--exclude=*.pt', '--exclude=*.lock',
             'results', 'docs', 's63_ldm_model.py',
             's66_ldm_mvt_gen.py', 's71_ldm_iter4.py', 's77_watchdog.py',
             's78_agent_loop.py', 's79_confirm_bias.py'],
            cwd=str(ROOT), capture_output=True, timeout=600)
        return str(out)
    except Exception as e:
        return f'[pack failed] {e}'


def upload_modelscope(tag: str = 'auto') -> str:
    """把最新包 + 报告 + 档案上传到魔搭数据集库（失败不影响本地）。"""
    token_file = Path('/root/.ms_token')
    if not token_file.exists():
        return '[skip] no /root/.ms_token'
    pkgs = sorted(ROOT.glob('package_*.tgz'), key=lambda p: p.stat().st_mtime)
    if not pkgs:
        return '[skip] no package'
    pkg = pkgs[-1]
    code = f'''
import json
from pathlib import Path
from modelscope.hub.api import HubApi
api = HubApi()
api.login(Path("/root/.ms_token").read_text().strip())
rid = "Yoneko/ldm-feasibility-testkit"
api.upload_file(repo_id=rid, path_or_fileobj="{pkg}",
                path_in_repo="autopilot/{pkg.name}", repo_type="dataset")
api.upload_file(repo_id=rid,
                path_or_fileobj="/root/ldm/results/agent_report.md",
                path_in_repo="autopilot/agent_report.md", repo_type="dataset")
for f in sorted(Path("/root/ldm/docs/hypotheses").glob("*.md")):
    api.upload_file(repo_id=rid, path_or_fileobj=str(f),
                    path_in_repo="autopilot/hypotheses/"+f.name,
                    repo_type="dataset")
print("UPLOAD_OK")
'''
    try:
        r = subprocess.run([PY, '-c', code], capture_output=True, text=True,
                           timeout=900)
        return ((r.stdout or '')[-200:] + (r.stderr or '')[-400:]).strip()
    except subprocess.TimeoutExpired:
        return '[upload timeout 900s]'
    except Exception as e:
        return f'[upload error] {type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args()

    RES.mkdir(exist_ok=True)
    st = load_state()

    if a.init:
        if STATE.exists():
            print(f'{STATE} 已存在，不覆盖')
        else:
            save_state(st)
            print(f'init: {len(st["items"])} 项队列已写入')
        return

    if a.summary:
        done = [i for i in st['items'] if i['status'] == 'done']
        run_ = [i for i in st['items'] if i['status'] == 'running']
        pend = [i for i in st['items'] if i['status'] == 'pending']
        print(f'done={len(done)} running={len(run_)} pending={len(pend)} '
              f'calls={st["calls"]}/{MAX_CALLS}')
        for i in st['items']:
            print(f"  [{i['status']:8s}] {i['tag']} ({i['phase']})")
        if REPORT.exists():
            print('\n--- report tail ---')
            print(REPORT.read_text(encoding='utf-8')[-1500:])
        return

    if not (a.run or a.once):
        ap.print_help()
        return

    report_append(f'# 自主实验开始 {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
                  f'队列 {len(st["items"])} 项，并发上限 {MAX_PARALLEL}，'
                  f'调用上限 {MAX_CALLS}，服务器截止 {time.ctime(DEADLINE)}')

    while True:
        if time.time() > DEADLINE:
            report_append(f'## {now()} 到达截止时间，停止新实验')
            pkg = pack_results('deadline')
            report_append(f'**打包**：{pkg}')
            report_append(f'**魔搭上传**：{upload_modelscope("deadline")}')
            break

        # 只读诊断任务（同步，不占并发）
        for it in st['items']:
            if it.get('script') and it['status'] == 'pending':
                report_append(f'- {now()} 运行诊断 {it["tag"]}')
                rc, tail = run_script_item(it)
                it['status'] = 'done' if rc == 0 else 'failed'
                report_append(f'**诊断 {it["tag"]}** rc={rc}\n```\n{tail}\n```')
                save_state(st)
                break

        # 幂等补齐诊断任务（防被旧内存状态覆盖）
        have = {i['tag'] for i in st['items']}
        added = False
        for d in DIAG_QUEUE:
            if d['tag'] not in have:
                st['items'].insert(0, {**d, 'status': 'pending'})
                added = True
        if added:
            save_state(st)

        alive = procs_by_tag()
        # 更新状态：进程消失 或 json 显示 early_stop/到目标 都算完成
        for it in st['items']:
            tag = it['tag']
            p = RES / f'{stem_of(it.get("args", ""), tag)}.json'
            if it['status'] != 'running':
                continue
            _bt = tag[2:] if tag.startswith('R_') else tag
            finished = (tag not in alive) and (_bt not in alive)
            if p.exists():
                try:
                    d = json.loads(p.read_text(encoding='utf-8'))
                    if d.get('early_stop'):
                        finished = True
                    ev = d.get('evals_hist') or []
                    if ev:
                        steps = (d.get('args') or {}).get('steps')
                        if steps and ev[-1].get('step', 0) >= steps:
                            finished = True
                except Exception:
                    pass
            if finished:
                # 未达目标步数且无早停 → 视为中断，回到 pending
                _partial = False
                if p.exists():
                    try:
                        d2 = json.loads(p.read_text(encoding='utf-8'))
                        ev2 = d2.get('evals_hist') or []
                        tgt = (d2.get('args') or {}).get('steps')
                        _es2 = d2.get('early_stop') or (
                            ev2[-1].get('early_stop') if ev2 else None)
                        if (tgt and ev2 and ev2[-1].get('step', 0) < tgt
                                and not _es2):
                            _partial = True
                    except Exception:
                        pass
                it['status'] = 'pending' if _partial else 'done'
        save_state(st)

        # 启动新实验
        launched = 0
        for it in st['items']:
            if it['status'] != 'pending':
                continue
            if len(alive) + launched >= MAX_PARALLEL:
                break
            if gpu_free_mb() < 2500:
                break
            if launch(it['args'], it['tag']):
                it['status'] = 'running'
                launched += 1
                report_append(f'- {now()} 启动 {it["tag"]} ({it["phase"]})')
        save_state(st)

        # 判读：只要有新完成的实验就触发（不等待其他 run 结束）
        done = [i for i in st['items'] if i['status'] == 'done']
        pending = [i for i in st['items'] if i['status'] == 'pending']
        if (st['calls'] < MAX_CALLS
                and time.time() - st['last_call'] >= MIN_CALL_GAP):
            new_done = [stem_of(i.get('args', ''), i['tag']) for i in done
                        if i['tag'] not in st.get('reported', [])]
            if new_done:
                st['batch'] = st.get('batch', 0) + 1
                summary = batch_summary(new_done)
                prompt = build_prompt(st['batch'], new_done, summary)
                out = call_dsh(prompt, st)
                report_append(
                    '## ' + now() + ' 批次 ' + str(st['batch']) + ' 判读（dsh）\n'
                    '实验：' + ', '.join(new_done) + '\n\n' + out)
                st['calls'] += 1
                st['last_call'] = time.time()
                st.setdefault('reported', []).extend(new_done)
                act = parse_actions(out)
                verdict = apply_actions(st, act, st['batch'])
                report_append(f'**动作**：{verdict}')
                save_state(st)
        if not pending and not [i for i in st['items']
                                if i['status'] == 'running']:
            report_append(f'## {now()} 队列全部完成')
            pkg = pack_results('done')
            report_append(f'**打包**：{pkg}')
            report_append(f'**魔搭上传**：{upload_modelscope("done")}')
            break

        # 每 2 小时做一次增量备份（防实例意外释放）
        if time.time() - st.get('last_pack', 0) > 7200:
            pkg = pack_results('snap')
            report_append(f'- {now()} 周期备份：{pkg}')
            st['last_pack'] = time.time()
            save_state(st)

        if a.once:
            break
        time.sleep(120)

    report_append(f'## {now()} 主循环退出')
    print('agent_loop exited')


if __name__ == '__main__':
    main()
