"""s78_agent_bridge.py — watchdog 异常 → dsh 诊断 的桥接层。

职责（与 watchdog 分工）：
  - watchdog（s77）：检测异常、自动停灾变、写 watchdog.log/state —— 已在跑
  - 本脚本：轮询 watchdog 状态，发现"无法自动处理的异常"时，
    调用一次 dsh headless 生成诊断报告（中文），写入 results/agent_report.md

Token 控制措施（硬性）：
  1. 每类异常每天最多触发 2 次（去重计数）
  2. 每次调用的上下文为摘要（≤2000 字符），不发原始 json
  3. DONE/正常完成不触发
  4. 默认 5 分钟轮询一次，dsh 调用限 240s 超时

用法（服务器后台）：
  nohup python s78_agent_bridge.py --interval 300 > results/agent_bridge.out 2>&1 &
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'results'
STATE = RES / 'watchdog_state.json'
REPORT = RES / 'agent_report.md'
TRIGGER_LOG = RES / 'agent_triggers.json'
DSH = '/root/dsh-test/node_modules/.bin/dsh'

# 哪些事件值得请示 LLM（DONE/STALL 不在其中——可自动处理或等人工）
WORTHY = {'DEAD_NOT_DONE', 'BAD_LOSS', 'CATASTROPHE'}
DAILY_LIMIT_PER_TYPE = 2


def load_triggers() -> dict:
    if TRIGGER_LOG.exists():
        try:
            return json.loads(TRIGGER_LOG.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {'day': '', 'counts': {}}


def save_triggers(t: dict):
    TRIGGER_LOG.write_text(json.dumps(t, indent=2), encoding='utf-8')


def today() -> str:
    return datetime.date.today().isoformat()


def summarize_run(name: str, info: dict) -> str:
    """把 run 状态压成 ≤300 字符的摘要。"""
    return (f"{name}: step={info.get('step')}/{info.get('target')} "
            f"peak={info.get('peak')}@{info.get('peak_at')} "
            f"zeros={info.get('zeros_last')} "
            f"events={','.join(info.get('events', []))}")


def build_prompt(states: dict, worthies: list) -> str:
    lines = [
        '你是训练监督代理。以下是今晚 LDM 循环模型训练的异常摘要。',
        '请给出：(1)最可能原因（一两句）；(2)建议处置（具体命令或动作）；',
        '(3)是否需要中止相关实验（是/否）。总长不超过 300 字，不要复述背景。',
        '',
        '异常 run 摘要：',
    ]
    for name in worthies[:4]:                       # 最多 4 个 run
        info = states.get(name) or {}
        lines.append('- ' + summarize_run(name, info))
    lines.append('')
    lines.append('背景：任务为 1.4M 循环 Transformer 解 2-6 跳图推理，'
                 'hop5 已知存在冷启动门槛（约半数种子卡在 0.5）。'
                 '里程碑存档按 _MILESTONE_*.pt 命名，早停档 _EARLYSTOP.pt。')
    return '\n'.join(lines)


def call_dsh(prompt: str, timeout: int = 240) -> str:
    env = dict(os.environ)
    # subprocess 不继承 bashrc：显式带 key（从 .env 读，避免写死）
    env_path = Path('/root/dsh-test/.env')
    if env_path.exists():
        for ln in env_path.read_text(encoding='utf-8').splitlines():
            if ln.startswith('DEEPSEEK_API_KEY='):
                env['DEEPSEEK_API_KEY'] = ln.split('=', 1)[1].strip()
    try:
        r = subprocess.run([DSH, '--profile', 'headless', prompt],
                           capture_output=True, text=True, timeout=timeout,
                           env=env, cwd=str(ROOT))
        return (r.stdout or '').strip()[:3000] or f'(stderr: {r.stderr[:500]})'
    except Exception as e:
        return f'(dsh call failed: {type(e).__name__}: {e})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--interval', type=int, default=300)
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()

    print(f'[agent-bridge] start interval={args.interval}', flush=True)
    t = load_triggers()
    if t.get('day') != today():
        t = {'day': today(), 'counts': {}}

    while True:
        if STATE.exists():
            try:
                st = json.loads(STATE.read_text(encoding='utf-8'))
            except Exception:
                st = None
            if st:
                runs = st.get('runs', {})
                worthies = []
                for name, info in runs.items():
                    evs = set(info.get('events', []))
                    for w in WORTHY:
                        if w in evs:
                            key = f'{today()}:{w}'
                            if t['counts'].get(key, 0) < DAILY_LIMIT_PER_TYPE:
                                worthies.append(name)
                                break
                if worthies:
                    prompt = build_prompt(runs, worthies)
                    print(f'[agent-bridge] calling dsh for {worthies}',
                          flush=True)
                    answer = call_dsh(prompt)
                    for name in worthies:
                        for w in WORTHY:
                            if w in (runs.get(name, {}).get('events', [])):
                                key = f'{today()}:{w}'
                                t['counts'][key] = t['counts'].get(key, 0) + 1
                    save_triggers(t)
                    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    with open(REPORT, 'a', encoding='utf-8') as f:
                        f.write(f'\n## {stamp}\n\n'
                                f'**触发**：{", ".join(worthies)}\n\n'
                                f'**摘要**：\n```\n'
                                + '\n'.join(summarize_run(n, runs.get(n) or {})
                                            for n in worthies)
                                + '\n```\n\n**dsh 诊断**：\n\n'
                                + answer + '\n\n---\n')
                    print(f'[agent-bridge] report appended ({len(answer)} chars)',
                          flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
