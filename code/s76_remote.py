"""s76_remote.py — 远程实验执行器（服务器 2080Ti 11G）。

用法：
  python s76_remote.py setup                     # 上传代码+数据、装依赖、冒烟
  python s76_remote.py run --args "..."          # 前台跑一条训练
  python s76_remote.py launch --tag NAME --args "..."   # 后台跑（nohup）
  python s76_remote.py poll --tag NAME           # 查后台状态
  python s76_remote.py fetch --tag NAME          # 取回结果 json
  python s76_remote.py shell "任意命令"           # 直通
  python s76_remote.py gpu                       # GPU 状态
"""
from __future__ import annotations

import argparse
import json
import os
import posixpath
import sys
import time
from pathlib import Path

import paramiko

HOST = 'region-41.seetacloud.com'
PORT = 53402
USER = 'root'
REMOTE = '/root/ldm'
PY = '/root/miniconda3/bin/python'

ROOT = Path(__file__).parent

# 凭据不入库：优先环境变量 LDM_SSH_PWD，其次本地未纳入版本控制的
# .ldm_ssh_pwd 文件（权限 600）。两者都没有时 conn() 明确报错，
# 而不是把密码写死在源码里。
_PWD_FILE = ROOT / '.ldm_ssh_pwd'


def _pwd() -> str:
    v = os.environ.get('LDM_SSH_PWD', '').strip()
    if v:
        return v
    if _PWD_FILE.exists():
        return _PWD_FILE.read_text(encoding='utf-8').strip()
    raise SystemExit(
        '未找到 SSH 凭据。请设环境变量 LDM_SSH_PWD，'
        f'或写入 {_PWD_FILE}（权限 600）。不要把密码提交到仓库。')


def conn():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=_pwd(), timeout=30)
    return c


def sh(c, cmd, timeout=180, quiet=False):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', 'replace')
    err = e.read().decode('utf-8', 'replace')
    code = o.channel.recv_exit_status()
    if not quiet:
        if out.strip():
            print(out.rstrip())
        if err.strip() and code != 0:
            print('[stderr]', err.rstrip()[:2000])
    return code, out, err


def put_dir(sftp, local: Path, remote: str, pattern: str):
    n = 0
    for p in sorted(local.glob(pattern)):
        if p.is_file():
            sftp.put(str(p), posixpath.join(remote, p.name))
            n += 1
    return n


def cmd_setup(a):
    c = conn()
    sftp = c.open_sftp()
    sh(c, f'mkdir -p {REMOTE}/data/ldm_mvt5 {REMOTE}/results')
    files = ['s63_ldm_model.py', 's65_ldm_gpu_smoke.py', 's66_ldm_mvt_gen.py',
             's67_ldm_exit.py', 's71_ldm_iter4.py', 's73_ldm_dashboard.py',
             's74_ldm_residual_diag.py']
    n = 0
    for pat in ('s6*.py', 's7*.py'):
        n += put_dir(sftp, ROOT, REMOTE, pat)
    print(f'[upload] code files: {n}')
    # 数据（只要训练/评测用到的三个）
    for f in ('train.jsonl', 'dev_random.jsonl', 'confirm_depth.jsonl'):
        lp = ROOT / 'data' / 'ldm_mvt5' / f
        if lp.exists():
            sftp.put(str(lp), f'{REMOTE}/data/ldm_mvt5/{f}')
            print(f'[upload] {f} ({lp.stat().st_size/1e6:.1f} MB)')
    sftp.close()
    # torch 2.5.1 兼容性检查
    sh(c, f'cd {REMOTE} && {PY} -c "\n'
          'import torch\n'
          'print(\'torch\', torch.__version__)\n'
          'import s63_ldm_model as M\n'
          'm = M.build_model(\'loop\', n_rounds=8, n_core=1, round_in_core=True, pos=\'none\').cuda()\n'
          'print(\'params\', m.param_report()[\'total\'])\n'
          'print(\'OK\')\n'
          '"', timeout=300)
    c.close()


def cmd_run(a):
    c = conn()
    cmd = f'cd {REMOTE} && {PY} s71_ldm_iter4.py {a.args}'
    print('[run]', cmd, flush=True)
    _, o, e = c.exec_command(cmd, timeout=a.timeout)
    out = o.read().decode('utf-8', 'replace')
    err = e.read().decode('utf-8', 'replace')
    print(out[-6000:])
    if err.strip():
        print('[stderr]', err[-3000:])
    c.close()


def cmd_launch(a):
    c = conn()
    log = f'{REMOTE}/results/{a.tag}.log'
    cmd = (f'cd {REMOTE} && nohup {PY} s71_ldm_iter4.py {a.args} '
           f'> {log} 2>&1 & echo PID=$!')
    code, out, err = sh(c, cmd)
    print(f'[launched] {a.tag} -> {log}')
    c.close()


def cmd_poll(a):
    c = conn()
    sh(c, f'ls -la {REMOTE}/results/{a.tag}* 2>/dev/null')
    sh(c, f'tail -3 {REMOTE}/results/{a.tag}.log 2>/dev/null')
    sh(c, 'nvidia-smi --query-gpu=memory.used,utilization.gpu '
          '--format=csv,noheader')
    c.close()


def cmd_fetch(a):
    c = conn()
    sftp = c.open_sftp()
    local = ROOT / 'results'
    local.mkdir(exist_ok=True)
    got = []
    for f in (f'{a.tag}.json', f'{a.tag}.log'):
        try:
            sftp.get(f'{REMOTE}/results/{f}', str(local / f'remote_{f}'))
            got.append(f)
        except IOError:
            pass
    # milestone/ckpt
    try:
        for fn in sftp.listdir(f'{REMOTE}/results'):
            if a.tag in fn and fn.endswith('.pt'):
                sftp.get(f'{REMOTE}/results/{fn}', str(local / f'remote_{fn}'))
                got.append(fn)
    except IOError:
        pass
    sftp.close()
    c.close()
    print('[fetched]', got)


def cmd_shell(a):
    c = conn()
    sh(c, a.cmd, timeout=a.timeout)
    c.close()


def cmd_gpu(a):
    c = conn()
    sh(c, 'nvidia-smi')
    c.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('setup'); p.set_defaults(fn=cmd_setup)
    p = sub.add_parser('run'); p.add_argument('--args', required=True)
    p.add_argument('--timeout', type=int, default=7200); p.set_defaults(fn=cmd_run)
    p = sub.add_parser('launch'); p.add_argument('--tag', required=True)
    p.add_argument('--args', required=True); p.set_defaults(fn=cmd_launch)
    p = sub.add_parser('poll'); p.add_argument('--tag', required=True)
    p.set_defaults(fn=cmd_poll)
    p = sub.add_parser('fetch'); p.add_argument('--tag', required=True)
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser('shell'); p.add_argument('cmd')
    p.add_argument('--timeout', type=int, default=300); p.set_defaults(fn=cmd_shell)
    p = sub.add_parser('gpu'); p.set_defaults(fn=cmd_gpu)

    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
