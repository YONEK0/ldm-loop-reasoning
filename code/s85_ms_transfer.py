"""s85_ms_transfer.py — 本地↔服务器 经魔搭(ModelScope)数据集中转。

约定（用户明确要求，2026-10-27）：**数据/代码/文档一律走魔搭**，
不用 SFTP 直传。SFTP 只保留给"查状态/启停进程"这类控制命令。

用法：
  # 本地 -> 魔搭（把文件推到中转仓）
  python s85_ms_transfer.py up --files data/ldm_mvtd/train.jsonl s84_dual_task.py

  # 魔搭 -> 服务器（在服务器上执行的命令，由本脚本远程触发）
  python s85_ms_transfer.py pull --files data/ldm_mvtd/train.jsonl s84_dual_task.py

  # 服务器 -> 魔搭 -> 本地（取回结果）
  python s85_ms_transfer.py down --files results/MVTD_s17.json

  # 列出中转仓内容
  python s85_ms_transfer.py ls

Token：
  - 服务器：/root/.ms_token（已存在，chmod 600）
  - 本地：优先环境变量 MS_TOKEN，其次 ~/.ms_token
  - 若本地没有 token，up/down 会提示改用 `pull`（由服务器侧 push 到魔搭后再 down）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
REPO = 'Yoneko/ldm-feasibility-testkit'
PREFIX = 'transfer'
REMOTE = '/root/ldm'
PY = '/root/miniconda3/bin/python'


def get_token() -> str | None:
    t = os.environ.get('MS_TOKEN', '').strip()
    if t:
        return t
    for p in (Path.home() / '.ms_token', ROOT / '.ms_token'):
        if p.exists():
            return p.read_text(encoding='utf-8').strip()
    return None


def _api(token):
    from modelscope.hub.api import HubApi
    api = HubApi()
    api.login(token)
    return api


def cmd_up(a):
    token = get_token()
    if not token:
        print('[error] 本地没有魔搭 token（设 MS_TOKEN 或 ~/.ms_token）。\n'
              '        替代：在服务器执行 pull，再从服务器 push。')
        return 2
    api = _api(token)
    for f in a.files:
        p = Path(f)
        if not p.exists():
            print('skip missing', f)
            continue
        remote_path = f'{PREFIX}/{f}'.replace('\\', '/')
        api.upload_file(repo_id=REPO, path_or_fileobj=str(p),
                        path_in_repo=remote_path, repo_type='dataset')
        print('up', f, '->', remote_path)
    return 0


def cmd_pull(a):
    """在服务器上从魔搭拉取到 /root/ldm。"""
    import s76_remote as R
    files = a.files
    code = (
        f"import sys; sys.path.insert(0,'{REMOTE}')\n"
        "from pathlib import Path\n"
        "from modelscope.hub.api import HubApi\n"
        "import modelscope.hub.snapshot_download as _s\n"
        "from modelscope.hub.snapshot_download import snapshot_download\n"
        f"tok=Path('/root/.ms_token').read_text().strip()\n"
        "api=HubApi(); api.login(tok)\n"
        f"dst=snapshot_download('{REPO}', repo_type='dataset')\n"
        "print('SNAP', dst)\n"
    )
    c = R.conn()
    rc, out, err = R.sh(c, f'cd {REMOTE} && {PY} -c "{code}"', timeout=600)
    print(out[-3000:])
    if rc != 0:
        print('[stderr]', err[-2000:])
        c.close()
        return rc
    # 从快照目录复制到工作路径
    lines = [l for l in out.splitlines() if l.startswith('SNAP ')]
    snap = lines[-1].split('SNAP ', 1)[1].strip() if lines else ''
    cp = []
    for f in files:
        rel = f'{PREFIX}/{f}'.replace('\\', '/')
        cp.append(f"mkdir -p $(dirname {REMOTE}/{f}) && "
                  f"cp '{snap}/{rel}' {REMOTE}/{f} && echo copied {f}")
    rc2, out2, err2 = R.sh(c, ' ; '.join(cp), timeout=300)
    print(out2)
    c.close()
    return rc2


def cmd_down(a):
    token = get_token()
    if not token:
        print('[error] 本地没有魔搭 token。改用服务器 push 后再运行本命令。')
        return 2
    api = _api(token)
    from modelscope import snapshot_download
    snap = snapshot_download(REPO, repo_type='dataset',
                             token=token)
    for f in a.files:
        rel = f'{PREFIX}/{f}'.replace('\\', '/')
        src = Path(snap) / rel
        if not src.exists():
            print('missing in repo', rel)
            continue
        dst = ROOT / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        print('down', rel, '->', dst)
    return 0


def cmd_push_server(a):
    """让服务器把 /root/ldm 下指定文件推到魔搭。"""
    import s76_remote as R
    body = (
        "from pathlib import Path\n"
        "from modelscope.hub.api import HubApi\n"
        f"api=HubApi(); api.login(Path('/root/.ms_token').read_text().strip())\n"
        "files=" + repr(a.files) + "\n"
        f"for f in files:\n"
        f"    api.upload_file(repo_id='{REPO}', path_or_fileobj='{REMOTE}/'+f,\n"
        f"                    path_in_repo='{PREFIX}/'+f, repo_type='dataset')\n"
        "    print('pushed', f)\n"
    )
    c = R.conn()
    rc, out, err = R.sh(c, f'cd {REMOTE} && {PY} -c {_q(body)}', timeout=600)
    print(out)
    if rc != 0:
        print('[stderr]', err[-2000:])
    c.close()
    return rc


def _q(s: str) -> str:
    """安全地把 python 源码作为 -c 参数传过去。"""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def cmd_ls(a):
    token = get_token()
    if not token:
        print('[error] 需要本地 token')
        return 2
    api = _api(token)
    files = api.get_dataset_file_list(REPO, f'{PREFIX}/', recursive=True)
    for f in files:
        print(f.get('Path', f))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, fn in (('up', cmd_up), ('pull', cmd_pull), ('down', cmd_down),
                     ('push-server', cmd_push_server)):
        p = sub.add_parser(name)
        p.add_argument('--files', nargs='+', required=True)
        p.set_defaults(fn=fn)
    p = sub.add_parser('ls')
    p.set_defaults(fn=cmd_ls)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == '__main__':
    main()
