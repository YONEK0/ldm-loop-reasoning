#!/bin/bash
# boot_dsw_auto.sh — 重启后的一键自恢复（不依赖浏览器/网络）。
# 用法（在实例终端里执行，或通过 DSW 的「启动命令」配置）：
#     bash /mnt/workspace/boot_dsw_auto.sh
#
# 它做四件事：
#   1. 恢复 token 与代码
#   2. 未完成的 run 从 MILESTONE 续跑（新的 --out，原档保留）
#   3. 拉起训练队列（槽位补位式，PAR=6）
#   4. 拉起守护进程（保守型监测）+ 进度轮询器
#
# 幂等：重复执行安全（已续跑的会跳过；已在跑的不会被重复启动）。
set -u
WS=/mnt/workspace
LOG=$WS/boot.log

exec >> $LOG 2>&1
echo "==== boot_dsw_auto $(date '+%F %T') pid=$$ ===="

# 0) 幂等：已在跑就别重复起
if [ -f "$WS/boot.lock" ] && kill -0 "$(cat "$WS/boot.lock" 2>/dev/null)" 2>/dev/null; then
  echo "已有 boot 进程在跑（$(cat "$WS/boot.lock")），退出"
  exit 0
fi
echo $$ > "$WS/boot.lock"

# 1) token
if [ -f "$WS/ms_token.txt" ]; then
  cp "$WS/ms_token.txt" /root/.ms_token && chmod 600 /root/.ms_token
  echo "[1] token ok"
fi

# 2) 代码（git pull 失败不致命，NAS 上已有副本）
if [ -d "$WS/ldm/.git" ]; then
  ( cd "$WS/ldm" && timeout 60 git pull --ff-only 2>&1 | tail -2 ) || echo "[2] git pull 失败，用现有副本"
fi
[ -f "$WS/dsw_guardian.py" ] && cp -f "$WS/dsw_guardian.py" "$WS/ldm/code/" 2>/dev/null
echo "[2] code ok"

# 3) 续跑未完成的 run + 准备队列
cd "$WS/ldm/code" || { echo "[3] 缺代码目录，退出"; exit 1; }
python - <<'PYEOF'
import glob, json, os, re
RESULTS = '/mnt/workspace/results/main'
JOBS = '/mnt/workspace/jobs.txt'

def cmd_for(name):
    if not os.path.exists(JOBS):
        return None
    for line in open(JOBS, encoding='utf-8'):
        m = re.search(r'--out\s+(\S+\.json)', line.strip())
        if m and os.path.basename(m.group(1)) == name + '.json':
            return line.strip()
    return None

def done(d):
    a = d.get('args') or {}
    tgt = a.get('steps') or 0
    ev = d.get('evals_hist') or []
    last = ev[-1].get('step') if ev else 0
    return bool(d.get('early_stop')) or bool(tgt and last >= tgt)

todo = []
for f in sorted(glob.glob(RESULTS + '/*.json')):
    name = os.path.basename(f)[:-5]
    if '_RESUMED' in name or '_R2_' in name:
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if done(d):
        continue
    cmd = cmd_for(name)
    if not cmd:
        continue
    ms = sorted(glob.glob(RESULTS + '/' + name + '_MILESTONE_*.pt'))
    if ms:
        st = int(re.search(r'_MILESTONE_s(\d+)\.pt$', ms[-1]).group(1))
        out = os.path.join(RESULTS, f'{name}_R2_s{st}.json')
        if os.path.exists(out):
            continue
        c = re.sub(r'--out\s+\S+', f'--out {out}', cmd)
        c = re.sub(r'--resume\s+\S+', '', c) + f' --resume {ms[-1]}'
        todo.append(c)
        print('  resume', name, 'from s' + str(st))
    else:
        os.rename(f, f + '.incomplete')
        print('  restart', name, '(no milestone)')
with open('/mnt/workspace/resume_jobs.sh', 'w', encoding='utf-8') as f:
    f.write('#!/bin/bash\n')
    for c in todo:
        f.write(c + '\n')
print('  resume jobs:', len(todo))
PYEOF

# 4) 起训练
if [ -s "$WS/resume_jobs.sh" ]; then
  nohup setsid bash "$WS/resume_jobs.sh" > "$WS/logs/resume.log" 2>&1 &
  sleep 2
fi
# 队列里跳过已完成的（run_queue.sh 见到已存在的 --out 会跳过）
if [ -f "$WS/jobs.txt" ]; then
  nohup setsid bash run_queue.sh "$WS/jobs.txt" --par 6 --stagger 25 > "$WS/queue.log" 2>&1 &
  echo $! > "$WS/queue.pid"
  echo "[3] queue pid $(cat "$WS/queue.pid")"
fi

# 5) 守护进程 + 进度轮询
if [ -f "$WS/guardian.pid" ]; then kill -9 "$(cat "$WS/guardian.pid")" 2>/dev/null; fi
nohup setsid python "$WS/dsw_guardian.py" --results "$WS/results/main" \
    --guard "$WS/results/guardian" --interval 60 > "$WS/guardian.log" 2>&1 &
echo $! > "$WS/guardian.pid"
echo "[4] guardian pid $(cat "$WS/guardian.pid")"

if [ -f "$WS/poll.sh" ]; then
  if [ -f "$WS/poll.pid" ]; then kill -9 "$(cat "$WS/poll.pid")" 2>/dev/null; fi
  nohup setsid bash "$WS/poll.sh" >/dev/null 2>&1 &
  echo $! > "$WS/poll.pid"
  echo "[5] poll pid $(cat "$WS/poll.pid")"
fi

sleep 20
echo "==== boot done, 进程数 $(pgrep -fc 's71_ldm_iter4' || echo 0) ===="
tail -5 "$WS/guardian.log" 2>/dev/null
echo "==== BOOT_AUTO_OK ===="
