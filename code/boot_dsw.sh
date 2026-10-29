#!/bin/bash
# boot_dsw.sh — DSW 实例开机自恢复脚本。
#
# 用途：DSW 实例有「单次部署最大时长」，到期会被回收；重启后需要
# 无人值守地把训练接上。本脚本在 NAS 持久盘（/mnt/workspace）上，
# 重启后手动执行一次即可恢复全部工作。
#
# 恢复内容：
#   1. ModelScope token（从 NAS 上的 ms_token.txt 复制回 /root/.ms_token）
#   2. 代码更新（git pull；若不可达则用 NAS 上已有的副本）
#   3. 未完成的 run → 从 MILESTONE 续跑（新 --out，原档保留）
#      无 MILESTONE 的 → 删掉残档让队列重跑（它本来就没产出有效结果）
#   4. 拉起训练队列（run_queue.sh，槽位补位式）
#   5. 拉起守护进程（dsw_guardian.py，保守型）
#
# 用法：
#   bash /mnt/workspace/boot_dsw.sh              # 恢复全部
#   bash /mnt/workspace/boot_dsw.sh --no-train   # 只恢复守护进程，不开训练
#   bash /mnt/workspace/boot_dsw.sh --dry-run    # 只打印将要做什么
set -u

WS=/mnt/workspace
CODE=$WS/ldm/code
GUARD=$WS/results/guardian
RESULTS=$WS/results/main
PY=python
DRY=0
TRAIN=1
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --no-train) TRAIN=0 ;;
  esac
done

echo "==== boot_dsw $(date '+%F %T') ===="
mkdir -p "$GUARD" "$RESULTS" "$WS/logs"

# ---- 1. token（NAS 上的副本 → /root/.ms_token） ----
if [ -f "$WS/ms_token.txt" ]; then
  cp "$WS/ms_token.txt" /root/.ms_token
  chmod 600 /root/.ms_token
  echo "[1] token restored ($(wc -c < /root/.ms_token) bytes)"
else
  echo "[1] !! 缺 $WS/ms_token.txt，ModelScope 相关操作会失败"
fi

# ---- 2. 代码 ----
if [ -d "$WS/ldm/.git" ]; then
  ( cd "$WS/ldm" && git pull --ff-only 2>&1 | tail -2 ) || \
    echo "[2] git pull 失败，用 NAS 上的现有副本"
else
  echo "[2] !! 缺 $WS/ldm（git 仓库），无法恢复代码"
fi
[ -f "$WS/dsw_guardian.py" ] && cp "$WS/dsw_guardian.py" "$CODE/" 2>/dev/null
[ -f "$WS/dsw_guardian.py" ] && cp "$WS/dsw_guardian.py" "$WS/ldm/code/" 2>/dev/null
echo "[2] code ok"

# ---- 3. 处理未完成的 run ----
echo "[3] 处理未完成的 run："
cd "$CODE" || exit 1
$PY - <<'PYEOF'
import glob, json, os, re, subprocess, sys
RESULTS = '/mnt/workspace/results/main'
JOBS_TXT = '/mnt/workspace/jobs.txt'

def cmd_for(name):
    """从 jobs.txt 找回命令行。"""
    if not os.path.exists(JOBS_TXT):
        return None
    for line in open(JOBS_TXT, encoding='utf-8'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        m = re.search(r'--out\s+(\S+\.json)', line)
        if m and os.path.basename(m.group(1)) == name + '.json':
            return line
    return None

def is_done(d):
    a = d.get('args') or {}
    tgt = a.get('steps') or 0
    ev = d.get('evals_hist') or []
    last = ev[-1].get('step') if ev else 0
    return bool(d.get('early_stop')) or (tgt and last >= tgt)

todo = []
for f in sorted(glob.glob(RESULTS + '/*.json')):
    name = os.path.basename(f)[:-5]
    if '_RESUMED' in name:
        continue
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if is_done(d):
        continue
    ms = sorted(glob.glob(RESULTS + '/' + name + '_MILESTONE_*.pt'))
    cmd = cmd_for(name)
    if not cmd:
        print('   ', name, '无命令行 → 跳过')
        continue
    if ms:
        mstep = int(re.search(r'_MILESTONE_s(\d+)\.pt$', ms[-1]).group(1))
        new_out = os.path.join(RESULTS, f'{name}_R2_s{mstep}.json')
        if os.path.exists(new_out):
            print('   ', name, '已续跑过 → 跳过')
            continue
        newcmd = re.sub(r'--out\s+\S+', f'--out {new_out}', cmd)
        newcmd = re.sub(r'--resume\s+\S+', '', newcmd)
        newcmd += f' --resume {ms[-1]}'
        todo.append((name, newcmd, new_out, ms[-1]))
        print('   ', name, f'从 MILESTONE s{mstep} 续跑 → {os.path.basename(new_out)}')
    else:
        # 无里程碑 = 没有任何可复用进度；残档只是半成品，删掉让队列重跑
        print('   ', name, '无 MILESTONE → 删残档，交由队列重跑')
        os.rename(f, f + '.incomplete')

with open('/mnt/workspace/resume_jobs.sh', 'w', encoding='utf-8') as f:
    f.write('#!/bin/bash\n')
    for name, cmd, out, ms in todo:
        f.write(cmd + '\n')
print('   待续跑', len(todo), '个')
PYEOF

# ---- 4. 拉起队列 ----
if [ "$TRAIN" = "1" ] && [ "$DRY" = "0" ]; then
  echo "[4] 拉起训练队列"
  # 队列跳过已存在的 --out；把续跑项单独跑，避免与队列抢同一份 jobs.txt
  if [ -s "$WS/resume_jobs.sh" ]; then
    ( cd "$CODE" && nohup setsid bash "$WS/resume_jobs.sh" > "$WS/logs/resume.log" 2>&1 & )
    sleep 3
  fi
  # 主队列：只跑还没跑过的（run_queue.sh 会跳过已存在的 --out）
  if [ -f "$WS/jobs.txt" ]; then
    ( cd "$CODE" && nohup setsid bash run_queue.sh "$WS/jobs.txt" --par 6 --stagger 25 \
        > "$WS/queue.log" 2>&1 & echo $! > "$WS/queue.pid" )
    echo "    queue pid $(cat "$WS/queue.pid" 2>/dev/null)"
  fi
else
  echo "[4] 跳过队列启动（dry-run 或 --no-train）"
fi

# ---- 5. 拉起守护进程 ----
if [ "$DRY" = "0" ]; then
  echo "[5] 拉起守护进程"
  if [ -f "$WS/guardian.pid" ]; then
    kill -9 "$(cat "$WS/guardian.pid")" 2>/dev/null
  fi
  nohup setsid python "$WS/dsw_guardian.py" \
      --results "$RESULTS" --guard "$GUARD" --interval 60 \
      > "$WS/guardian.log" 2>&1 &
  echo $! > "$WS/guardian.pid"
  echo "    guardian pid $(cat "$WS/guardian.pid")"

  # 进度轮询器（把状态写到 progress.txt，供外部读回）
  if [ -f "$WS/poll.sh" ]; then
    if [ -f "$WS/poll.pid" ]; then kill -9 "$(cat "$WS/poll.pid")" 2>/dev/null; fi
    nohup setsid bash "$WS/poll.sh" >/dev/null 2>&1 &
    echo $! > "$WS/poll.pid"
    echo "    poll pid $(cat "$WS/poll.pid")"
  fi
fi

echo "==== boot done ===="
echo "看进度：cat $WS/progress.txt"
echo "看事件：tail -20 $GUARD/events.jsonl"
