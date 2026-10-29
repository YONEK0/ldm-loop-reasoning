#!/bin/bash
# run_queue.sh — CPU 感知的并行队列（无批屏障，槽位补位）
#
# 为什么不用「每 N 个一批、等整批结束」：
#   13 个 run / 6 路时，末批只剩 1 个 run，机器空转一半时间。
#   本脚本始终保持 N 个在跑，一个结束立刻补下一个。
#
# 用法：
#   bash run_queue.sh jobs.txt --par 6                 # 单卡，6 路
#   bash run_queue.sh jobs.txt --par 6 --gpus 0,1,2    # 多卡，轮转分配
#   bash run_queue.sh jobs.txt --par 6 --stagger 30    # 启动随机延迟 0~30s
#
# jobs.txt：每行一条**完整的训练命令**（不带 nohup、不带 &）。
#   脚本会自动补 --out 存在则跳过（靠命令里的 --out 路径判断）。
#
set -u
JOBS=${1:?用法: bash run_queue.sh jobs.txt --par 6 [--gpus 0,1] [--stagger 30]}
shift
PAR=6
GPUS=""
STAGGER=20
while [ $# -gt 0 ]; do
  case "$1" in
    --par) PAR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --stagger) STAGGER="$2"; shift 2 ;;
    *) echo "未知参数 $1"; exit 1 ;;
  esac
done

mkdir -p logs results
IFS=',' read -r -a DEVS <<< "$GPUS"
NDEV=${#DEVS[@]}
n=0
launched=0
skipped=0

running() { jobs -rp | wc -l | tr -d ' '; }

while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac

  # --out 已存在则跳过（可安全重跑）
  out=$(sed -n 's/.*--out[ =]\([^ ]*\).*/\1/p' <<< "$cmd" | tail -1)
  if [ -n "$out" ] && [ -e "$out" ]; then
    echo "skip (exists) $out"; skipped=$((skipped+1)); continue
  fi

  # 等空槽
  while [ "$(running)" -ge "$PAR" ]; do sleep 5; done

  # 多卡时轮转绑定
  if [ "$NDEV" -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES="${DEVS[$((n % NDEV))]}"
    n=$((n+1))
  fi

  # 错开相位：所有 run 的 eval 都发生在相同 step，同时 eval 会造成
  # CPU/GPU 同步尖峰。随机延迟让它们错开。
  if [ "$STAGGER" -gt 0 ]; then sleep $((RANDOM % STAGGER)); fi

  echo "[$(date +%H:%M:%S)] launch: $cmd"
  ( eval "$cmd" ) &
  launched=$((launched+1))
  sleep 2
done < "$JOBS"

wait
echo "=== 完成：启动 $launched 个，跳过 $skipped 个 ==="
