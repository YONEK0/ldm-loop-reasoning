#!/bin/bash
# 昇腾 910B 主矩阵：13 seed × 48k，warmup=1280，linear 读出。
# 判据与理由见 docs/plans/ASCEND_REAL_TRAINING_PLAN.md §4.2 阶段 2。
# 并发：12（16 vCPU 留 4 核给系统；显存约 650 MB/run）
set -u
cd "$(dirname "$0")" || exit 1
mkdir -p results/main logs

WARMUP=${WARMUP:-1280}       # 48k 的标准值；改总步数时按比例改这个
STEPS=${STEPS:-48000}
MAXPAR=${MAXPAR:-12}
SEEDS="71 17 53 13 19 73 11 29 61 67 79 83 89"
PY=python

running=0
for s in $SEEDS; do
  out="results/main/MAIN_w${WARMUP}_s${s}.json"
  if [ -e "$out" ]; then echo "skip (exists) $out"; continue; fi
  $PY s71_ldm_iter4.py \
    --task mvtd --data-dir data/ldm_mvtd2 --pos none \
    --ouro --step-sup --curriculum 2,3,4,5,6 --curriculum-step 8000 \
    --steps "$STEPS" --warmup "$WARMUP" --bs 64 --lr 1e-3 --r-max 8 \
    --eval-every 2000 --early-stop \
    --seed "$s" --out "$out" > "logs/MAIN_w${WARMUP}_s${s}.log" 2>&1 &
  echo $! > "results/main/MAIN_w${WARMUP}_s${s}.pid"
  echo "launched s=$s pid=$!"
  running=$((running+1))
  if [ "$running" -ge "$MAXPAR" ]; then wait; running=0; fi
done
wait
echo "all done"
