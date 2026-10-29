#!/bin/bash
# AMD 实例批量训练：并行度由 PAR 环境变量指定（默认 6），可选先校准。
cd /mnt/workspace/ldm/code
export PYTHONUNBUFFERED=1
LOG=/mnt/workspace/batch.log
exec >> $LOG 2>&1
echo "=== BATCH START $(date) ==="

PAR=${PAR:-6}
if [ "${CALIBRATE:-0}" = "1" ]; then
  echo "=== 1. 并行度校准 ==="
  python s92_parallel_calibration.py --slots 1,3,6,9 --steps 600 2>&1 | tee /mnt/workspace/cal.txt
  PAR=$(python - <<'PY'
import re
rows=[]
for line in open('/mnt/workspace/cal.txt', encoding='utf-8'):
    m = re.match(r'\s*(\d+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s*$', line)
    if m: rows.append((int(m.group(1)), float(m.group(2))))
best = rows[0][0] if rows else 6
for slots, eff in rows:
    if eff >= 0.7: best = slots
print(best)
PY
)
fi
echo "=== PAR = $PAR (nproc=$(nproc)) ==="

echo "=== 2. 生成 jobs ==="
mkdir -p /mnt/workspace/results/main /mnt/workspace/logs
cd /mnt/workspace
: > jobs.txt
DATA=/mnt/workspace/ldm/data/ldm_mvtd2
CODE=/mnt/workspace/ldm/code
for s in 71 17 53 13 19 73 11 29 61 67 79 83 89; do
  echo "cd $CODE && python s71_ldm_iter4.py --task mvtd --data-dir $DATA --pos none --ouro --step-sup --curriculum 2,3,4,5,6 --curriculum-step 8000 --steps 48000 --warmup 1280 --bs 64 --lr 1e-3 --r-max 8 --eval-every 4000 --early-stop --seed $s --out /mnt/workspace/results/main/MAIN_w1280_s${s}.json > /mnt/workspace/logs/MAIN_w1280_s${s}.log 2>&1" >> jobs.txt
done
for s in 17 53 13 71 89 19; do
  echo "cd $CODE && python s71_ldm_iter4.py --task mvtd --data-dir $DATA --pos none --ouro --step-sup --curriculum 2,3,4,5,6 --curriculum-step 8000 --steps 48000 --warmup 1280 --bs 64 --lr 1e-3 --r-max 8 --eval-every 4000 --early-stop --head-type mlp --seed $s --out /mnt/workspace/results/main/MLP_w1280_s${s}.json > /mnt/workspace/logs/MLP_w1280_s${s}.log 2>&1" >> jobs.txt
done
wc -l jobs.txt

echo "=== 3. 拉起队列 ==="
cd $CODE
nohup setsid bash run_queue.sh /mnt/workspace/jobs.txt --par $PAR --stagger 25 > /mnt/workspace/queue.log 2>&1 &
echo $! > /mnt/workspace/queue.pid
echo "QUEUE_PID=$(cat /mnt/workspace/queue.pid)"
sleep 40
echo "=== 4. 初始状态 ==="
pgrep -fc "s71_ldm_iter4" || true
ls /mnt/workspace/results/main/ 2>/dev/null | head -25
echo "=== BATCH_SETUP_DONE ==="
