#!/bin/bash
# H2 批次：warmup 长度单变量实验。
# 起因（P19 意外发现）：同 seed、同配置，48k 基线 s53 h6=0.705/h7=0.695；
# 而 --steps 64000 的 run 在 48000 步处 h6=0.900/h7=0.900。唯一配置差异是
# warmup = steps//50 → 960 vs 1280。故"步数效应"与"warmup 效应"此前被绑死。
# 本批固定 steps=48000，只改 --warmup，把两者彻底分离。
cd "$(dirname "$0")" || exit 1
mkdir -p results/warmup logs
PY=python
for seed in 53 17; do
  for w in 480 960 1280 1920; do
    nohup $PY s71_ldm_iter4.py \
      --task mvtd --data-dir data/ldm_mvtd2 --pos none \
      --ouro --step-sup --curriculum 2,3,4,5,6 --curriculum-step 8000 \
      --steps 48000 --warmup $w --bs 64 --lr 1e-3 --r-max 8 \
      --eval-every 4000 --seed $seed \
      --out "results/warmup/H2_w${w}_s${seed}.json" \
      > "logs/H2_w${w}_s${seed}.log" 2>&1 &
    echo "launched w=$w seed=$seed pid=$!"
  done
done
wait
echo "all done"
