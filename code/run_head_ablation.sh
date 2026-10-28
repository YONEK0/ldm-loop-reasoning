#!/bin/bash
# H1 批次：决策模块形态消融，四臂同机同 seed（见 PREREG_HEAD_ABLATION_20261028.md）
cd "$(dirname "$0")" || exit 1
mkdir -p results/head_ablation logs
PY=python
for ht in linear bilinear mlp xattn; do
  nohup $PY s71_ldm_iter4.py \
    --task mvtd --data-dir data/ldm_mvtd2 --pos none \
    --ouro --step-sup --curriculum 2,3,4,5,6 --curriculum-step 8000 \
    --steps 48000 --bs 64 --lr 1e-3 --r-max 8 --eval-every 4000 \
    --seed 17 --head-type "$ht" \
    --out "results/head_ablation/H1_${ht}_s17.json" \
    > "logs/H1_${ht}_s17.log" 2>&1 &
  echo "launched $ht pid=$!"
done
wait
echo "all done"
