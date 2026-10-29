#!/bin/bash
# bootstrap_amd.sh — AMD(ROCm) 环境一键准备 + 并行度校准
#
# 在 ModelScope × AMD 工作区的终端里直接粘贴运行：
#   bash bootstrap_amd.sh            # 只准备 + 校准，不启动训练
#   bash bootstrap_amd.sh --launch   # 校准完直接拉起主矩阵队列
#
# 前提：网络能到 GitHub（代码）与魔搭（数据备份二选一即可）。
set -euo pipefail
WORK=${WORK:-/mnt/workspace}
REPO=https://github.com/YONEK0/ldm-loop-reasoning.git
MSDATASET=Yoneko/ldm-feasibility-testkit
LAUNCH=0
[ "${1:-}" = "--launch" ] && LAUNCH=1

echo "==== 1/5 环境自检 ===="
python - <<'PY'
import torch
print('torch     ', torch.__version__)
print('cuda/ROCm ', torch.cuda.is_available())
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print('卡数      ', n)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f'  [{i}] {p.name}  {p.total_memory/1e9:.0f} GB')
    q = torch.randn(64, 4, 32, device='cuda')
    o = torch.nn.functional.scaled_dot_product_attention(q, q, q)
    print('SDPA      OK', tuple(o.shape))
    y = torch.zeros(320, 192, device='cuda'); x = torch.randn(64, 192, device='cuda')
    y.index_add_(0, torch.randint(0, 320, (64,), device='cuda'), x)
    print('index_add OK')
PY
echo "CPU 核数: $(nproc)"

echo "==== 2/5 取代码 ===="
mkdir -p "$WORK" && cd "$WORK"
if [ -d ldm/.git ]; then
  cd ldm && git pull --ff-only && cd ..
else
  git clone --depth 1 "$REPO" ldm
fi
cd "$WORK/ldm"
# 脚本里硬编码了 /root/ldm，用软链兼容（不要改脚本）
if [ ! -e /root/ldm ]; then ln -s "$WORK/ldm" /root/ldm 2>/dev/null || true; fi

echo "==== 3/5 取数据 ===="
if [ -f data/ldm_mvtd2/train.jsonl ] && [ -s data/ldm_mvtd2/train.jsonl ]; then
  echo "数据已在仓库内（GitHub 版自带 data/ldm_mvtd2）"
else
  echo "仓库无数据，改从魔搭拉取"
  python - <<'PY'
from modelscope import snapshot_download
import shutil, pathlib
p = pathlib.Path(snapshot_download('Yoneko/ldm-feasibility-testkit', repo_type='dataset'))
src = p / 'transfer' / 'data' / 'ldm_mvtd2'
dst = pathlib.Path('data/ldm_mvtd2'); dst.mkdir(parents=True, exist_ok=True)
for f in src.glob('*'):
    shutil.copy(f, dst / f.name)
print('copied', [f.name for f in dst.glob('*')])
PY
fi
wc -l data/ldm_mvtd2/*.jsonl

echo "==== 4/5 冒烟（200 步）===="
python s71_ldm_iter4.py --task mvtd --data-dir data/ldm_mvtd2 --pos none \
  --ouro --step-sup --curriculum 2 --steps 200 --bs 64 --lr 1e-3 --r-max 8 \
  --eval-every 200 --seed 17 --out results/AMD_smoke_s17.json 2>&1 | tail -2

echo "==== 5/5 并行度校准（1/3/6 路，各 600 步）===="
PAR_REC=6
python s92_parallel_calibration.py --slots 1,3,6 --steps 600 2>&1 | tee /tmp/cal.txt
# 依据相对效率给出建议：<0.7 的那一档的前一档
python - <<'PY'
import re
rows = []
for line in open('/tmp/cal.txt', encoding='utf-8'):
    m = re.match(r'\s*(\d+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s*$', line)
    if m:
        rows.append((int(m.group(1)), float(m.group(2))))
best = rows[0][0] if rows else 6
for slots, eff in rows:
    if eff >= 0.7:
        best = slots
print(f'\n建议 --par {best}（效率 >=0.7 的最大档位）')
open('/tmp/best_par', 'w').write(str(best))
PY

if [ "$LAUNCH" = "1" ]; then
  PAR=$(cat /tmp/best_par 2>/dev/null || echo 6)
  echo "==== 生成主矩阵 jobs.txt（13 seed × 48k，warmup=1280）===="
  mkdir -p results/main
  : > jobs.txt
  for s in 71 17 53 13 19 73 11 29 61 67 79 83 89; do
    echo "python s71_ldm_iter4.py --task mvtd --data-dir data/ldm_mvtd2 --pos none --ouro --step-sup --curriculum 2,3,4,5,6 --curriculum-step 8000 --steps 48000 --warmup 1280 --bs 64 --lr 1e-3 --r-max 8 --eval-every 2000 --early-stop --seed $s --out results/main/MAIN_w1280_s${s}.json" >> jobs.txt
  done
  wc -l jobs.txt
  echo "==== 拉起队列（--par $PAR）===="
  nohup bash run_queue.sh jobs.txt --par "$PAR" --stagger 30 > logs/queue.out 2>&1 &
  echo "队列已在后台运行，PID $!"
  echo "看进度：python s86_progress.py --pattern 'MAIN_*'"
  echo "看翻车：python s91_campaign_monitor.py --watch 300 --glob 'MAIN_*.json' --dead-check"
fi

echo "==== 完成 ===="
