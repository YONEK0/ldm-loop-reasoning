# 复现指南

按顺序做，全部可在单卡（≥6 GB 显存）上完成。总耗时约 3 小时。

## 0. 环境

```bash
pip install torch        # 仅此一个依赖；无自定义 CUDA/NPU 算子
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

昇腾 910B 上 `torch.npu.is_available()` 为真时设备自动切换，代码无需修改。

## 1. 生成任务数据

```bash
cd code
python s84_dual_task.py --out ../data/ldm_mvtd2
```

产出 `train.jsonl` / `dev_random.jsonl` / `dev_ood.jsonl` / `manifest.json`。

**先跑捷径审计**（这一步不能跳过，它决定了结论是否可信）：

```bash
python - <<'PY'
import json
for split in ('train', 'dev_random', 'dev_ood'):
    rows = [json.loads(l) for l in open(f'../data/ldm_mvtd2/{split}.jsonl')]
    hit = tot = ties = 0
    for ex in rows:
        d = ex.get('divergence')
        if not d:
            continue
        a, b = d['a_tail'], d['b_tail']
        if a == b:                 # 两候选下游等长，该启发式无信息
            ties += 1
            continue
        pick = 0 if a > b else 1    # 启发式：选下游链更长的候选
        tot += 1
        hit += int(pick == ex['answer_index'])
    print(f'{split:11s} n={len(rows):5d} pick_longer={hit/tot:.4f} '
          f'(可比样本 {tot}) ties={ties} ({ties/len(rows)*100:.0f}%)')
PY
```

MVT-D v2 上的实际输出：`train pick_longer=0.4959`、`dev_random=0.4312`、
`dev_ood=0.4431`，且 30–36% 的样本两候选下游等长。**若任何朴素启发式接近 1.0，
说明任务有捷径，不要用它下任何能力结论**——本项目正是因此在 MVT-5 上作废了
整批结论（"选更长的候选"在原任务上 19750/19750 = 1.0000，
见 `docs/errors/err-task-shortcut.md`）。

对**已训练的 checkpoint** 还可用 `s83_shortcut_probe.py` 做三组对照
（orig / inverted / tied），判据写在脚本头部的预注册注释里。

## 2. 冒烟（200 步，约 20 秒）

```bash
python s71_ldm_iter4.py --task mvtd --data-dir ../data/ldm_mvtd2 --pos none \
  --ouro --step-sup --curriculum 2 --curriculum-step 8000 \
  --steps 200 --bs 64 --lr 1e-3 --r-max 8 --eval-every 200 \
  --seed 17 --head-type linear --out ../results/SMOKE_s17.json
```

预期：dev_r* ≈ 0.48–0.52（随机水平），`{"done": ...}` 正常退出。

## 3. 标准 run（48k 步，单卡约 70 分钟）

```bash
python s71_ldm_iter4.py --task mvtd --data-dir ../data/ldm_mvtd2 --pos none \
  --ouro --step-sup --curriculum 2,3,4,5,6 --curriculum-step 8000 \
  --steps 48000 --bs 64 --lr 1e-3 --r-max 8 --eval-every 4000 \
  --seed 71 --out ../results/REPRO_s71.json
```

## 4. 判读

```bash
python - <<'PY'
import json
d = json.load(open('../results/REPRO_s71.json'))
e = d['evals']
print('r=1 (单轮):      ', e['dev_r1'],   '  <- 应为随机水平')
print('r=2:            ', e['dev_r2'])
print('r=8:            ', e['dev_r8'])
print('by hop (r=5):   ', e['dev_r5_by_hop'], '  <- hop6 是关键')
print('外推 (r=6):     ', e['confirm_r6_by_hop'], '  <- hop7/hop8 未训练')
print('train task loss:', d['hist'][-1]['task'], '  <- 判断收敛 vs 停滞')
PY
```

**判读顺序**（重要，见 §5.5 的结论）：

1. `dev_r1` 必须接近 0.50。若显著高于 0.5，任务可能有单步可解的捷径。
2. `hist[-1]['task']` < 0.15 才算训练分布拟合上；≥ 0.21 属于**优化停滞**，
   此时 hop6/hop7 低不是外推问题。
3. `dev_r5_by_hop['6']` 是瓶颈所在；`confirm_r6_by_hop['7']` 是未训练深度的外推。

## 5. 多 seed（必须）

**单 seed 的绝对分数不能作为任何结论的依据。** 同配置重跑同一 seed，
hop7 的波动可达 ±0.10，个别 seed 达 +0.20（`docs/notes/P19_LONG64K_VERDICT_20261028.md`
记录了同 seed 同配置两次运行相差 0.205 的实例）。任何配方对比至少 3 个 seed，
并报告均值与跨度。

## 6. 决策模块消融（四臂）

```bash
bash run_head_ablation.sh        # linear / bilinear / mlp / xattn，48k 步，约 2.5 小时
```

判据已在 `docs/plans/PREREG_HEAD_ABLATION_20261028.md` 跑前写定：
任一非对照臂 hop6 ≥ 0.85 且 hop7 ≥ 0.80 → 决策模块承重；
四臂 hop6 全在对照 ±0.05 内 → 轻量决策模块足够。

## 7. warmup 单变量

```bash
bash run_warmup_ablation.sh      # seed 53，steps 固定 48000，warmup 960/1280
```

用于分离"训练步数"与"warmup 长度"——原实现 `warm = steps // 50` 把两者绑死，
任何 48k-vs-64k 对比都同时改变了 warmup。
