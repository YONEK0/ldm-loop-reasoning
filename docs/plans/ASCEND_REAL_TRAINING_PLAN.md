# 昇腾 910B 正式训练配置与监测方案（101 小时）

> 2026-10-28。面向"真实训练 + 持续监测防翻车"，不是探路批次。
> 前提：1×NPU 910B3，16 vCPU，32 GiB；配额 **101 小时（墙上时间）**；
> 只有 `/mnt/workspace` 与 `/home` 关机后保留。

---

## 一、实例配置（三项决策）

| 项 | 选择 | 理由 |
|---|---|---|
| 模板 | **`cann_9.0.0.py3.11-A2-arm-20261019`** | 不是选新，是选 **torch_npu 轮子覆盖**。9.1.0 的 torch_npu 通常滞后发布，装不上就得重建实例、白烧小时数。**开机后先 `pip list \| grep -i npu`——若镜像预装了 torch_npu，那预装的那个模板才是对的**（预装永远优于自装）|
| NPU 规格 | `1*NPU 910B3 16vCPU 32GiB` | 唯一选项。**16 vCPU 才是并行瓶颈**：数据生成是纯 Python 单线程（3080 上实测每进程 pcpu 恒 100%）。留 4 核给系统 → **12 路并行**；显存每个 run 约 650 MB，32 GiB 装得下 |
| 持久化 | 全部放 `/mnt/workspace` | 其他目录关机清零。脚本里硬编码了 `/root/ldm`，用 `ln -s /mnt/workspace/ldm /root/ldm` 兼容，**不要**去改脚本（改错了更难查）|

---

## 二、开机后前 30 分钟（必做，失败不要往下走）

```bash
# 1. 环境
npu-smi info
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"

# 2. 代码（GitHub 出网不通就改走魔搭 Yoneko/ldm-feasibility-testkit）
mkdir -p /mnt/workspace && cd /mnt/workspace
git clone https://github.com/YONEK0/ldm-loop-reasoning.git ldm
ln -s /mnt/workspace/ldm /root/ldm

# 3. 算子自检
cd /mnt/workspace/ldm/code
python -c "
import torch, torch_npu
d=torch.device('npu')
q=torch.randn(64,4,32,device=d)
try: o=torch.nn.functional.scaled_dot_product_attention(q,q,q); print('SDPA OK')
except Exception as e: print('SDPA FAIL ->',e)
y=torch.zeros(320,192,device=d); x=torch.randn(64,192,device=d)
y.index_add_(0,torch.randint(0,320,(64,),device=d),x); print('index_add OK')
a=torch.randn(8,8,device=d); print('matmul OK',(a@a).shape)
"

# 4. 4k 步数值一致性（约 12 分钟）
python s71_ldm_iter4.py --task mvtd --data-dir ../data/ldm_mvtd2 --pos none \
  --ouro --step-sup --curriculum 2,3,4,5,6 --curriculum-step 8000 \
  --steps 4000 --warmup 960 --bs 64 --lr 1e-3 --r-max 8 \
  --eval-every 4000 --seed 53 --out ../results/ASC_consist_s53.json
# 判据：与 results/remote_P04_cs48k_s53.json 的 hist 前 4 个点比，
#     loss 容差 ±0.05、准确率 ±0.02。NPU 浮点顺序不同，不会逐位相同。
```

SDPA 失败 → 必须手写注意力 fallback（`d=192`，性能可接受）；不要跳过一致性验证。

---

## 三、监测体系（防翻车的核心）

### 3.1 已实现：`s91_campaign_monitor.py`

盯本项目**实际出现过**的四种翻车形态，阈值写死在源码里、跑前定稿：

| 形态 | 判据 | 历史实例 |
|---|---|---|
| `COLLAPSE` | 某跳曾 ≥0.85，现 <0.60 | cap 升档后 hop4 从 0.915 崩到 0.1 |
| `SEESAW` | 同轮最高−最低 ≥0.40 且最低 <0.60 | hop5=0.0 同时 hop6=1.0（H-007）|
| `STALL` | 连续 4 次 eval 的 hop6 峰值无提升且 hop7 <0.80 | 13 seed 里 5 个 task loss 卡在 0.21+（H-029）|
| `FLAT` | r1..r8 彼此相差 <0.02 **且** 都在 0.65 以下 | 评测漏传 `adj_bias_list` 导致恒 0.50（H-026）|
| `LATE_STALL` | 进度 ≥75% 时 task loss 仍 ≥0.15 | 同上，是 H-029 的直接签名 |

用法：

```bash
# 每 5 分钟刷新一次看板，有 ALERT 就以退出码 2 结束（接告警）
python s91_campaign_monitor.py --watch 300 --glob 'MAIN_*.json' \
       --glob 'MLP_*.json' --dead-check --alert-exit 2
# 看板：results/campaign_status.md（人读）+ campaign_status.json（机读）
```

`--dead-check` 只在监测**正在跑**的批次时开启；历史归档开启会全部误报。

### 3.2 训练侧必须同时开的三件事

```bash
python s71_ldm_iter4.py ... \
  --eval-every 2000 \        # 48k 共 24 次评估，早期即可发现异常（评估开销约 6 分钟/run）
  --early-stop \             # 课程覆盖的全部难度都 ≥0.95 即停，防止过冲遗忘
```
外加 `s77_watchdog.py --interval 300 --auto-stop` 做进程级监护（loss 爆炸/灾变时
自动回滚到最近 MILESTONE）。

### 3.3 回滚流程（翻车时照做）

每个 eval 都会写 `<tag>_ckpt.pt`（EMA 权重，评测口径）与 `<tag>_state.pt`
（含模型/优化器/调度器/RNG/step/evals_hist，续跑口径）；任一跳首次解到
≥0.95 时另存 `_MILESTONE_s<step>.pt` 永久档。

```bash
# 1. 找到该 run 最好的里程碑
ls -t results/MAIN_*.json | head
python - <<'PY'
import json, glob
d = json.load(open(glob.glob('results/MAIN_s53.json')[0]))
for r in d['evals_hist']:
    print(r['step'], (r.get('dev_r5_by_hop') or {}).get('6'),
          (r.get('confirm_r6_by_hop') or {}).get('7'))
PY
# 2. 从里程碑续跑（新 --out，不覆盖原档）
python s71_ldm_iter4.py ... --resume results/MAIN_s53_MILESTONE_s32000.pt \
  --out results/MAIN_s53_RE1.json
```

**注意**：续跑必须用**新的 `--out`**，原档保留作对照；续跑会重算 LR 调度
（`opt.param_groups[0]['lr'] = lr * min(1, start_step/warm)`），所以
`--warmup` 要与原 run 保持一致。

### 3.4 两个必须写进手册的教训

1. **评测路径必须与训练路径传相同的输入**。H-026 就是评测漏传
   `adj_bias_list`，读数恒 0.50，看起来像"模型没学会"，实际是 bug。
   任何新增训练期输入（掩码、hop_vec、邻接偏置）都要同步检查评测侧。
2. **杀进程只能用数字 PID**。`pkill -f` 对 setsid 脱离的进程不可靠，
   之前两次没杀掉。启动时把 PID 写进 `results/<tag>.pid`。

---

## 四、101 小时的编排

### 4.1 先明确一件事：时间不该花在"更长的训练"上

已有证据表明训练长度不是杠杆：

- P0-3：同 step 处 24k 与 48k **完全相同**，48k 的收益仅来自课程走到了 hop6；
- P19：s53 从 48k 的 0.900 训到 64k 反而降到 0.805（非单调）；
- H2：真正的杠杆是 **warmup 长度**，不是步数。

所以 101 小时应花在**宽度**（多 seed 买统计功效）和**规模上探**（审稿必问），
而不是把单个 run 拉到 200k 步。

### 4.2 排期

| 阶段 | 内容 | 并行 | 预算 |
|---|---|---|---|
| 0 迁移验证 | 环境自检 + 4k 一致性 | 1 | **3h** |
| 1 试跑闸门 | seed 53 @48k warmup=1280，确认 NPU 上能复现 h7≥0.80，同时验证监测器 | 1 | **4h** |
| 2 主矩阵 | **13 seed × 48k warmup=1280**（linear 读出）——论文主表 | 12 | **10h** |
| 3 决策模块 | mlp 读出 × 6 seed（s17/53/13/19/71/89）@48k | 6 | **8h** |
| 4 warmup 扫描 | s53 × {1920, 2560, 3840}，看是转折点还是概率平移 | 3 | **6h** |
| 5 规模上探 | d=384（参数 ×3.9）与 n_core=2，各 4 seed @48k | 8 | **20h** |
| 6 第二任务 | 新图族生成器 + 捷径审计 + 6 seed | 6 | **20h** |
| 7 论文图表 | r 扫描矩阵、训练曲线、参数-成功率散点 | — | **8h** |
| 缓冲 | 迁移排错、返工、复跑 | — | **22h** |

合计 101h。**阶段 2 只占 10 小时**——这就是为什么不必砍实验：矩阵很便宜。

### 4.3 小时账本（每次开机都记）

```bash
# 开机时
date +%s > /mnt/workspace/boot_ts
# 关机前
python -c "
import time; b=int(open('/mnt/workspace/boot_ts').read())
print('本次已用 %.1f h' % ((time.time()-b)/3600))"
```

把每次的用量抄进本文件末尾的账本表，剩余小时数随时可见——这是 101 小时
配额下最容易失控的地方。

| # | 开机 | 关机 | 用量 | 剩余 | 做了什么 |
|---|---|---|---|---|---|
| | | | | 101 | |

---

## 五、这一台上**不要**做的两件事

1. **不要跑 24/7 常驻迭代链**。`sleep 300` 等评估的时间照样计费。
   闭环应改为"开窗跑批"：开机 → 拉起 12 路 → 监看 → 推结果 → 关机。
2. **不要在未验证数值一致性前开主矩阵**。NPU 上若 SDPA 走了不同实现或
   出现非确定性，13 个 seed 会一起偏，白烧最贵的 10 小时。
