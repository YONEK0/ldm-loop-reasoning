# Looped Decision Model (LDM) — 循环深度是否承载串行推理？

一个 1.47M 参数的单层共享循环模型，在没有捷径的多跳图任务上，用**一次性决策**
（而非逐 token 生成）完成需要多步串行追踪的推理。本仓库包含完整代码、任务生成器、
预注册判据、全部实验记录与已归档结论。

**一句话结论**：循环确实被用于计算（单轮恒为随机），但能力在 r=2–3 饱和；
深度外推由训练覆盖深度决定；决定成败的不是外推手段，而是**训练分布能否拟合上**。

---

## 1. 模型

```
Prelude(1 层) → [ Core(1 层, 权重共享) × R ] → Coda(1 层) → 候选打分
```

| 项 | 值 |
|---|---|
| 总参数 | **1,474,307** |
| d / heads / ffn | 192 / 4 / 768 |
| 位置编码 | **无**（NoPE，H-002 证实是必要条件） |
| 输入编码 | 融合编码（每条事实 = subj + rel + obj 三个 embedding 之和，占一个位置） |
| 词表 | 320 |
| 读出 | `gather_pos(h, cand_pos) → Linear(192,1) → softmax`，**193 参数（0.013%）** |
| 轮次 | `--r-max 8`；Ouro 损失下实测熵 ≈ ln 8、`e_halt ≈ 4.5` |

参数分解：`tok_emb` 61,440 · `prelude` 444,864 · `core`(共享) 444,864 ·
`coda` 444,864 · `round_emb` 3,072 · `exit_mlp` 12,673 · `step_head` 61,760 ·
`score` 193 · `proc_head` 193。

按"单趟路径 vs 循环块"切分：**单趟 1,029,059（69.8%）、循环块 445,248（30.2%）**。

## 2. 任务（MVT-D v2）

两条等长链从 `e0` 引出，只有一条在第 `hop` 步可达；**候选结构完全对称**
（都有入边、都有出边、下游长度随机）。已消除三类局部捷径：

| 捷径 | 在原任务上的准确率 | 状态 |
|---|---|---|
| "选下游更长的候选" | **19750/19750 = 1.0000** | 已证实并弃用原任务（H-023） |
| 干扰候选无入边 | 1000/1000 = 1.0000 | v2 已修 |
| 下游链长差 1 | — | v2 已随机化 |

审计指标（在 train 上，"选下游更长候选"规则含等长计失败的准确率）：
`pick_longer = 0.3305`；仅在非等长样本上为 0.4959；30–36% 的样本两候选下游
**等长**（该启发式无信息）。标签均衡 0.49。生成器：`code/s84_dual_task.py`。
复核命令见 `REPRODUCE.md` §1。

> **为什么强调这个**：一条不看输入的启发式规则在全部样本上 100% 正确，
> 会让模型"全解"而实际什么都不会。去捷径后，此前"完全学会"的模型
> hop4/hop6 从 1.000/0.995 跌到 0.607/0.520。
> 完整调查见 `docs/errors/err-task-shortcut.md`（E-015）。

## 3. 快速开始

```bash
pip install torch            # 仅依赖 torch，无自定义算子
cd code

# 生成任务数据（MVT-D v2，无捷径）
python s84_dual_task.py --out ../data/ldm_mvtd2

# 训练一个 run（标准配方，48k 步）
python s71_ldm_iter4.py --task mvtd --data-dir ../data/ldm_mvtd2 --pos none \
  --ouro --step-sup --curriculum 2,3,4,5,6 --curriculum-step 8000 \
  --steps 48000 --bs 64 --lr 1e-3 --r-max 8 --eval-every 4000 \
  --seed 71 --out ../results/MY_RUN_s71.json

# 只跑 200 步冒烟
python s71_ldm_iter4.py ... --steps 200 --eval-every 200
```

设备自动选择：**NPU（昇腾 910B）→ CUDA → CPU**，无需改代码。
`results/*.json` 内含完整 `evals`（各轮 r=1..8 的 by-hop 准确率）、
`evals_hist`（训练全程曲线）、`hist`（每千步 loss / 熵 / e_halt）。

## 4. 目录

```
code/
  s63_ldm_model.py      模型 + 编码（head_type 消融开关）
  s66_ldm_mvt_gen.py    MVT 任务生成（原任务）
  s84_dual_task.py      MVT-D v2 生成器（无捷径，当前使用）
  s71_ldm_iter4.py      训练主脚本（--ouro / --step-sup / --head-type / --warmup）
  s65_ldm_gpu_smoke.py  评测函数
  s81_eval_rounds.py    逐轮准确率评测
  s88_adj_task.py       邻接掩码构造（已结案，保留备查）
  s89_auto_loop.py      迭代链编排器（scan→evaluate→decide→act→record）
  s85_ms_transfer.py    经 ModelScope 的数据中转
  s86_progress.py       多 run 进度一览
  run_head_ablation.sh  决策模块四臂消融启动脚本
  run_warmup_ablation.sh warmup 单变量实验启动脚本
docs/
  hypotheses/           假说三态（PENDING / CONFIRMED / REFUTED）+ INDEX
  notes/                判读报告与证据汇编
  plans/                预注册判据（跑前写定，不可事后改）
  errors/               事故与设计缺陷记录
results/                298 个 run 的 JSON（不含 checkpoint）
```

## 5. 核心结论（都有数据支撑）

### 5.1 循环被真实使用
三个 seed 的 **r=1（单轮）全部落在随机水平**（0.494–0.508）。若任务可用单步
注意力解决，至少某些 seed 会在 r=1 取得部分正确率。单轮在物理上解不了这个任务。

### 5.2 收益在 r≈2 饱和
r=2 → 0.87，r=3–8 → 0.88–0.90（增益 < 0.03）。Ouro 损失下熵稳定在 ln 8、
`e_halt ≈ 4.5`，即模型**仍在使用 8 轮**，但更多轮不带来准确率——
说明饱和来自计算本身，不是调度问题。文献：arXiv:2610.26030（"two fully
trained loops as the best looped setting"）、arXiv:2604.21106（φ = 0.46）。

→ 论文口径是**压缩式推理**（一轮推进多跳），**不是**"每轮一跳"。

### 5.3 稠密模型在同类任务上学不动
dense `mid`=1/3/5（1.40M / 2.29M / 3.18M），loss **精确停在 0.6931 = ln 2**，
参数 ×2.3 无变化；手动训 2000 步 train acc 仅 0.528（连训练集都学不会）。
loop 1.47M 同条件达 r8 = 0.90+。文献：arXiv:2402.12875（TC⁰ / CoT 串行表达能力）。

### 5.4 深度外推只靠训练覆盖
四种干预（dyn-r / warm-noise / sup-cap / r→16）全败。同 step 处 24k 与 48k
完全相同，48k 的收益仅来自"课程走到了 hop6"。

### 5.5 成败由"训练分布能否拟合"决定（最新，最重要）
13 个 48k run 的最终 train task loss 与 hop7 单调相关：

- task loss < 0.15 → **8/8 个 seed 的 hop7 ≥ 0.80**
- task loss ≥ 0.21 → **5/5 个 seed 的 hop7 < 0.80**

拟合到 0.0006–0.0031 的三个 seed，hop7 也只有 0.835 / 0.825 / 0.805。
**把训练集背下来不会带来更强的 hop7 外推**，上界约 0.80–0.84。

详见 `docs/notes/P19_LONG64K_VERDICT_20261028.md`。

## 6. 预注册纪律

所有判据在实验启动**前**写入 `docs/plans/`，跑完不改口径；负结果照实记录。
例如 `docs/plans/PREREG_HEAD_ABLATION_20261028.md` 在四臂训练启动前写定了
0.85 / 0.05 两个阈值与"防跷跷板"附加判据。

假说采用三态命名（`H-xxx-PENDING-…` / `-CONFIRMED-…` / `-REFUTED-…`），
状态变更必须重命名文件并更新 `docs/hypotheses/INDEX.md`。证伪的假说保留，
因为它们排除了搜索空间。

## 7. 已知失效路径（别再试）

| 方向 | 结论 |
|---|---|
| 邻接掩码检验"每轮一跳" | 三变体全败（H-025）。根因：节点=事实、候选=实体，不在同一张图；感受野被压过小直接降低可分性 |
| 四种外推干预 | 全败（H-014） |
| 深度插值 {2,4,6,8} → {3,5,7} | 证伪（H-022），s17 训练 hop6=0.995 而夹中间的 hop5=0.205 |
| 每轮层数 L=1/2/4 | 完全等价（H-021） |
| 加宽 d×2（参数 ×3.9） | 零收益（H-008） |
| EMA / 降 lr / 课程硬切换 / 课程回采 | 全无效（H-004/005/006/010/012） |
| xattn 决策头 | 完全学不动，loss 不降（H-026 进行中） |

## 8. 昇腾 910B 迁移

代码已设备无关（`_pick_device()` / `_accel()` / RNG state 跨设备）。
迁移验证标准见 `docs/plans/ASCEND_910B_MIGRATION_CHECKLIST.md`：
用同一配置跑 4k 步，loss 容差 ±0.05、准确率 ±0.02，逐点对照已有 JSON 的
`hist`（48 个 loss 点）与 `evals_hist`。

## 9. 致谢与文献

主要参照：arXiv:2510.25741（Ouro 损失）、2610.26030（looped scaling law）、
2604.21106（iso-depth scaling，φ=0.46）、2402.12875（CoT 与 TC⁰）、
2603.21676（depth-recurrent）、2605.30757、2512.02677、2604.07822、
2609.33144、2510.18825、2502.12352。

## 10. 许可

MIT（见 `LICENSE`）。
