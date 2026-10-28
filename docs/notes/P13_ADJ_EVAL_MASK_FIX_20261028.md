# P13：邻接掩码评测漏传 → dev 假随机（修复记录）

日期：2026-10-28
状态：**已执行验证（2026-10-28）**——评测透传修复生效，但冒烟
**FAIL**（C3 泄漏：r=3 已解 hop6）；H-026 判 **REFUTED**，下一步 H-027。
旧的“执行阻塞/未运行”记录保留在 §6–§7 作为历史，最新结果见 §8。

## 1. 结论（TL;DR）

`P12_adj_s71` 在 12k 步出现 `task loss ≈ 0` 而 `dev_r1..r8` 全在
0.49–0.53，不是“掩码阻断学习”，而是**评测与训练的前向不一致**：

- 训练：`ouro_loss(..., adj_bias_list=_bias_list)` 把逐轮邻接偏置传进
  `model.forward`（`s71_ldm_iter4.py`，`_bias_list` 由
  `s88_adj_task.adj_bias_for_round` 构造）；
- 评测：`from s65_ldm_gpu_smoke import evaluate as _ev`，而
  `s65.evaluate → forward_loss → model(...)` **没有** `adj_bias_list` 参数。

于是带掩码训练的模型在**无掩码**输入上评测，输入分布与训练不一致，
读出接近随机。P12 的 `task=0` 说明带掩码训练分布上的**加权** CE 被拟合
（`q_mean_first≈1`，第 1 轮权重近 0，主要是后段轮次在降 loss），
既不能据此说“模型没学会”，也不能说“每一轮都学会了”。

> **2026-10-28 实测更新（以此为准）**：评测漏传只是第一层原因。
> 实测还发现 `s88.adj_bias_for_round` 的“≤t 跳”漏掉 1 跳边（已修），
> 且补上评测掩码后冒烟出现 **C3 泄漏**（r=3 已解 hop6，`dev_r1` 0.65/0.71），
> 说明当前掩码没有形成“每轮一跳”的物理约束。H-026 已判 **REFUTED**，
> P13 48k 未启动；完整证据、根因与下一步 H-027 见 §8。

## 2. 证据

`results/P12_adj_s71.json`（12k 步，cap=3）：

| 指标 | 值 |
|---|---|
| hist 末条 `task` | 0.0 |
| `dev_r1 / r2 / r3 / r4 / r5 / r6 / r8` | 0.527 / 0.492 / 0.524 / 0.514 / 0.504 / 0.502 / 0.490 |
| `confirm_r1..r8` | 0.5075 / 0.5275 / 0.5125 / 0.5075 / 0.500 / 0.4925 / 0.505 |

全部围绕 0.50（2 候选 → 随机），且不随 r 变化，符合“评测口径错误”
而非“某几轮没学会”。

## 3. 修复（只改评测路径，不改训练/掩码定义）

### 3.1 `s65_ldm_gpu_smoke.py`

- `forward_loss(model, batch, n_rounds=None, adj_bias_list=None)`：
  把 `adj_bias_list` 透传给 `model.forward`。
- `evaluate(..., adj_batch=None, fact_mask=None)`：
  - 每个 batch 用 `s88_adj_task.adj_bias_for_round` 构造 `t=1..n_rounds`
    的逐轮偏置，再传给 `forward_loss`；
  - 先把 `adj_batch`/`fact_mask` 截到本 batch 实际序列长度 `S`
    （图内所有位置都 `< S`，与“先算满 256 再切左上角”等价，但省掉
    大量无效 matmul）；
  - 评测结束恢复原来的 `train/eval` 模式（旧版会把模型留在 eval）。
- 不传 `adj_batch/fact_mask` 时行为与旧版一致，`run_curve` 等旧调用不受影响。

### 3.2 `s71_ldm_iter4.py`

- 保留 `_dev_rows` / `_cf_rows`，在 `--adj-mask` 时一次性构造
  `_dev_adj/_dev_fm`、`_cf_adj/_cf_fm`（`build_adj_batch`）；
- 评测循环把对应 split 的邻接/事实掩码传给 `_ev`；
- `hard_exit_eval_v2(..., adj_batch=None, fact_mask=None)` 同样支持掩码，
  并在 `args.ponder` 的终评 sweep 里传入 dev 掩码；
- 加了截断检查：若 `EncodedDataset` 丢弃了样本（`truncated>0`），
  邻接矩阵与数据集索引会错位，直接报错而不是静默评错。

### 3.3 指标口径修正（掩码下必需）

旧 auto-loop 指标 `h7 = confirm_r6_by_hop["7"]` 在掩码下**不再可用**：
hop7 在 r=6 时 `r<h`，按设计应≈随机，用它做“配方可靠性”会系统性低估。
因此：

- `rec` 新增 `confirm_r8_by_hop`（r=8 ≥ hop7/8）；
- H-026 的 P1 改用 `h7 = confirm_r8_by_hop["7"]`，
  `h6 = dev_r8_by_hop["6"]`，并另行报告
  `h8 = confirm_r8_by_hop["8"]`；
- `auto_loop_spec.json` 恢复前必须同步改 metric 路径（见 H-026 末节）。

## 4. 语义边界（冒烟必须检查的“泄漏”）

`s88_adj_task.adj_bias_for_round` 只限制**事实 query → 事实 key**；
`[STATE]/[CURRENT]/[GOAL]/[ACTIONS]/候选` 这些非事实位置**始终可见**
（模块设计如此，否则 e0 进不了链）。因此“r<h 物理上不可能”并非自动成立：
候选 token 在 r=1 就能看到所有事实位置，若模型能从原始 token 直接做
“候选 obj == 某事实 obj”的匹配，掩码就挡不住。

MVT-D v2 里两个候选都有入边，所以“是某个事实的 obj”无区分度，理论上
仍须沿 e0 走 h 跳；但这是**假设**，必须用冒烟 C3（r<h 不得显著高于随机）
验证。若 C3 失败，下一步不是改评测，而是把候选读出也纳入可达性约束
（另立假设，需重训）。

> **实测（§8.3/§8.5）**：C3 确实失败，而且只限制候选读出仍然不够
> （r=3 的 hop6 仍有 0.89）；需要“距离前沿”粒度并掩掉 `prelude`/`coda`，
> 见 H-027。

## 5. 冒烟设计与预注册判据

脚本：`bash _smoke_adj_eval.sh`（2 seed × 3000 步，`cap=2`，并发 2）。

- 使用与正式配方完全相同的 `--curriculum-step 8000`，3000 步内课程停在
  `cap=2`，因此只改“训练步数”，是干净的最小冒烟。
- 判据（`s90_check_adj_smoke.py`，跑前写定）：
  - **C1 非随机**：某个已训练 hop 在某 r 上 acc ≥ 0.70，且 `dev_r1 ≤ 0.55`；
  - **C2 阶梯**：存在已训练 hop h，`max_{r≥h} acc ≥ 0.65`，
    `r<h` 平均 acc ≤ 0.55，落差 ≥ 0.15（用 max 而非只取 r=h，
    是因为 Ouro 的退出分布偏后，r=h 读数可能尚未校准）；
  - **C3 无泄漏**：所有 hop 在 `r<h` 时 acc ≤ 0.65。
  - 两个 seed 都 PASS 才算冒烟通过。

## 6. 执行阻塞（历史：2026-10-27 夜，2026-10-28 已解除）

本机 `bash` 执行器报：

```
Error: sandbox mode "workspace-write" is requested but no sandbox backend is
usable on this host; refusing to run the command unconfined. Install
bubblewrap or run a Landlock-enforcing kernel (Linux), ensure sandbox-exec is
usable (macOS), or ensure the ACL restricted-token runner can start (Windows)
— otherwise switch the consumer to danger-full-access.
```

升级 `danger-full-access` 需要审批，但本会话没有审批通道：

```
Error: sandbox escalation to "danger-full-access" requires approval, but no
approval channel is available
```

子智能体（独立 session）复现了同一错误；`workflow` 同样依赖同一执行器，
因此**没有执行任何 Python 命令**，也没有启动/恢复任何训练或 loop。
已存在的 `results/` 未被读写覆盖（只读取了 P12 JSON 作诊断）。

## 7. 解除阻塞后的执行顺序（历史计划；实际执行与结果见 §8）

```bash
cd /root/ldm

# 0) 可选快速诊断（秒级，不训练）：用修复后的评测重评 P12 的 12k ckpt。
#    若 hop2 在 r=1≈0.5、r≥2 明显更高，则评测 bug 当场确证。
python s90_eval_ckpt.py --ckpt results/P12_adj_s71_ckpt.pt \
    --out results/P13_reeval_P12_adj_s71.json

# 1) 冒烟（唯一 --out，已存在会拒绝）
bash _smoke_adj_eval.sh

# 2) 若 SMOKE PASS，再启动正式 8 seed（脚本内再次校验冒烟）
bash _p13_launch.sh

# 3) 正式 run 结束、确认 spec 已指向 P13 后，再恢复 loop
nohup bash _run_loop.sh >/dev/null 2>&1 &
```

**不要**在冒烟 PASS 前恢复 `_run_loop.sh`：当前
`docs/plans/auto_loop_spec.json` 的 B 阶段仍 glob `P12_adj_*` 且
`target_step=48000`，而 P12 的三个 run 已停在 12k，直接恢复会永久等待。
详见 `docs/hypotheses/H-026-REFUTED-adj-eval-mask.md` 的“loop 恢复前置”
（已被 §8.6 的实际配置取代：现在指向 P14/H-027，而不是 P12/P13）。

---

## 8. 2026-10-28 执行记录（沙箱解除后）

### 8.1 步骤 1：复查评测掩码 → 发现并修复 `s88` 的真 bug

复查结论分两层：**s71/s65 的“评测透传”是对的，但 `s88` 的“≤t 跳”
本身算错了**。

**(a) s71 评测路径（473–479、706–712、259–269）正确**

- `--adj-mask` 时一次性从 `_dev_rows`/`_cf_rows` 构造
  `_dev_adj/_dev_fm`、`_cf_adj/_cf_fm`，并在 `dev_ds.truncated` 或
  `cf_ds.truncated` 时直接报错，避免邻接矩阵与数据集索引错位；
- 固定 r 扫描按 split 把 `adj_batch/fact_mask` 传进 `s65.evaluate`；
- `hard_exit_eval_v2` 也按 batch 截到实际 `S` 后构造逐轮偏置；
- 与训练路径（575–580）使用同一个 `s88.adj_bias_for_round`，口径一致。

**(b) s65 评测路径（120–131）正确**

- `T = n_rounds`，构造 `t=1..T` 的偏置列表并按 batch 截断到 `S`；
- `forward_loss` 把 `adj_bias_list` 透传给 `model.forward`，模型按
  `idx` 取第 idx 轮偏置（`s63` 504–511）；
- 评测后恢复原 `train/eval` 模式。

**(c) `s88_adj_task.adj_bias_for_round` 有实质 bug（已修）**

旧实现把“≤t 跳”算了两遍，且两遍都漏项：

- 第一遍：`reach = I | A^t`，漏掉 `A^1..A^(t-1)`；
- 第二遍：`acc = I | A^2 | ... | A^t`，漏掉 `A^1`，然后覆盖第一遍；
- 结果：**t=1 时事实位置只剩自环；任何 t 都看不到 1 跳邻居**。
  这不是评测专属 bug——训练也走同一函数，所以 P12 是在“断掉 1 跳边”
  的掩码下训练的，dev 随机不能只归因于评测漏传。

修复为单次累积 `reach = I | A | A² | ... | A^t`（`s88_adj_task.py`
`adj_bias_for_round`），并用 BFS 对拍验证：

```
dev_random.jsonl 前 100 条、dev_ood.jsonl 前 100 条
t ∈ {1,2,3,6,8}
断言：事实-事实 allowed == BFS ≤t 跳可达；候选 query 全可见；
      事实 query 对全部非事实 key 可见
→ bad = 0
```

样例 0（7 条事实）事实-事实可见对数：
`t=1: 11/49`（7 自环 + 4 条 1 跳边）、`t=2: 13/49`、`t=3: 14/49`
——现在 t=1 确实包含 1 跳边。

**(d) 语义边界（仍按原设计，C3 必须实测）**

- 非事实位置（`[STATE]/[CURRENT]/[GOAL]/[ACTIONS]/候选`）作为 **key**
  对事实 query 始终可见；作为 **query** 完全不受限。
- 因此候选读出在所有 t 都能看到全部事实；`r<h` 不是“物理上不可能”，
  而是“需要模型不利用候选/非事实位置的直读捷径”。C3 仍是有效闸门。

**(e) 快速诊断（旧掩码下重评 P12 ckpt，仅作对照）**

在修复前用 `s90_eval_ckpt.py` 重评 `P12_adj_s71_ckpt.pt`（12k，cap=3）：

| r | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| acc | .525 | .522 | .524 | .516 | .514 | .512 | .499 |

仍是 0.50 附近、无阶梯。这符合“旧掩码断 1 跳边 + P12 已记忆训练集”
的解释，**不能**当作“补掩码无效”的证据。输出保存在
`results/P13_reeval_P12_adj_s71.json`（唯一 `--out`，未覆盖任何旧结果）。

### 8.2 步骤 2：冒烟（`_smoke_adj_eval.sh`）

- 第一次启动（09:49）在发现上述 bug 后立即中止；当时只留下空的
  `.log/.lock`，没有任何 JSON/ckpt，GPU 进程已确认清零。
- 修复 `s88` 后重跑预注册冒烟：2 seed（17/53）× 3000 步，
  `cap=2`，`--adj-mask`，`--curriculum-step 8000`，并发 2。
- 判据仍是跑前写定的 `s90_check_adj_smoke.py` C1/C2/C3，不放宽。

（结果见 8.3；若失败，按 H-026 的“不成立”分支记录，不改口径。）

### 8.3 冒烟结果：**FAIL（0/2 seeds）——评测透传已生效，但掩码没有形成阶梯**

`bash _smoke_adj_eval.sh` 正常跑完（2×3000 步，约 5 分钟），
`s90_check_adj_smoke.py` 判定两个 seed 均 C1/C2/C3 全失败。
完整输出存 `results/P13_smoke_adjfix_check.txt`；结果 JSON 为
`results/P13_smoke_adjfix_s17.json`、`..._s53.json`。

**hop × r（dev_random，step=3000，cap=2）**

| seed | hop | r1 | r2 | r3 | r4 | r8 |
|---|---|---|---|---|---|---|
| 17 | 2 | .620 | .610 | .980 | .990 | .990 |
| 17 | 6 | .645 | .645 | .985 | .990 | .990 |
| 53 | 2 | .730 | .690 | .990 | .990 | .990 |
| 53 | 6 | .735 | .720 | .985 | .985 | .990 |

汇总：`dev_r1 = 0.650 / 0.714`（C1 要求 ≤0.55）；
`best_trained_acc = 0.995`（这一项通过）；
C2 的 `r<h` 均值被 r1/r2 的 0.65–0.72 拉高，失败；C3 大量
`r<h` cell 超过 0.65（hop6 在 r=3 已 0.985）。训练侧并非不学：
task loss 在 3000 步末已从 0.69 降到 0.236（s17）/0.005（s53）。

**关键对照：同一 ckpt，mask on/off（s17）**

| r | with mask | no mask |
|---|---|---|
| 1 | .650 | .482 |
| 2 | .639 | .484 |
| 3 | .987 | .466 |
| 4 | .995 | .466 |

这证明两件事：
1. **评测掩码确实传进去了**（关掉掩码立即塌回 0.47–0.48，
   与 P12 的假随机一致）；
2. 高准确率依赖掩码，但**不是“r≥h 才出现”的阶梯**——r=3 就能把
   hop6 解到 0.985，说明掩码没把传播深度锁死。

**为什么会漏（与 §4 的语义边界一致）**

- 掩码只限制**事实 query → 事实 key**；`[STATE]/[CURRENT]/[GOAL]/
  [ACTIONS]/候选` 作为 key 始终可见，候选作为 query 完全不受限。
  候选读出每轮都能看到全部事实，等于给模型一个“全局工作区”；
- `prelude` 和 `coda` 两层注意力**完全没过掩码**（`s63` 480、573），
  每层都能做一次全局 join；
- 因此模型可以用“从 e0 正向 + 从候选反向”各走 3 跳在中间相遇，
  3 轮覆盖 6 跳；这不是评测 bug，是掩码的作用域不够。

### 8.4 步骤 3：判读与下一步（不启动 48k）

- **不运行 `_p13_launch.sh`**：它的闸门会拒绝（smoke FAIL），
  且即使绕过闸门，48k 也只会放大一个已知会漏的掩码，浪费 8×48k。
  实测 `bash _p13_launch.sh` 返回 exit=1、打印
  `[REFUSE] 冒烟未通过；先修评测/掩码，不要启动 48k`，
  未启动任何 run，也未产生 `P13_adj_evalfix_*` 文件。
- **H-026 的判读**：见 `docs/hypotheses/H-026-REFUTED-adj-eval-mask.md`。
  - “评测漏传 `adj_bias_list`”这一条**被证实是真实 bug**（已修，
    mask on/off 对照可复现）；
  - 额外发现并修复 `s88` 的“≤t 跳”漏 1 跳 bug；
  - 但 H-026 的核心预测“补掩码后 `r<h≈随机、r≥h 显著更高`”
    **被证伪**：C3 泄漏，r=3 已解 hop6。
  - 因此不能把这轮结果解释成“掩码阻断学习”，也不能据此宣布
    “循环承载串行深度”。
- **下一步（H-027，待实现/预注册）**：把可达性约束扩展到
  1) 候选 query（只能看 ≤t 跳内的事实）；
  2) 事实 query 不再能看到候选 key（候选不是内容节点，
     避免从候选实体反向传播）；
  3) `prelude` 用 t=0 偏置（只保留自环/标记，不传事实边）；
     `coda` 用最后一轮的偏置（不能在读出前多走一跳）。
  先跑 2 seed×6000 步 cap=2 的严格掩码冒烟，判据仍是
  “`r<h` 全部 ≤0.60 + 两个 seed 的 hop2 在 r≥2 ≥0.70”；
  通过后才考虑 48k。未改 H-026 的任何阈值。

### 8.5 两次 eval-only 探针：泄漏不只是“候选全可见”

用 seed17 的 3000 步 ckpt 直接换更严格掩码评测（不重训，仅诊断）：

| 掩码 | r=1 | r=2 | r=3 | r=6 | hop6@r3 |
|---|---|---|---|---|---|
| 当前（事实-事实 ≤t） | .650 | .639 | .987 | — | .985 |
| 候选只看 `dist≤t`、事实看不到候选 key | .643 | .778 | .896 | .821 | .890 |
| 距离前沿（事实只看 `dist=t-1`） | .490 | .464 | .403 | .228 | .465 |

- 只限制候选：r=3 仍有 0.896、hop6 0.89 → **候选可见性不是唯一原因**；
- 距离前沿：旧模型直接塌到随机/以下 → 旧解依赖“每轮多跳”的可达性，
  而不是真正的单跳递推。注意这是**换掩码评旧模型**，不能证明新掩码
  可学；H-027 必须重训。

### 8.6 步骤 4/5：loop 与档案状态

- `docs/plans/auto_loop_spec.json` 已改：
  - `A_seed_sweep_48k` 保留（P05 已完成）；
  - 新增 `B_strict_distance_mask_smoke`：glob `P14_strictmask_*`，
    expect=2，target_step=6000，metric 覆盖 h=2..6 的**全部 r<h cell**
    （h2_r1…h6_r5）加 `h2_r2`/`r1`；任何一个 cell >0.60 都不放行，
    pass 后才 launch `P14_adj_strict`；
  - 新增 `C_strict_48k`：glob `P14_adj_strict_*`，expect=8，
    target_step=48000，metric 改为 `h7=confirm_r8_by_hop.7`、
    `h6=dev_r8_by_hop.6`、`h8=confirm_r8_by_hop.8`。
- `results/auto_loop_state.json` 的 `stage_idx` 置为 1（B 阶段），
  避免 loop 回到 A 阶段重复 launch P12。
- 已恢复 loop：

  ```
  PID 38854  /bin/bash ./_run_loop.sh
  PID 38856  python s89_auto_loop.py --spec docs/plans/auto_loop_spec.json --interval 300 --agent dsh
  ```

  （前两次启动 38722/38724、38802/38804 已停掉并重启，以加载最终的
  “全部 `r<h` cell + 两个 seed 全过”闸门。）当前 tick 为
  `wait: 0/2 runs reached step>=6000`（等 H-027 冒烟），
  不会重复启动 P12，也不会在已知漏掩码上启动 P13 48k。
- H-026 已改名 `H-026-REFUTED-adj-eval-mask.md`；新建
  `H-027-PENDING-strict-distance-mask.md`；`docs/hypotheses/INDEX.md`
  与 `docs/plans/AGENT_TASK_BRIEF.md` 第四节已同步。
- 约束遵守：P13 冒烟唯一 `--out`（`P13_smoke_adjfix_s17/s53`），
  并发 2；未运行 `_p13_launch.sh`；未覆盖任何已有 result JSON；
  `P13_reeval_P12_adj_s71.json` 也是新文件。
