# V6 训练计划：Jaume 700M 原生循环 + MoE（Loop-MoE Embedding 模型）

> 版本：V6，2026-10-09 起草。
> 状态：**纯规划文档，未启动任何训练，未占用 GPU。**
> 前置：`TRAINING_PLAN_V5_HYBRID_LOOPUS.md`（Qwen3.5-4B retrofit 线）、`TRAINING_METHOD.md`。
> 目标：设计一个 700M 原生（from-architecture）循环深度 + MoE 的 embedding 模型，
> 解决"循环深度 × MoE 路由混乱"问题，而不是先训普通模型再改造。

---

## 0. 定位：V6 与 V5 的关系

V5（HybridLoop / Qwen3.5-4B）回答的问题是：**在冻结的混合架构大模型上，
后训练能否注入循环深度**。结论是"可以，但天花板受限"：
v4-c 实现了质量无损的逐 token 自适应退出（CE 1.917–1.923，平均退出深度 4.74），
但冻结主干的深度价值结构无法被路由器改变（GSM8K 深度收益块始终只有 2/16），
且无 KV cache、基础数学能力弱。

V6 换一个问题：**如果从架构设计阶段就原生采用循环深度 + MoE，
能否得到比"retrofit"和"纯 dense"都更好的 embedding 模型？**

V5 的角色从"主线"变为"方法验证器"：它的失败教训（冻结损伤、halting predictor
不可学、复读病）全部转化为 V6 的设计约束，见 §2。

---

## 1. 底模事实（已核查）

来源：ModelScope `LightNeko/Jaume-Embedding` 仓库 API 与代码包。

| 项 | 值 |
|---|---|
| 350m 基础配置 | 16 层，hidden 512，intermediate 2048 |
| 注意力 | 8 heads / 4 KV heads（双向 GQA） |
| FFN | **已是 MoE**：expert_num=4，topk=2 |
| 序列长度 | 8192 |
| 输出 | attention pooling → projection → L2 norm |
| embedding_dim | 512，Matryoshka dims [128, 256, 384, 512] |
| 官方阶段 | Stage1 MLM → Stage2 对比(MNRL/InfoNCE+hard neg) → Stage3 hard neg → Stage4 Matryoshka → Stage5 decision |
| 已知退化 | 英文对比训练导致 STS22 多语言退化（官方 PLAN 记录） |
| decision-v2 | choice acc≈0.9975，noul AUROC≈0.837 |

**关键事实：Jaume 本身就是 MoE 模型**。所以 V6 的问题不是"要不要加 MoE"，
而是"已有的 4-expert MoE 如何与循环深度共存而不路由塌缩"。

---

## 2. 复用自 V5 的已验证资产

以下全部是本项目跑出来的实证结论，不是推测：

| 资产 | 来源 | 在 V6 中的用途 |
|---|---|---|
| 转折点存档协议（append-only 结论表、atomic write、rolling ckpt、RNG 保存/恢复、确定性数据顺序） | V5 全程 | V6 训练器直接沿用，应对断电 |
| 冻结 vs 策略三模式对照法（A 全跑 / B 只记录 / C 冻结） | `s10_freeze_test.py` | V6 每个路由改动的标准验收工具：分离"路由策略损伤"与"状态冻结损伤" |
| soft decay 退出：`h = h + keep*(h_new-h) + (1-keep)*0.05*(h_new-h)` | v4-c，freeze damage 从 +0.273 降到 0 | V6 adaptive exit 阶段的默认退出形式 |
| "只训 router 不改变主干深度价值结构" | `s8_r0_value_gsm.py`（v3 0.58M 与 v4-c 2.5M 逐位一致） | V6 必须联合训练主干 + 深度机制，不能指望后期 router 补救 |
| token 级 halting 从标量特征不可学（AUC 0.50–0.52） | R1/R2 | V6 的退出必须是路由器改变轨迹，不是事后 predictor |
| h0 再注入修复复读病 | L1-v2 | V6 循环核心第 2 圈起注入初始表征 |
| 协议 v1.1 严格评分（段界完成 + 42/42 对抗门） | `scorer_strict.py` | V6 生成式探针复用 |

---

## 3. 核心风险：循环深度导致 MoE 路由混乱（论文证据）

这是用户提出的核心质疑，答案是：**是的，风险真实存在，且不能只靠 load-balance
loss 避免；但已有论文证明可控可行，条件是路由必须跨循环分化。**

### 3.1 风险是什么

| 风险 | 机理 | 证据 |
|---|---|---|
| 跨循环路由塌缩 | 同一 token 每圈选同一组 expert → 循环退化为重复计算，深度增益消失 | Sparse Layers (2605.09165)：**路由跨 loop 分化是 Loop-MoE 成功的关键机制**，反之为失败模式 |
| expert 塌缩 / 死专家 | 循环放大热门 expert 的马太效应 | Sparse Upcycling (2212.05055)：load balance 过小死 expert、过大压主任务 |
| 小数据 MoE 劣于 dense | 对比数据不足时稀疏性无法被利用 | Nomic Embed v2 (2502.07972)：100M pairs 时 MoE 可能低于 dense；235M pairs 才稳定获益 |
| 退出与路由互相干扰 | halting 信号和 router 信号同时训练，梯度冲突 | 本项目 V5：v4 hard freeze CE +0.273 的教训（结构不同但同源：硬切换损伤轨迹） |

### 3.2 论文给出的解法（按证据强度排序）

1. **Sparse Layers are Critical to Scaling Looped Language Models**（arXiv:2605.09165）
   ——直接研究 dense/looped/MoE/Loop-MoE 四格对照。结论：dense looped scaling
   不如标准模型，**MoE 稀疏层恢复 looped 表达能力**；E=8、top-2、标准
   load-balance + router z-loss；loop boundary 是天然 early-exit 点。
   ⇒ V6 主案直接采用 Loop-MoE，且必须有对照组"native loop dense"来复现
   "dense looped 更差"这一前提。

2. **Tied Expert Layers**（arXiv:2606.16825）
   ——expert FFN 权重跨层共享，attention/router/norm 保持独立；2+2 untied
   prelude/coda 对质量重要；ablation 显示 router 是否独立影响小、attention
   是否独立影响大。
   ⇒ V6 循环核心共享 expert FFN 池（同层循环复用天然就是 tied），但
   **E/D 两段和全部 attention/norm 不共享**。

3. **Layerwise Recurrent Router (RMoE)**（arXiv:2408.06793）
   ——独立 per-layer router 缺历史信息；用低维 projector + GRU 维护跨层路由
   状态，recurrent gradient 关键；比加深 router MLP 更有效，提升 expert
   diversity 和 exploration。
   ⇒ V6 把 RMoE 的"跨层状态"改造成"跨圈状态"：router 输入 = 当前 hidden
   + 上一圈路由历史的 GRU 状态，这是防止跨圈塌缩的结构性手段。

4. **Nomic Embed v2**（arXiv:2502.07972）
   ——embedding MoE 直接先例：alternating 层 upcycle 为 8 experts、只改 6 层
   优于全层、235M weakly-supervised 对比对。
   ⇒ V6 对比阶段数据量按 ≥200M pairs 规划，MoE 只放中间层。

5. **Upcycling 系列**（2212.05055 / 2410.07524）
   ——dense→MoE：virtual group 初始化、weight scaling、softmax-then-topK、
   load balance。upcycle 不必然优于从零，取决于计算预算。
   ⇒ 回答"是否先训稠密再稀疏化"：**V6 采用"继承式部分 upcycling"**——
   从 Jaume 350m 已有 MLM/对比权重初始化 700M 的非循环部分和 expert 池，
   不做完整 dense 700M 预训练（预算不允许，且论文表明非必然更优）。

6. **原生循环侧**：Universal Transformer（1807.03819，ACT 逐位置停机）、
   LARM（2606.04678，循环 encoder + CTC checkpoint + FiLM 深度条件）、
   RecursiveGPT（2608.26973，有限数据下共享循环更优）、LoopUS（B=20/K=5
   稀疏深度监督）。
   ⇒ V6 的深度监督用 LoopUS 配方（已在本项目 V5 验证过训练动态），
   深度条件用 LARM 式 FiLM/clock embedding 帮助 router 区分圈数。

### 3.3 对"路由混乱"的完整对策清单

- dense-like warmup：R=1 阶段模型退化为标准 Jaume，先稳定再开循环；
- MoE 只放循环核心 M 段，E/D 两段 dense（Tied Expert + Nomic v2 双证据）；
- shared expert：1 个始终激活的 dense expert 保底，top-2 从 4 个 routed expert 选；
- router 输入显式含圈数编码（FiLM / clock embedding），让"第几圈"可路由；
- RMoE 式 GRU 跨圈路由状态（结构性反塌缩）；
- load-balance loss + router z-loss + softmax-then-topK（数值稳定三件套）；
- 全程监控（§6），任一指标越界即触发诊断存档而不是继续盲跑；
- adaptive exit 最后加（S7），且用 v4-c soft decay，绝不 hard freeze。

---

## 4. 700M 架构设计（主案）

### 4.1 E/M/D 切分

从 Jaume 350m 的 16 层扩展为 24 层结构：

```
E 段（encoder 进入）：L0–5，6 层，dense FFN，不循环
M 段（recurrent core）：L6–17，12 层物理结构，循环 R∈{1..8}
                         FFN = shared expert(1) + routed experts(4, top-2)
                         attention / norm per-layer 独立
D 段（decoder 收尾）：L18–23，6 层，dense FFN，不循环
```

- hidden size 512 → 768（参数主要增量来源），intermediate 2048 → 3072；
- 参数预算目标：总存储 ≈700M，active ≈420–470M（top-2 of 4 + shared）；
  **最终以脚本核算为准，本节数字是设计起点不是承诺**；
- 第 1 圈恒等 + 不注入 h0；第 2 圈起 `h + h0`（V5 L1-v2 反复读配置）；
- 深度条件：每圈 FiLM 调制（scale/shift 由圈数 embedding 生成）作用于
  M 段每层 attention 输出，帮助 router 与 attention 区分圈语义（LARM）。

### 4.2 MoE 细节

| 项 | 值 | 依据 |
|---|---|---|
| expert 数 | shared 1 + routed 4 | Shared expert 保底（防死专家兜底） |
| top-k | 2（softmax-then-topK） | 2605.09165 / upcycling 系 |
| 初始化 | routed experts 从 Jaume 350m expert 权重复制 + virtual group 扰动；shared expert 用 350m dense 等价路径均值 | 2212.05055 / 2410.07524 |
| router | per-physical-layer MLP + 圈数 embedding 拼接 + GRU 跨圈状态 | RMoE 2408.06793 |
| 损失 | load-balance（aux 系数 0.01 起调）+ router z-loss（1e-3） | 2605.09165 |
| 范围 | 仅 M 段 | Tied Expert / Nomic v2 |

### 4.3 退出机制（最后阶段才启用）

- v4-c 式 monotonic-mask router：逐 token、逐圈二值 keep/exit + soft decay
  （`h = h + keep*(h_new-h) + (1-keep)*0.05*(h_new-h)`）；
- bottom-K ponder loss 约束平均深度；
- **在 S7 之前模型必须已收敛**，退出层只在最后微调（V5 教训：联合从头训
  退出与主干 = 死轨迹风险）。

---

## 5. 训练阶段

每个阶段都有明确 go/no-go 门，未过门不进下一阶段。全程沿用 V5 的
转折点存档 + 协议 v1.1 + atomic checkpoint。

### S0 基线冻结（不训练）
- 跑通 Jaume 350m 官方权重推理，记录 MTEB/STS22 多语言基线分数；
- 核实 350m 每阶段 checkpoint 的实际指标（MLM / stage2 / stage3）；
- 产出：`results/v6_s0_baseline.json`。

### S1 dense-like warmup（R=1，无新机制）
- 700M 结构、R=1：数学上等价于一个 24 层标准 MoE embedding 模型；
- 从 Jaume 350m 权重继承初始化（部分 upcycling）；
- MLM 训练至 CE 曲线平台；
- 门：MLM CE 不差于 350m 同 step 的 110%（参数翻倍应更好，留容差）。

### S2 原生循环 dense（对照组，不加 MoE 变化）
- 开 M 段循环 R∈{1,2,4,8}，LoopUS B=20/K=5 稀疏深度监督；
- FiLM 圈数条件启用；
- **预期：dense looped 相对 S1 劣化**（2605.09165 的前提复现）；
- 门：确认劣化幅度并记录 CE(d) 曲线；若反而更好，记录并仍继续 S3
  （Loop-MoE 仍要与之对照）。

### S3 循环核心 MoE（主案第一步）
- M 段 FFN 换 shared + 4 routed top-2；
- upcycling 初始化 + load balance + z-loss；
- 同 S2 预算重训 MLM；
- 门：CE(R=最优) 优于 S2 同预算；expert 无死亡（active load ≥1%）。

### S4 recurrent-aware router
- router 加入圈数 embedding + GRU 跨圈状态（RMoE 式）；
- 核心验收：**cross-loop expert overlap 显著低于 S3**（见 §6 指标定义），
  且 CE 再改善或持平；
- 若 overlap 不降：改用 per-loop 独立 router head（每圈一套投影）。

### S5 多语言大规模对比训练（Jaume Stage2–4 配方）
- MNRL/InfoNCE + hard negatives + Matryoshka，数据按 ≥200M pairs 规划
  （Nomic v2 教训：小数据 MoE 劣于 dense，此阶段数据量是硬前提）；
- 多语言配比必须覆盖 STS22 全语言（官方 PLAN 记录的英文偏科教训）；
- 门：MTEB 多语言子集 ≥ 350m 官方权重的 95%，Matryoshka 各维度不塌。

### S6 decision / calibration 微调
- 沿 Jaume Stage5/6 配方；门：choice acc ≥0.99，noul AUROC ≥0.83。

### S7 adaptive exit（可选终局）
- 移植 v4-c soft decay router，逐 token 退出；
- 验收用 V5 三模式对照法：策略损伤必须 =0，冻结损伤 ≤0.01 CE；
- 门：embedding 质量（检索 NDCG / STS）在平均深度 ≤50% 时损失 ≤1%。

---

## 6. 监控与诊断（S3 起全程，每 N step 落盘）

| 指标 | 定义 | 越界动作 |
|---|---|---|
| expert load | 每 expert 处理 token 比例直方图 | 任一 expert <1% 连续 1k step → 存档 + 提高 aux 系数 |
| router entropy | batch 级 router 分布熵 | 持续下降至 <0.3×初值 → 塌缩预警 |
| **cross-loop overlap** | 同 token 同物理层相邻两圈 top-2 集合 Jaccard，batch 平均 | >0.8 持续 → 触发 S4 router 改造/加重圈数条件 |
| per-depth routing | 每圈 expert 选择分布矩阵（圈×expert） | 某圈分布与其他圈相同 → 该圈退化标记 |
| CE(d) | 深度条件 CE 曲线 | d 增大 CE 不降 → 深度监督失效复查 |
| load-balance / z-loss | 损失分量曲线 | 异常放大 → 降学习率先于降系数 |
| Matryoshka 各维度 | 各截断维度相似度质量 | 低维塌 → Matryoshka 损失加权复查 |
| 生成探针 | 协议 v1.1 段界完成 + 复读率 | rep4>4 → 复查 h0 注入 |

所有诊断按 V5 惯例：异常自动存诊断档（不覆盖永久档），结论 append-only
写入本文档 §9。

---

## 7. 对照实验与验收表

同数据、同 step 预算、同评测协议下四组对照：

| 组 | 架构 | 角色 |
|---|---|---|
| C0 | Jaume 350m 官方权重 | 外部基线 |
| C1 | 700M dense MoE（R=1） | 参数量基线（S1 产物） |
| C2 | 700M native loop dense（S2 产物） | 复现"dense looped 劣化" |
| C3 | 700M native loop MoE + recurrent router（S4 产物） | 主案 |

**主案成功的判定**（全部满足才算 V6 成立）：
1. C3 ≥ C1（循环 + MoE 至少不输同参数 dense——2605.09165 的核心主张）；
2. C3 > C2（MoE 确实恢复了 dense looped 的损失）；
3. C3 多语言 ≥ C0 的 95%（700M 原生模型不弱于 350m 官方产物）；
4. cross-loop overlap 在合理区间（0.2–0.6：既要分化又不是完全乱路由）；
5. 若做 S7：平均深度减半时质量损失 ≤1%。

任何一条不满足，如实记录为负结果——本项目的结论表只收实测，不收愿望。

---

## 8. 明确不做的事

- 不从零随机初始化一次性联合训练 MoE + 循环 + halting（无论文支持，V5 有死轨迹教训）；
- 不做完整 dense 700M 预训练后再全面 retrofit（预算不允许，upcycling 论文
  表明非必然更优）；
- 不用小规模 STS/NLI 数据直接训完整 MoE 对比头（Nomic v2 明确劣化风险）；
- 不用 hard freeze / 硬切退出（V5 实测 +0.273 CE 损伤）；
- 不在本文档批准范围外占 GPU：**任何训练启动须用户明确指令**。

---

## 9. 结论表（append-only）

| 日期 | 阶段 | 结论 | 证据 |
|---|---|---|---|
| 2026-10-09 | 规划 | V6 文档起草，未启动训练 | 本文 |

---

## 附：论文依据索引

| arXiv | 简称 | 在本计划中的角色 |
|---|---|---|
| 2605.09165 | Sparse Layers / Loop-MoE | 主案可行性 + 反塌缩机制来源 |
| 2606.16825 | Tied Expert Layers | expert 共享范围与 prelude/coda 设计 |
| 2408.06793 | RMoE | 跨圈 recurrent router 结构 |
| 2502.07972 | Nomic Embed v2 | embedding MoE 数据量与部分层证据 |
| 2212.05055 | Sparse Upcycling | upcycling 初始化细节 |
| 2410.07524 | Upcycling into MoE | "是否先稠密后稀疏"的决策依据 |
| 1807.03819 | Universal Transformer | 原生循环 + ACT 谱系 |
| 2606.04678 | LARM | 原生循环 encoder + FiLM 深度条件 |
| 2608.26973 | RecursiveGPT | 有限数据下循环共享效率 |
| — | LoopUS（V5 主线） | B=20/K=5 深度监督配方（本项目已验证训练动态） |
| — | AdaPonderLM（v4-c 来源） | monotonic mask + ponder + 不冻结状态原则 |
