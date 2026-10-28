# V6 训练计划：Jaume 700M 原生循环 MoE Embedding

> 版本：V6.2（新增 §7 算力优化路径；S6 改为样本级退出并加入物理吞吐验收；对照采用 compute-matched 标准）
> 状态：规划中，未启动训练，未占用 GPU。
> 适用底座：ModelScope `LightNeko/Jaume-Embedding` 的 350M 系列；目标为约 700M 总参数的原生循环 MoE Encoder。
> 前置资料：`TRAINING_PLAN_V5_HYBRID_LOOPUS.md`、`TRAINING_METHOD.md`、`TRAINING_PLAN_V6R_JAUME_700M_LOOP_MOE.md`。

## 1. 目标和边界

### 1.1 目标

在约 700M **总参数预算不变**的前提下，从架构设计阶段就引入循环深度与 MoE，验证：

1. 原生循环 Encoder 是否能用更深的有效计算提升 embedding/decision 质量；
2. MoE 路由是否能在循环之间产生有益分化，而不是专家塌缩或路由漂移；
3. 相比同参数量 dense/MoE 基线，是否取得更好的参数效率；
4. 循环深度的算力收益能否**物理兑现**：不只是平均循环深度下降，
   而是同硬件实测吞吐/墙钟提升，以及在检索场景通过深度级联降低端到端编码算力（见 §7）。

### 1.2 不承诺的事项

- 不预先承诺循环深度一定提升 STS、检索或决策指标；
- 不把参数减少等同于墙钟速度提升；
- 不把 MoE 的总参数量等同于 active 参数量；
- 不把 350M 权重直接复制到不兼容的 hidden size/层数配置；
- 不在 RTX 5060 8GB 上承诺 700M 全参数预训练可行。

## 2. 已核实的 Jaume 事实

来源：ModelScope `LightNeko/Jaume-Embedding` 仓库 API、`config.json`、源码包与模型卡。

| 项目 | 已核实内容 |
|---|---|
| 模型类型 | 双向 Transformer Encoder（BERT-style embedding/decision model） |
| 350M 配置 | 16 层，hidden 512，intermediate 2048 |
| 注意力 | 8 attention heads，4 KV heads，RoPE，RMSNorm |
| FFN | MoE，4 experts，top-2，load-balance weight=0.01 |
| 输出 | attention pooling → projection → L2-normalized embedding |
| embedding | 512 维，Matryoshka [128, 256, 384, 512] |
| 序列长度 | 8192 |
| 已有阶段 | MLM → MNRL/InfoNCE embedding → hard-negative → decision-v2 |
| 700M checkpoint | ModelScope 页面说明 700M 尚未公开 |

Jaume 已经是 MoE。V6 的问题不是“是否增加 MoE”，而是“已有 MoE 如何和原生循环深度共存”。

## 3. 文献依据与采用方式

### 3.1 原生循环 Encoder

- **Universal Transformer**，arXiv:1807.03819：共享 Transformer transition 反复更新所有位置，ACT 可为不同位置分配不同计算步数。
- **LARM**，arXiv:2606.04678：循环 Encoder 使用稀疏中间监督、supervision clock、FiLM depth conditioning 和反馈；报告了参数匹配/更深非共享 Encoder 的比较。
- **RecursiveGPT**，arXiv:2608.26973：有限数据预算下，递归权重共享与 factorized embedding 解耦表示容量和每 token 计算。
- **Ouro/LoopLM**，arXiv:2510.25741：从预训练阶段建立循环深度，而不是只在后训练阶段补循环；其自适应退出用熵正则防止塌缩到全浅或全深，分两阶段（熵正则训练 → detached improvement 信号的门）。
- **Huginn / Recurrent Depth**，arXiv:2502.05171：3.5B 循环深度模型，测试时按需加深潜空间循环而不生成显式 CoT token，是"循环深度 = 测试时算力轴"的代表证据。

采用结论：700M 方案从原生循环架构开始，使用深度条件和稀疏深度监督。

### 3.2 Loop-MoE 和专家共享

- **Sparse Layers are Critical to Scaling Looped Language Models**，arXiv:2605.09165：Looped-MoE 比 dense looped scaling 更好；关键观察是同一物理层在不同循环中出现跨循环 expert routing 分化。配置包含 E=8、top-2、load-balance 与 router z-loss。
- **Tied Expert Layers in MoE LMs**，arXiv:2606.16825：共享 expert FFN，同时保持 attention/norm/router 的逐层差异；2+2 untied prelude/coda 重要；expert tying 的参数节省可达约 43–52%。注意：这是跨层 tying，不等价于完整循环，但可作为组件共享边界的证据。
- **RMoE**，arXiv:2408.06793：用低维 projector + GRU 维护跨层路由状态，改善路由协调、专家多样性与 router 梯度；V6 将其作为跨循环路由状态的候选，而非原文结论的直接复现。
- **Nomic Embed v2**，arXiv:2502.07972：MoE embedding 的直接先例；数据规模对 MoE 是否超过 dense 很关键，且部分层 MoE 优于全层转换。
- **Sparse Upcycling**，arXiv:2212.05055；**Upcycling LLMs into MoE**，arXiv:2410.07524：提供 dense checkpoint → MoE 的初始化、virtual grouping、weight scaling、load-balance 与路由顺序经验。
- **SMELT**，arXiv:2609.01343：中间半层循环两次的 looped-MoE，四个尺寸到 54B 非 embedding 参数；对 per-token FLOPs、总参数、KV cache 三项 compute-matched 后各自拟合 Chinchilla 式 scaling law，结论是 looping 是有效的效率杠杆而非训练技巧。V6 的对照预算对齐采用其 compute-matched 标准（见 §6）。相关工作 arXiv:2609.40316 给出 compute/内存约束下 looped-MoE 的设计用 scaling law。

## 4. 目标架构

### 4.1 参数预算原则

目标是约 700M **总存储参数**，同时报告：

- total/stored parameters；
- active parameters per forward；
- unique recurrent parameters；
- 每次循环的实际 FLOPs；
- 训练与推理的实际显存。

在 hidden/intermediate/layer 数定稿前，不把 420–470M active 作为承诺数字。必须先运行参数核算脚本。

### 4.2 E/M/D 结构

初版候选（V6.2 修订：加入算力兑现考量）：

```text
Prelude E：4 层，dense，attention/norm 不共享
Recurrent core M：12 个物理层，重复 R=1/2/4/8
Coda D：4 层，dense，attention/norm 不共享
```

E/D 占比直接决定算力兑现上限：早退只能省 M 段计算，
按等效层数估算可省上限 ≈ M×R / (E+D+M×R)。
R=8 时 E4/M12/D4 上限约 92%，E6/M12/D6 约 89%；R=4 时分别约 86% 和 80%。
因此 M 段取大、边界层取小。深度自适应推理文献（arXiv:2608.09444）同样发现：
边界层占比越高的循环模型，depth-adaptive 推理的相对收益越被摊薄
（Huginn 2-4-2 边界层占约一半参数，refill 策略收益受限）。
若 S1 稳定性结果要求更大的 E/D，须同步下调算力收益预期并记录在案。
层数最终以 S0 参数核算为准，本节数字是设计起点。

循环核心中：

- expert FFN 是主要共享候选；
- attention 保持逐物理层独立，或至少使用 depth conditioning；
- norm 保持逐物理层独立；
- 首尾层不循环；
- 第 1 圈保留 identity-first 路径；
- 第 2 圈起再注入 `h0`，但由 S1/S2 稳定性结果决定注入比例。

### 4.3 MoE 结构

第一版采用保守配置：

```text
shared expert：1 个，始终激活
routed experts：4 个
routing：top-2
MoE 范围：只在 recurrent core 的部分物理层起步
load-balance：从 0.01 起调，与 Jaume 现有配置对齐
router z-loss：从 1e-3 起调，需网格验证
```

不第一版采用：

- 所有层都转 MoE；
- 16/32 个专家；
- 完全移除 shared expert；
- 每圈一套完全独立专家参数；
- 与自适应退出同时从随机初始化联合训练。

### 4.4 Router 方案

主方案：**共享 router 参数 + 当前 hidden/depth 条件驱动跨循环分化**。

```text
router_input = project(h_t) + depth_embedding(t)
expert_logits = router(router_input)
```

备选方案：加入轻量跨循环状态：

```text
r_t = GRU(project(h_t), r_{t-1})
expert_logits_t = router(h_t, r_t)
```

不把“每圈独立 router”作为默认方案。Loop-MoE 论文的关键机制是共享 router 在不同 hidden state 上自然产生不同路由；Tied Expert 的消融也显示，router 是否独立对 loss 的影响小于 attention 是否独立。

## 5. 训练阶段

所有阶段沿用 V5 的：

- 原子 checkpoint：tmp → `os.replace`；
- `ckpt_latest`/`ckpt_prev` 滚动双份；
- append-only `history.jsonl`；
- RNG、optimizer、data-order 一并保存；
- 质量转折点永久存档，固定步数不自动冒充永久档；
- 训练期间看板和验收循环独立运行；
- 任一 go/no-go 失败即暂停，不盲目推进。

### S0：事实与硬件核验

产出：`results/v6_s0_baseline.json`、参数核算表、显存预算表。

必须完成：

1. 下载/读取 Jaume 350M 的 350m-mlm、stage2、stage3、decision-v2 配置与权重；
2. 核实 700M 官方 checkpoint 是否可得；
3. 运行参数核算，明确 hidden/intermediate/layer/expert/active 参数；
4. RTX 5060 8GB 只做 tiny/smoke/LoRA 验证，不把全参训练作为本地路线；
5. 服务器路线使用 bf16、梯度检查点、分布式 MoE 和 expert parallel；
6. 统一固定验证集，所有阶段用固定样本评测，禁止用“相同步数”跨数据配方比较。

**Go 条件**：参数账本闭合、配置可加载、smoke 前向/反向/保存恢复全通过。

### S1：原生循环 dense-like warmup

结构已是 E/M/D 循环结构，但 recurrent core 暂不启用 routed MoE，使用 shared/dense-like FFN。

- R=1 起步；
- 逐步扩展到 R=2/4；
- 使用 depth embedding 或 FiLM；
- h0 再注入；
- 稀疏深度监督：在 R=2、R=4、R=8 检查点计算 MLM/embedding loss；
- 保留 MoE balance 以外的基础稳定性指标。

**Go 条件**：

- MLM 验证 CE 不超过对应 Jaume 350M 基线的 110%；
- hidden norm 无爆炸；
- `CE(R)` 不出现持续单调恶化；
- 固定 R=1/2/4/8 均可恢复；
- 至少保存 S1 转折点：循环稳定、深度反转或深度平台出现时永久存档。

### S2：局部 MoE 开启

只在 recurrent core 的部分物理层开启 MoE：

1. shared expert 保持 dense 路径；
2. 加 4 routed experts，top-2；
3. 复制/初始化采用同结构权重继承或局部 upcycling；
4. router 初期使用 soft routing/噪声探索；
5. load-balance + router z-loss；
6. 不开启 adaptive exit。

**Go 条件**：

- 无 dead expert：每个 expert token load ≥1%（连续 1000 step）；
- 最大/最小 expert load 比值 <4；
- router entropy 不快速跌至初始值的 0.3 以下；
- cross-loop expert overlap 不能长期接近 1；
- MoE 验证 CE/embedding loss 不劣于 S1 超过 1%。

### S3：循环×MoE 稳定训练

在 S2 通过后扩展 recurrent core 的 MoE 范围。

重点监控：

- `expert_load[depth, layer, expert]`；
- `router_entropy[depth, layer]`；
- `cross_loop_overlap`；
- 专家参数相似度；
- per-depth MLM/contrastive loss；
- MoE auxiliary loss 占总 loss 比例；
- attention/norm 的 residual norm 与循环速度。

**Go 条件**：

- 至少 2 个循环深度的 expert assignment 有稳定差异；
- 不出现单一专家长期占据大多数 token；
- 循环深度增加不导致 embedding/decision 质量崩溃；
- 对照 C2（native loop dense）时，C3（native loop MoE）质量不劣且路由分化存在。

### S4：大规模 MLM/embedding 预训练

数据采用分级目标：

- **消融下限**：至少 200M weakly-supervised contrastive pairs；
- **生产目标**：参考 Nomic Embed v2 的 1.6B pairs 量级；
- 多语言对比数据必须混入，不能只依赖英语；
- MLM、MNRL/InfoNCE、hard negatives、Matryoshka 共同评估；
- embedding 训练时按 R=2/4/8 进行稀疏深度监督。

**Go 条件**：

- STS-B、BEIR/MIRACL、多语言检索不低于 350M 基线的 95%；
- Matryoshka 128/256/384/512 不塌陷；
- 所有语言的 expert load 不出现系统性死专家；
- 路由分化在验证集保持，而不是只出现在训练集。

### S5：Decision 微调

在 embedding 质量稳定后训练：

- category choice；
- importance score；
- relevance/noul；
- calibration；
- 多循环深度输出一致性。

**Go 条件**：choice accuracy、noul AUROC、ECE 与 Jaume decision-v2 基线对照；不以训练集指标替代验证集。

### S6：自适应退出（最后启用）

自适应退出必须在循环×MoE稳定后启用，使用 soft decay，不使用硬冻结：

```python
h = h + keep * (h_new - h) + (1 - keep) * eps * (h_new - h)
```

V6.2 修订——退出粒度与物理兑现：

1. **主退出粒度是样本级，不是 token 级**。Embedding 模型输出是整序列 pooled 向量，
   没有逐 token 生成语义。样本级退出把整条 query/doc 作为一个退出单元，
   退出后从后续循环的 batch 维度物理移除（batch compaction），
   后续圈的真实计算量随退出单调下降。V5 v4-c 的逐 token mask 是逻辑退出：
   soft decay 下状态每圈仍在更新，batch 内计算无法跳过，实测墙钟无收益。
2. **收敛判据**用相邻圈 pooled 表征余弦/keep 信号（与 v4-c 的 keep 同源），
   阈值扫描出 quality-compute Pareto，不拍单点。
3. **退出分布防塌缩**：监控 exit depth 直方图的熵。Ouro（arXiv:2510.25741）
   报告自适应门会塌缩到全浅或全深，需熵正则；出现塌缩先加正则再调阈值。

退出评估：

- 平均循环深度；
- **同硬件实测吞吐提升（物理指标，主验收）**；
- 样本 exit depth 分布与分布熵；
- quality-vs-compute Pareto（检索 NDCG@10、STS 对平均深度作图）；
- cross-loop expert routing 是否因退出而塌缩；
- 开启退出前后的 embedding/decision 质量差。

**通过条件**：

- 平均深度降至 full depth 的 50% 左右时，核心任务质量损失 ≤1%；
- **batch compaction 实测吞吐 ≥1.3×（同硬件、同数据流对照）**；
- 路由分化/专家负载保持健康，退出分布不塌缩。

## 6. 对照组

| 对照 | 架构 | 用途 |
|---|---|---|
| C0 | Jaume 350M 官方 checkpoint | 外部基线 |
| C1 | 目标参数量 dense Jaume，R=1 | 700M 参数基线 |
| C2 | 700M 原生循环 dense | 检验循环自身是否引入 scaling deficit |
| C3 | 700M 原生循环 MoE | 主案 |
| C4 | C3 + recurrent router state | 路由稳定性对照 |
| C5 | C3 + adaptive exit | 退出效率对照 |

对照预算对齐采用 SMELT（arXiv:2609.01343）的 **compute-matched 标准**：
per-token FLOPs、总参数、active 参数、（生成探针场景的）KV 显存逐项对齐后再比较质量，
不允许"参数多算力少"或"算力多参数少"的错位比较。

所有对照必须使用相同数据、相同验证集、相同训练 token 或 compute budget，并同时报告：

- total/stored parameters；
- active parameters；
- 每次 forward FLOPs；
- 训练时间；
- 推理时间；
- embedding/decision 质量。

## 7. 算力优化路径（V6.2 新增）

循环深度若不能兑现为物理算力节省，就只是一个更贵的模型——这正是 V5 的实际结局：
v4-c 平均退出深度 4.74/8，逻辑上省了约 40% 循环计算，但逐 token soft decay
下状态每圈仍更新、batch 内无法跳过计算，叠加无 KV cache，墙钟反而更慢。
本节把算力兑现拆成四层，前三层属于 700M 训练任务本身，第四层记录为部署选项。

### 7.1 架构层：循环段占比决定可省上限

早退只能省 M 段计算。按等效层数估算，E4/M12/D4、R=8 时 M 段占总计算约 92%，
是算力兑现的结构前提（详见 §4.2）。深度自适应推理研究（arXiv:2608.09444）
证实：边界层占比越高的循环模型，depth-adaptive 收益越被摊薄。

### 7.2 训练层：稀疏深度采样

沿用 LoopUS B/K 稀疏深度监督：每个 batch 只在采样的 K 个深度上加监督，
不全深度展开。训练 FLOPs 账本随 K 线性。S0 必须给出
"每 sup-token 训练 FLOPs vs dense 基线"的比值，超预算即减 K 或减 R，
不允许用"训练更久"掩盖循环训练更贵的事实。

### 7.3 推理层：样本级退出 + batch compaction（主兑现路径）

见 S6 修订。要点：逻辑深度下降不产生物理节省；embedding 场景改用样本级退出，
退出样本从 batch 维物理移除，后续圈计算量真实下降。这是 V6 相对 V5 的
关键架构动机之一：双向 encoder + pooling 的输出形式让样本级退出语义干净、
实现廉价，不需要生成式场景的 continuous batching/refill 机制。

### 7.4 应用层：深度级联检索

类比 Matryoshka 级联（arXiv:2205.13147；ESANN 2025 的轨迹嵌入两级检索
报告约 8× 计算时间节省）：

```text
离线：全库用低深度 d_low（如 R/4）编码建索引
在线：query 用 d_low 编码 → 粗筛 top-K 候选
      仅对 K 个候选用 d_full 重新编码精排
端到端编码算力 ≈ N·d_low + K·d_full，K≪N 时接近 d_low/d_full 比例的节省
```

这与 Jaume 已有的 Matryoshka 维度轴正交，构成"深度 × 维度"二维算力-质量平面。
S4 的稀疏深度监督天然保证 d_low 表征可用；S6 验收加入级联模式的
端到端 NDCG 与编码算力比。

### 7.5 部署层（记录，不在 700M 训练范围内）

生成式探针/decision 场景可借用 training-free 的循环模型加速工作，
模型定型后再评估：

- **Continuous depth batching**（arXiv:2608.09444）：退出 token 空位用新 token
  补齐 + depth-aware KV cache + 异步调度，Ouro 1.4B 1.32–1.53× 吞吐、
  Huginn 3.5B 1.30–1.37×；并给出延迟模型 t(Nc,Nb)=δ+γNb+αNc，
  解释了小 batch memory-bound 区"省 FLOPs 不省墙钟"的机理；
  其 KV 单槽共享在 Ouro 上崩溃（GSM8K 0.23% vs 78%），需模型侧 co-design；
- **WISE working-set 推理**（arXiv:2609.27373）：attention routing 比 hidden state
  更早收敛，前若干步发现块稀疏工作集、后期循环只在工作集内算注意力，
  4K context 全程 1.36×，说明循环深度的价值主要在后期表示精炼而非反复全局搜索；
- **FlashLoop**（arXiv:2609.29812）：跨循环冗余压缩（token 级 lazy 更新、
  loop-aware 稀疏注意力、KV 残差量化），无损精度下 1.64× 端到端、6× KV 内存；
- **MELT**（arXiv:2605.07721）：单 KV cache 解耦推理深度与内存。

## 8. 监控和永久存档

### 8.1 必须监控

1. expert token load；
2. router entropy；
3. cross-loop expert overlap；
4. per-depth expert assignment；
5. shared/routed expert 输出范数；
6. MLM/contrastive loss at each loop depth；
7. embedding cosine drift；
8. Matryoshka dimension quality；
9. decision calibration；
10. adaptive exit depth distribution。

### 8.2 永久存档触发

永久档**按测评转折点触发，不按固定步数堆档**：

- S1：循环稳定点/第一次深度反转；
- S2：MoE 开启后首次 expert load 平衡；
- S3：首次跨循环 expert routing 分化；
- S4：embedding/decision 成熟点；
- S6：adaptive exit 首次达到质量-算力 Pareto 门槛。

固定步数只使用 `ckpt_latest`/`ckpt_prev` 滚动保护，不自动变成永久档。

## 9. 硬件可行性

RTX 5060 8GB：

- 允许做 CPU/tiny/smoke、冻结底座、LoRA、单层 MoE/routing 小实验；
- 不适合 700M 原生循环 MoE 的全参数 MLM/embedding 预训练；
- 不得把本地小实验结果直接宣称为 700M 生产结果。

700M 主线需要外部设备：

- bf16；
- gradient checkpointing；
- distributed data parallel；
- MoE expert parallel；
- activation/optimizer sharding；
- 在 Ascend 910B 上先完成 torch_npu 等价性验证，再做大规模训练。

## 10. 最终 go/no-go

### Go

只有在以下条件同时满足时进入大规模 S4：

- S1 深度稳定；
- S2 无死专家；
- S3 cross-loop expert routing 分化且不漂移；
- embedding 验证集质量不劣；
- 参数账本、显存账本、FLOPs 账本闭合；
- 断点恢复和永久档验收通过。

### No-go / 回滚

任一条件失败：

1. 先保存当前转折点档；
2. 回滚到最近稳定档；
3. 降低循环次数或 MoE 范围；
4. 重新检查 expert load、router entropy、cross-loop overlap；
5. 不用继续训练时间掩盖结构问题。

## 11. 引用索引

- Universal Transformer：arXiv:1807.03819
- Matryoshka Representation Learning：arXiv:2205.13147
- Sparse Upcycling：arXiv:2212.05055
- RMoE：arXiv:2408.06793
- Huginn / Recurrent Depth：arXiv:2502.05171
- Upcycling Large Language Models into MoE：arXiv:2410.07524
- Nomic Embed v2：arXiv:2502.07972
- LARM：arXiv:2606.04678
- Tied Expert Layers：arXiv:2606.16825
- MELT：arXiv:2605.07721
- Sparse Layers / Looped-MoE：arXiv:2605.09165
- RecursiveGPT：arXiv:2608.26973
- Depth-adaptive Inference of Looped LMs（CDB）：arXiv:2608.09444
- Ouro / LoopLM：arXiv:2510.25741
- SMELT：arXiv:2609.01343
- Looped MoE Scaling Laws：arXiv:2609.40316
- WISE Working-Set Inference：arXiv:2609.27373
- FlashLoop：arXiv:2609.29812

## 12. 当前状态

- 本计划：**规划中，未启动 V6 训练**；
- V6.2（2026-10-09）：复核算力兑现文献后新增 §7，S6 改为样本级退出 + 物理吞吐验收，
  E/M/D 初版倾向 E4/M12/D4，对照改用 compute-matched 标准；
- Jaume 700M checkpoint：ModelScope 当前未公开；
- V5 HybridLoop：已完成 retrofit 与 adaptive-exit 机制验证，
  但算力未兑现（逻辑退出无物理节省、无 KV cache），该教训是 V6 §7 的直接动机；
- 下一步：先完成参数/显存/数据/FLOPs 账本与 S0 baseline，用户明确批准后再启动 S1。
