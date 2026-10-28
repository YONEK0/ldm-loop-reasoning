# 算力兑现线执行契约与 AI 执行计划

> 文档：`COMPUTE_REALIZATION_EXECUTION_CONTRACT_V1.md`
> 版本：V1
> 日期：2026-10-18
> 项目：Qwen3.5-0.8B HybridLoop
> 执行模式：本文件交给后续 AI 执行；执行前必须读取本文件和契约索引。
> 当前状态：**未启动本线测试或训练**。

---

## 0. 任务目标

把已经验证的“自适应深度可降低循环次数”兑现为可复现、可审计的实际推理算力和性能数据。

本线必须区分三个结论：

1. **机制节省**：路由器实际分配的平均深度下降。
2. **执行节省**：真实执行的 M 层前向数下降。
3. **系统节省**：在正确缓存、正确 GDN 单步语义和目标硬件上，实际 FLOPs、ms/token、整题时间或吞吐下降。

只有第 3 项通过墙钟和正确性门禁后，才允许使用“实际加速/实际省算力”表述。层数推导、teacher-forced 漂移和理论比例不能替代墙钟结果。

本线的默认顺序是：

```text
E0 资产冻结与环境门禁
  -> E1 GDN 单步一致性 + exact 增量 cache
  -> E2 序列级自适应深度真实生成基准
  -> E3 cache/内存工程优化
  -> E4 token 级 exit-skip（条件分支）
  -> E5 报告、归档、论文表格
```

---

# 第一部分：不可变契约

## C-01 模型与权重契约

- 所有主实验使用 `Qwen3.5-0.8B-Base`，不得替换成 4B、7B、2B 或其他模型。
- 循环模型结构固定为：
  - `E=[0,4)`：4 层
  - `M=[4,16)`：12 层
  - `D=[16,24)`：8 层
  - M 包含物理层 7、11、15 的全注意力层。
- M 权重共享；逻辑循环深度最大 `B=8`。
- Q1 质量追平后的 router checkpoint：
  - `runs/s08_qat_exit/ckpt_latest.pt`
- 循环基线 arm B checkpoint：
  - `runs/s08_armB/ckpt_latest.pt`
- 非循环基线 arm A checkpoint：
  - `runs/s08_armA/ckpt_latest.pt`
- GDN 单步一致性补丁参考：
  - `s34_gdn_step_fix.py`
- 任何权重、LoRA、门、router、tokenizer 或配置变更都必须新建实验 ID，不得覆盖上述资产。

## C-02 数据契约

- 生成质量主评测使用既有 validation 集：
  - `data/gsm8k/s08/validation.jsonl`
- 数据切分清单：
  - `data/gsm8k/s08/manifest.json`
- GSM8K test 不得用于阈值选择、模式选择、早停或调参。
- 生成评测固定：
  - greedy decoding
  - 协议 v1.1：`scorer_strict.py`
  - 同一题目、同一 prompt 模板、同一最大输出长度
- 正式对外结果必须记录题目数量、题目 ID、种子、prompt 模板、最大输出长度和 scorer 版本。

## C-03 正确性契约

任何性能测试前必须通过正确性门禁：

- 单遍 E/M/D 与原始模型路径的全位置 logits 对齐。
- exact 增量 cache 与全前缀参考路径的 teacher-forced 对齐。
- 覆盖：
  - prefill
  - 第一个 decode token
  - 后续至少 8 个 decode token
  - GDN conv state
  - GDN recurrent state
  - 全注意力 KV
  - rotary 绝对位置
  - 清空 cache 后重新请求
- 记录：
  - max/mean logits diff
  - CE diff
  - argmax flip count
  - top-k overlap
- 数值容差必须按 dtype/backend 预先记录。不能把未经验证的 `bf16 maxdiff < 1e-3` 当作硬规则。
- 正确性未通过：立即停止性能测试，保留日志和 checkpoint，创建问题记录。

## C-04 GDN 单步实现契约

Qwen3.5 GDN 的单步增量路径必须使用与 chunk 路径一致的 delta-rule 语义：

- 单步有缓存时使用 `chunk_gated_delta_rule`，并传入对应深度/物理层的 `initial_state`。
- conv state 使用官方 update 逻辑。
- recurrent state 按 `(logical_depth, physical_layer)` 独立保存和更新。
- 不得直接把官方 `recurrent_gated_delta_rule` 当作与 chunk 路径数学等价。
- 不得仅通过复制 hidden 或广播 KV 修复 GDN 历史状态。
- `s34_gdn_step_fix.py` 是已验证的修复参考，但正式实验应记录补丁版本、代码 hash 和 Transformers 版本。

已知证据：未修复时首个增量 token 在 E0 层产生约 0.19 的真实差异，经过 arm B 高增益循环后放大到 logits 级 25；fp32 复测仍存在，因此不是普通 bf16 噪声。

## C-05 计数契约

每个实验同时记录四种计数：

1. `physical_layer_calls`：实际调用的 decoder layer 次数。
2. `M_layer_calls`：实际 M 层调用次数。
3. `kv_backfill_projection_calls`：为了补 KV 额外执行的投影/归一化/RoPE 次数。
4. `total_estimated_layer_equivalent`：明确公式的代理值，不得冒充 FLOPs。

推荐代理公式：

```text
base single-pass per generated token = E(4) + M(12) + D(8) = 24 layer calls
fixed d per generated token          = 12 + 12*d layer calls
d=1                                 = 24 layer calls
d=2                                 = 36 layer calls
d=8                                 = 108 layer calls
sequence-adaptive d*                 = 12 + 12*d*
```

注意：该公式只表示 decoder layer call 代理，不等于 FLOPs。GDN、全注意力、FFN、KV 读写、补投影和调度成本必须另记。

## C-06 墙钟测量契约

墙钟结果必须：

- 使用 CUDA event 或等价设备同步计时；
- warm-up 后测量；
- 至少 3 次重复，报告均值、中位数、P50/P95；
- 分开报告：
  - TTFT/prefill latency
  - decode ms/token
  - 完整请求时间
  - tokens/s
  - 输出 token 数
  - 峰值显存
- 固定 batch、prompt 长度、输出上限、dtype、硬件和软件版本。
- 游戏或其他 GPU 任务运行时不得采集正式墙钟数据；若检测到外部 GPU 负载，标记 invalid 并暂停正式测量。
- Python 循环、日志写盘、进程启动和模型加载时间不能混入 steady-state decode，除非明确标为端到端时间。

## C-07 质量契约

真实加速必须同时满足：

- strict 正确数不低于固定 d2 基线的预设容差；
- 完成率不低于固定 d2 基线；
- 重复率、格式崩坏率不出现系统性恶化；
- teacher-forced CE 漂移在执行前锁定的阈值内；
- 质量差异与算力收益必须同时报告，不能只报节省比例。

当前已知基线：

- arm A strict：21/40，完成 36/40；
- arm B fixed-d2 strict：22/40，完成 37/40；
- 序列级自适应 d*=2/3：21/40，完成 36/40；
- 历史 `results/s33_s08_seq_adaptive_gen.json` 及台账中的 61.7% 只能标记为 `legacy_pending_revalidation`：旧脚本的层数公式曾使用错误的 `20+12d`，且旧计数没有 hook 级真实计数、未把 prefill/decode 分开。
- 新执行线不得直接复述 61.7% 为已通过结果；必须用 E1/E2 的新计数和统一 prefill/decode 口径重新计算。

## C-08 阶段边界契约

- E1 exact cache 正确性通过前，不得进入 E2 正式性能。
- E2 序列级自适应通过前，不得声称 token 级跳过已可用。
- E4 token 级 exit-skip 是条件分支，不得因 E2 成功而自动启动训练。
- 任何新增训练（部署语义训练、Q2、LoRA、蒸馏）都需要单独 checkpoint、token budget、看板和评估报告。

## C-09 存档契约

- 每 25 步或每个可恢复工作单元保存 `latest/prev`。
- 每个转折点永久保存，并记录：
  - 触发指标
  - 阈值
  - 实验配置
  - 输入数据 hash
  - 代码版本/hash
  - 评估结果
- 结果 JSON 必须逐题落盘，不能等全部实验结束才第一次写文件。
- 日志必须 append-only；恢复运行允许重做区间，但不得静默覆盖旧记录。
- 现有总台账：
  - `docs/plans/S08_CAMPAIGN_LOG.md`

## C-11 旧结果与语义阻断项

- 旧序列自适应脚本的 prefill 可能使用 soft/全圈路径，而 decode 使用固定 d*；未完成同 d* 的完整 prefill，因此不能直接称为 exact incremental。
- 旧 GDN 补丁同时涉及历史状态初始化和单步 delta 内核，正式执行必须做 A/B：占位 seed、补丁开关、完整 E/M/D prefill、至少多步 logits；不得把单次单步 diff 当作充分证明。
- `s31_s08_hard_exit_gate.py` 的 mask 是 teacher-forced 冻结近似，不是真实 skip；其节省值只作为前置可行性参考。
- token 级 skip 的物理层映射必须使用 `physical_layer - M_start`，即本模型为 `physical_layer - 4`；任何 `-8` 映射都阻断执行。
- 首个生成 token 必须来自 prefill 末位 logits；不得重复喂入 prompt 最后 token。
- 版本必须固定为 `loopus_env` 的 Transformers 5.3.0；若解释器或 cache API 不匹配，E0 阻断。

## C-10 停止契约

立即停止并保留现场的条件：
- 输出出现统一乱码、重复 token、整批格式崩坏；
- 进程异常退出、OOM、GPU 外部负载进入正式测量；
- checkpoint 无法恢复；
- 质量下降超过预设阈值；
- 发现实际实现与本契约语义不一致。

停止后先产出问题报告，不得盲目重试或自动跳过门禁。

---

# 第二部分：执行计划

## E0：资产冻结与环境门禁

### 输入

- 本文件；
- `s34_gdn_step_fix.py`；
- `s24_s08_cache_check.py`；
- `runs/s08_armA/ckpt_latest.pt`；
- `runs/s08_armB/ckpt_latest.pt`；
- `runs/s08_qat_exit/ckpt_latest.pt`；
- `results/s31_s08_hard_exit_gate.json`；
- `results/s33_s08_seq_adaptive_gen.json`；
- `S08_CAMPAIGN_LOG.md` §1–§15。

### 步骤

1. 确认 GPU 型号、显存、CUDA、PyTorch、Transformers 版本。
2. 确认无游戏/外部 GPU 进程。
3. 确认上述 checkpoint 可加载。
4. 记录代码 hash、配置 hash、环境版本。
5. 不改权重，不启动训练。

### 产物

- `results/e0_compute_env.json`
- `results/e0_asset_manifest.json`

### 通过条件

所有资产可读取、无外部 GPU 负载、版本固定。

## E1：exact 增量 cache 正确性与基线墙钟

### 模式

1. `full_recompute_soft`：全前缀重算，soft/Q1 语义。
2. `exact_incremental_d2`：固定 d2，缓存增量，GDN 使用 s34 单步补丁。
3. `exact_incremental_d8`：固定 d8，缓存增量，作为高深度基线。
4. `sequence_adaptive_d2_or_d3`：prefill 选 d*，生成段固定 d*，缓存增量。

### 测试矩阵

- prompt 长度：128、512、1024 token（硬件允许时）；
- 生成上限：32、128、256 token；
- batch：1、2（资源允许时）；
- 每格至少 3 次重复；
- 正确性先用 3 题/短上限，再扩大。

### 产物

- `results/e1_cache_correctness.json`
- `results/e1_latency_baseline.json`
- `runs/e1_cache.log`

### 通过条件

exact 增量与参考路径通过 C-03；否则停止，不进入 E2。

## E2：序列级自适应真实算力兑现

### 语义

- prompt prefill 后聚合 router 决策得到 `d*`；
- 整个生成请求固定使用 d*，不在生成中途让 token 缺席 GDN 深度；
- d* 只允许使用已校准的 `{2,3}`，并记录每题选择；
- 这是当前最稳的真实加速模式，不是 token 级稀疏。

### 对照

- fixed d8 exact incremental；
- fixed d2 exact incremental；
- sequence adaptive exact incremental；
- full recompute 仅作正确性/失真参考，不作主性能基线。

### 必报指标

- strict、完成率、重复率；
- 平均 d* 和直方图；
- 实际 M layer calls；
- prefill 与 decode 分开 ms/token；
- 端到端每题时间；
- 相对 fixed d8 的节省；
- 相对 arm A 的额外成本，禁止只报“省 61.7%”而不报参照系。

### 通过条件

质量不低于 fixed d2 容差，且真实生成段 `M_layer_calls` 和 steady-state decode 时间低于 fixed d8。

## E3：缓存工程优化

只在 E1/E2 通过后执行：

1. 对 KV/状态做预分配；
2. 使用分块写入，避免每 token `torch.cat`；
3. 减少 Python 层日志与模块调度；
4. profiler 采样 kernel、CPU 调度、显存带宽；
5. 每项优化单独 A/B，不能一次叠加后无法归因。

每项必须重跑 C-03 与 C-06。

## E4：token 级 exit-skip（条件分支）

当前 token 级真跳过已失败一次，不能直接复用旧实现。只有在明确选择以下一种路径后才能继续：

- E4-A：部署语义训练：训练期直接采用单步增量 GDN 语义；
- E4-B：自写严格一致 GDN 单步 kernel/参考实现，并证明与 chunk 路径等价；
- E4-C：保留序列级自适应，不做 token 级稀疏。

若执行 E4-A/B，必须新增训练/代码版本、独立 checkpoint、G2' 门禁和质量预算。不得把冻结 GDN 状态、KV 补条目或 hidden 广播当作已验证 exact 语义。

## E5：报告与归档

### 结果表至少包含

| 模式 | 质量 | 平均深度 | 层前向/token | TTFT | decode ms/token | 完整请求 ms | 显存 | 参照系 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full recompute | | | | | | | | |
| fixed d2 cache | | | | | | | | |
| fixed d8 cache | | | | | | | | |
| seq adaptive cache | | | | | | | | |

### 结论标签必须分开

- `quality_parity_passed`
- `exact_cache_correctness_passed`
- `sequence_adaptive_layer_saving_passed`
- `wall_clock_speedup_passed`
- `token_level_skip_passed`
- `npu_port_passed`

任何一个标签不能替代其他标签。

### 论文素材

追加至：

- `docs/plans/S08_CAMPAIGN_LOG.md`

并保存：

- 原始 JSON；
- 环境 manifest；
- 逐题生成结果；
- profiler 摘要；
- checkpoint/代码 hash；
- 失败实验和停止原因。

---

# 第三部分：给执行 AI 的固定操作规则

1. 开始前先读本文件、`TRAINING_PLAN_V7_0P8B.md` V5、`S08_CAMPAIGN_LOG.md` §13–§15。
2. 先执行 E0 和 E1，不能跳过正确性门。
3. 每个阶段先做小规模 smoke，再扩大矩阵。
4. 任何异常先暂停、保留现场、写问题报告；不得隐瞒或只重试到成功。
5. 任何墙钟数据若检测到游戏或其他 GPU 负载，标记 invalid，不纳入正式结果。
6. 不修改历史结果 JSON；新实验使用新文件名和新 run 目录。
7. 训练、部署语义适配和新增权重操作必须有明确授权；本文件本身不等于无限训练授权。
8. 结束时先汇报事实，再汇报是否通过各个契约标签，最后给出下一阶段建议。

## 契约索引

- `C-01` 模型与权重
- `C-02` 数据隔离
- `C-03` 正确性
- `C-04` GDN 单步一致性
- `C-05` 算力计数
- `C-06` 墙钟测量
- `C-07` 质量
- `C-08` 阶段边界
- `C-09` 存档
- `C-10` 停止
- `E0` 资产/环境
- `E1` exact cache
- `E2` 序列级自适应
- `E3` cache 工程
- `E4` token 级跳过条件分支
- `E5` 报告归档

## C-04 澄清（2026-10-18，E1 实测后追加）

E1 受控 A/B（seed 适配正确施加于两条路径）表明：GDN 单步增量
（recurrent 内核）与分块重算（chunk 内核）的差异**真实但有界**
（logits ≤0.39、state 0.25，bf16 运算次序级），不会自发放大。
`s34` 补丁是合法的替代适配器（直接读 `recurrent_states[layer_idx]`，
绕过 `has_previous_state` 坏分支），但**不是唯一解**：prefill 后为
E/M 分区 cache 的 `conv_states[22]` 补占位同样可使官方路径正确。
分区 cache 部署的必检项是 has_previous_state 适配，而非内核选择。
