# 预注册：DSW 训练守护进程（保守型）

> 2026-10-29 写定，**部署前定稿，跑起来后不改阈值**。
> 目的：在无人值守的长训练中自动发现"翻车"，做**最小必要的干预**，
> 遇事查文献并如实记录，不做无根据的干预。
> 实现：`code/dsw_guardian.py`；契约沿用 `docs/plans/AGENT_TASK_BRIEF.md` 第八节。

## 一、设计原则（保守优先）

1. **默认不动手。** 守护进程的第一职责是**观察与记录**，不是干预。
   任何自动动作都必须是「可逆」或「不丢数据」的。
2. **不碰产物。** 守护进程**只读** `results/**/*.json`，**绝不**删除、
   覆盖、修改任何 run 结果或 checkpoint。
3. **不改配方。** 守护进程**不修改**任何训练参数（lr / steps / 课程 /
   warmup / batch 等）。它只能：(a) 记录；(b) 停止一个已经坏掉的 run；
   (c) 从**已有 MILESTONE** 重启同一个 run 到**新的 `--out`**。
4. **宁少勿多。** 同一异常不重复动作；所有动作有冷却期。
5. **不猜。** 检测到异常时先找文献，把"已知的机制"写进事件记录，
   **不做因果推断**；因果判断留给有据可查的实验。

## 二、异常类型与阈值（跑前定稿）

| 代码 | 判据 | 严重度 | 自动动作 |
|---|---|---|---|
| `NAN_LOSS` | 任一 hist 点的 `loss`/`task` 为 NaN 或 ±inf | **致命** | 停止该 run（kill），记录，检索文献 |
| `LOSS_SPIKE` | 最近一个 loss > 该 run 历史中位数 × 10 且 > 1.0 | 致命 | 同上 |
| `COLLAPSE` | 某跳曾 ≥ 0.85，最近一次 eval < 0.60 | 警告 | **不动作**，只记录（可能是课程升档的暂时遗忘） |
| `SEESAW` | 同一 eval 内最高跳 − 最低跳 ≥ 0.40 且最低 < 0.60 | 警告 | 不动作，只记录 |
| `STALL` | 连续 5 次 eval，`dev_r8` 峰值无提升，且当前 step 已过 60% | 警告 | 不动作，只记录 |
| `FLAT_ALL_RANDOM` | 最近 eval 的 r1..r8 极差 < 0.02 且最大值 < 0.65 | 警告 | 不动作，只记录（可能是评测 bug，见 H-026） |
| `DEAD_RUN` | run 的进程不在了，但 JSON 未完成（step < target）且无 early_stop | 致命 | **从最近 MILESTONE 重启到新 `--out`**；无 MILESTONE 则只记录 |
| `QUEUE_STALL` | 队列活着但 30 分钟内没有新 run 启动 | 警告 | 不动作，只记录 |
| `DISK_PRESSURE` | `/mnt/workspace` 可用 < 20 GB | 致命 | **停止启动新 run**（不杀正在跑的） |
| `GPU_IDLE` | 连续 20 分钟没有任何训练进程 | 警告 | 不动作，只记录 |

### 明确**不**判为异常的（避免误报）

- 单跳准确率下降（课程升档时正常，见"升档遗忘"）
- task loss 在升档点反弹（历史基线同样形态）
- r1 偏低（Ouro 配方下 r1 本就在 0.4–0.87）
- 不同 seed 之间分数差异（已知方差 ±0.10）

## 三、自动动作的安全边界

| 允许 | 禁止 |
|---|---|
| kill 已被判定为 NaN/SPIKE 的**单个** run（按数字 PID） | kill 队列管理器、守护进程本身、或任何未触发致命判据的 run |
| 从 MILESTONE 重启到**新的 `--out`**（原档保留作对照） | 用同一个 `--out` 重跑（会覆盖原档） |
| 停止启动新 run（磁盘不足时） | 修改 `--steps`/`--lr`/`--warmup`/课程参数 |
| 删除自己的临时文件（bundle/tmp） | 删除任何 `results/` 下的文件 |
| 写 `results/guardian/` 下的新文件 | 修改 `results/` 下任何已有 JSON |

**PID 纪律**：一律用数字 PID（`kill -9 <pid>`），**禁止** `pkill -f`——
它会匹配到守护进程自己的命令行（10-27 曾因此误杀父 shell）。

**单次事件冷却**：同一 run 的同一异常类型，60 分钟内只动作一次。

## 四、文献检索（自动，无需 LLM）

触发致命异常时，守护进程自动查 arXiv（`export.arxiv.org/api/query`，
实例已验证可达，198 篇命中 "looped transformer"），按**异常类型**映射到
预置检索词：

| 异常 | 检索词 |
|---|---|
| `NAN_LOSS` | `all:"gradient explosion" AND all:"transformer"`, `all:"loss spike" AND all:"fine-tuning"` |
| `COLLAPSE` | `all:"catastrophic forgetting" AND all:"curriculum"`, `all:"continual learning" AND all:"task interference"` |
| `SEESAW` | `all:"interference" AND all:"multi-task" AND all:"representation"` |
| `STALL` | `all:"optimization stagnation" AND all:"deep networks"`, `all:"plateau" AND all:"learning rate schedule"` |
| `DEAD_RUN` | `all:"checkpoint" AND all:"resume" AND all:"training"` |

检索结果（标题 + arXiv ID + 摘要前 300 字 + 检索时间）写入
`results/guardian/literature/<event_id>.md`。

**记录纪律**：只写"检索到的论文说了什么"，**不写"因此本 run 的原因是 X"**。
因果结论必须有本项目的对照实验支撑。

## 五、记录格式（append-only，永不重写）

`results/guardian/events.jsonl` —— 每行一个 JSON：

```json
{"ts": 1793300000.0, "event": "NAN_LOSS", "run": "MAIN_w1280_s17",
 "severity": "fatal", "evidence": {...}, "action": "killed",
 "action_detail": {"pid": 12345}, "lit_file": "literature/20261029T1530_NAN_LOSS_s17.md",
 "note": "observed only; no causal claim"}
```

`results/guardian/state.json` —— 每个 run 的滚动状态（供跨轮比较）。
`results/guardian/report.md` —— 人读看板，每轮覆盖写（这是唯一被覆盖的文件）。

## 六、判定标准（不因结果改判）

- **守护进程本身失败**（崩溃、误杀）比漏报更严重。因此所有 kill 动作都要求
  **两条独立证据**（如 NaN + loss spike），单一证据只记录不动作。
- **误报成本**：警告级误报只浪费人工一次查看；致命级误报会毁掉一个 run。
  因此致命级一律要求双证据。
- **不动作的合理性也要被检验**：若守护进程连续 3 轮都只记录不动作，
  且期间确实没有异常，说明它工作正常（不是失灵）。这一点会写进看板。

## 七、与 dsh 智能体的分工

- **dsw_guardian.py（确定性）**：负责检测、记录、以及**仅**两类安全动作
  （停坏 run、从 MILESTONE 重启）。**不需要 LLM，不会幻觉。**
- **dsh 智能体（可选）**：在守护进程**记录了异常之后**，
  按 `AGENT_TASK_BRIEF.md` 第八节契约工作——先判读 → 行不通查文献 →
  提假设（带预注册判据）→ 才启动实验 → 单变量对照 → 归档。
  **智能体不得绕过守护进程的阈值自行 kill 训练**；
  需要干预时，它写"建议 + 预注册判据"到 `results/guardian/proposals/`，
  由守护进程校验判据存在后才执行。

## 八、部署前验证（必须做）

在正式守护生产训练前，注入三种故障做演练，确认：

1. kill 一个训练进程 → 守护进程在 5 分钟内检测到 `DEAD_RUN`，
   从 MILESTONE 重启到新 `--out`，原档完好；
2. 往某个 run 的 hist 里注入 NaN → 守护进程检测到 `NAN_LOSS` 并停止该 run，
   **其余 run 不受影响**；
3. 制造磁盘不足（模拟）→ 守护进程停止启动新 run，但不杀正在跑的。

演练记录写入 `docs/notes/GUARDIAN_DRILL_20261029.md`。
