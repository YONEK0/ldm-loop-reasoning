# 错误索引（ERRORS INDEX）

> **用途**：LDM / 循环深度研究线的所有已确认错误、失败方向、错误假设，按"症状 → 原因 → 修法"归档。
> **路由方式**：先读本索引的【快速路由表】，匹配症状后进入对应的 `err-*.md` 详情文件。
> **纪律**：每次遇到问题先查索引；确认是**新问题**才新建 `err-<slug>.md` 并在下表登记。
> **配套**：假说记录见 `../hypotheses/INDEX.md`（三态：待验证/证实/证伪）。

---

## 快速路由表（按症状查）

| 症状（你可能看到的） | 错误编号 | 详情文件 | 状态 |
|---|---|---|---|
| 训练 loss 正常下降，但某个 hop 的精度反复从 ~0.9 崩到 ~0.0，相邻 hop 此消彼长 | E-001 | [err-hop-seesaw.md](err-hop-seesaw.md) | **已解决**（persist-sup，见 H-011） |
| 同一配方换 seed，结果从"全解"变成"只解一半" | E-002 | [err-seed-variance.md](err-seed-variance.md) | **已定位为 E-001 的表现** |
| 训练后期精度突然下滑（20k 峰值 0.916 → 24k 0.786） | E-003 | [err-late-forgetting.md](err-late-forgetting.md) | 已定位，已有缓解（早停/里程碑） |
| `rounds_mask` 冻结后退出控制器 q 恒为 1.00（永不停止） | E-004 | [err-exit-identity.md](err-exit-identity.md) | **已确证并修复**（PonderNet 损失） |
| 退出头训练 loss 下降但硬退出全跑满 8 轮 | E-005 | [err-train-infer-mismatch.md](err-train-infer-mismatch.md) | **已确证并修复** |
| 深度外推到 hop7/8 全部 0.5（随机） | E-006 | [err-extrapolation.md](err-extrapolation.md) | 已定位（轮次基准偏移），未修复 |
| 同一 `--out` 两条进程互相覆盖，产生不可信读数 | E-007 | [err-file-race.md](err-file-race.md) | **已修复**（文件锁） |
| 训练进程"查不到"就重启，实际还活着，重复启动 | E-008 | [err-orphan-process.md](err-orphan-process.md) | **已修复**（判定方法修正） |
| 本地 nohup / start /b 后进程消失 | E-009 | [err-local-bg.md](err-local-bg.md) | **已绕过**（改服务器 or 分段） |
| 零初始化导致梯度为 0，循环块学不动 | E-010 | [err-zero-init.md](err-zero-init.md) | **已修复**（small_out std=0.02） |
| 单步捷径：不循环也能答对（68.9% 一跳可区分） | E-011 | [err-single-step-shortcut.md](err-single-step-shortcut.md) | **已修复**（MVT-2+ 非函数式图） |
| 存活头统计捷径：靠断链深度猜答案 | E-012 | [err-surv-shortcut.md](err-surv-shortcut.md) | **已修复**（断链深度均匀化） |
| **任务级捷径：一条不看输入的规则满分，模型高分≠学会能力（准确率违反物理约束）** | **E-015** | [err-task-shortcut.md](err-task-shortcut.md) | **已确证**（MVT-5 下游链长，H-023）；已建检测流程 |
| exact-T 分组批导致记忆（train loss→0.0001，hop3 塌到 0.45） | E-013 | [err-grouped-batch.md](err-grouped-batch.md) | **已修复**（混批 + per-sample R） |
| 梯度范数为 0 / 模型不更新（torch 2.x 特定） | E-014 | [err-grad-zero-torch.md](err-grad-zero-torch.md) | 待验证（服务器 torch 2.5.1 环境） |

---

## 已确认的失败方向（不要重复尝试）

| 方向 | 结论 | 证据 |
|---|---|---|
| 降低学习率（1e-3 → 5e-4 / 3e-4） | **无效**，三档都失败，3e-4 更差（0.176） | `results/iter4_v8_s53_lr*.json` |
| 权重 EMA（decay 0.99） | **部分有效但不可靠**：seed53 0.49→0.86，seed17 从全解掉到 0.691 | `results/iter4_v9_*_ema99.json` |
| 残差膨胀假说（residual explosion） | **证伪**：成功/失败残差增长 6.78× vs 6.63×，无区分度 | `results/s74_residual.json` |
| 加大训练步数（24k → 32k） | **无效**，seed53 32k 反而更差（0.493） | `results/iter4_v7_exactT_s53.json` |
| 用 PonderNet 损失解 hop6 | **失败**，hop6 只有 0.47–0.49（exact-T 能到 0.96） | `results/iter4_v5b_ponder_*.json` |

---

## 记录规范

新建 `err-<slug>.md` 必须包含：

1. **症状**（可操作：你会看到什么数字/报错）
2. **复现方式**（命令 + 参数）
3. **根因**（有证据指向的机制，不是猜测；猜测要标"未证实"）
4. **修法**（已验证有效的写"已验证"；未验证的写"候选"）
5. **相关假说编号**（指向 `../hypotheses/`）
6. **触发文件**（代码位置或脚本名）

新增或改状态后，**同步更新本索引的路由表**。
