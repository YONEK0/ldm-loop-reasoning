# 交接：自动迭代编排器 + dsh/pi 基础设施调研

> 2026-10-27 23:20。服务器 region-41（RTX 3080 10GB）。

## 一、已在运行：自动"测试→判读→优化"迭代链

**`s89_auto_loop.py`**（无外部智能体依赖，也可接 LLM）。已实际做出自主决策：

```
23:17:23  P05 阶段完成
          summary: h7=0.789 h6=0.855 pass_rate=0.25 (1/4)
          → 决策：launch P12_adj seeds=[71,17,53] (--adj-mask)
          → 阶段推进到 B_adjacency_mask
```

### 架构

```
spec.json → [scan] → [evaluate] → [decide] → [act] → [record]
                ↑                                    │
                └────────── loop (180s) ─────────────┘
```

| 组件 | 作用 |
|---|---|
| `docs/plans/auto_loop_spec.json` | 阶段定义：glob、目标步数、指标映射、通过条件、决策规则 |
| `results/auto_loop.out` | 常驻日志（stdout） |
| `results/auto_loop_log.jsonl` | 每次决策的结构化记录 |
| `results/auto_loop_report.md` | 人读报告（覆盖式，最新一次判读） |
| `results/auto_loop_state.json` | 阶段进度 + 已启动清单 |

### 扩展方式（改判据不用改代码）

```json
{"name": "阶段名", "glob": "P05_*", "expect": 4, "target_step": 48000,
 "metric": {"h7": "confirm_r6_by_hop.7", "h6": "dev_r8_by_hop.6"},
 "pass_if": "h7 >= 0.80 and h7 >= h6 - 0.05",
 "on_done": [{"if": "pass_rate >= 0.25",
              "then": {"type": "launch", "tag": "P12_adj",
                       "seeds": [71,17,53], "extra": "--adj-mask",
                       "next": "B_adjacency_mask"}}]}
```

动作类型：`launch`（起训练，可带 `next` 推进阶段）、`stage`（只推进）、
`stop`（终止并写报告）。

### 已知缺陷（待修）

1. **重复记账**：同一 tick 内若 `.json` 尚未生成会重复 launch。
   实际被训练脚本的文件锁挡住（不会双跑），但 state 里会有重复条目。
   修法：launch 前写一个 `.launched` 标记文件。
2. **launch 进程可被误杀**：`ps | grep P06 | xargs kill` 这类清理会误伤，
   需用数字 PID 精确匹配。

---

## 二、dsh / pi 基础设施调研结论

### 可用的真包

| 包 | 版本 | 说明 |
|---|---|---|
| **`dsh-tui`** | 0.2.19 | **DeepSeek Harness 终端客户端**。`dsh-tui run "<prompt>" --json` 可无头一次性调用；默认连 `DSH_URL`（默认 http://127.0.0.1:3080） |
| **`dsh-host`** | 0.1.0 | **DeepSeek Harness 无头持久后端**（SDK，无 CLI）。README：需 `pnpm build` 后用 `dsh plugin --profile host add <path>`，再 `dsh --profile host` 启动 |

### 不可用的名字（重要）

| 包 | 实际情况 |
|---|---|
| `dsh` | npm 上是 "A shell written in JavaScript"（无关项目）→ **`dsh` 这个 CLI 名字被占用** |
| `pi` | "Going deeper inside of the PI number"（玩笑包） |
| `deepseek-harness` | 0.0.1 占位包，官方写明"开发完成后才发布" |

**结论**：`dsh-tui`（客户端）与 `dsh-host`（后端 SDK）都是真的，
但把 host 装成可用服务所需的 **`dsh` harness CLI（plugin/profile 管理器）
在 npm 上不存在**（名字被占）。因此 dsh 链路目前只能到
"客户端已装、host 未运行"。

### 服务器已装

```
node v22.20.0  npm 10.9.3      (/usr/local/node22, 软链到 /usr/local/bin)
deepseek-cli 1.0.2  → bin: deepseek / ds   （需 `deepseek config` 交互配置，未配）
dsh-tui 0.2.19      → bin: dsh-tui（host 未起，`run` 报 fetch failed）
dsh-host 0.1.0      （SDK，未部署为服务）
```

### 要打通 dsh 需要什么（按优先级）

1. **拿到 `dsh` harness CLI**（plugin/profile 管理器）——
   若你本地或私有源有，复制到服务器即可：
   `npm i -g <your-dsh-cli>` 或直接拷贝 `/root/dsh-test` 整个目录。
2. 或有 **dsh-host 的构建产物 + 启动脚本**（`pnpm build` 后的 lib/）。
3. 拿到后：`dsh plugin --profile host add /root/dsh-host` →
   `dsh --profile host`（detached）→ 设置 `DEEPSEEK_API_KEY` →
   `dsh-tui run "..." --json` 验证。

### 备选：deepseek-cli

已装，但配置是交互式（`deepseek config`）。若要走这条，
我可以反查它的配置文件路径直接写入 API key（本地已有 `DEEPSEEK_API_KEY`）。

---

## 三、本轮实验状态

### P05 完成：48k × 7 seed

| seed | h6 | h7 | h8 | 判据 |
|---|---|---|---|---|
| s71 | 0.900 | **0.915** | 0.880 | PASS |
| s11 | 0.860 | **0.840** | 0.815 | PASS |
| s29 | 1.000 | 0.805 | 0.815 | fail（h7 < h6−0.05） |
| s13 | 0.780 | 0.765 | 0.670 | fail |
| s19 | 0.780 | 0.745 | 0.670 | fail |
| s17 | 0.725 | 0.740 | 0.710 | fail |
| s53 | 0.720 | 0.695 | 0.670 | fail |

- `h7 ≥ 0.80`：**3/7（43%）**
- 严格判据（h7≥0.80 且 h7≥h6−0.05）：**2/7（29%）**

### 正在跑：P12_adj（邻接掩码，H-025）

3 个 seed（71/17/53），48k，`--adj-mask`。用于检验
"每轮只多传播一跳"时是否出现真正的阶梯。

---

## 四、交接建议

1. **短期**：让 `s89_auto_loop.py` 继续跑（它会等 P12_adj 完成后自动判读并 stop）。
2. **中期**：把 `dsh` harness CLI 放进服务器，用
   `dsh-tui run --json` 替换 `s89_auto_loop.py` 的规则决策
   （接口只需实现 `decide(summary) -> action`，约 20 行）。
3. **prompt 模板**已可复用 `s78_agent_loop.build_prompt` 的结构（读背景 +
   读摘要 + 输出 JSON 动作）。
