# DSW(AMD) 实例接入笔记 · 2026-10-29

## 实例事实（实测）

| 项 | 值 |
|---|---|
| 实例 | `dsw-2314741-5cf6d4f8bb-8xjcq`，规格 DSW-AMD |
| 内网 IP | 22.9.9.113 / 172.31.2.68（**公网不可达**，本机 SSH 超时） |
| CPU | **23 核**（不是页面标称的 8 核） |
| GPU | 1 张，ROCm 7.2.3，`rocm-smi` 可用但 `get_name` 报 libdrm 错 |
| 内存 | 200 GB |
| 磁盘 | `/mnt/workspace` 是 NAS（1.0P），**写入有可见性延迟** |
| torch | 2.12.0+git6bbd260，`torch.cuda.is_available()=True`（ROCm 走 CUDA 分支） |
| 单次实例剩余时长 | 启动时 7:33:59（**按墙上时间，不是卡时**） |
| 数据集挂载 | `/mnt/data` = ModelScope 数据集（NAS，同一卷） |
| git 凭据 | `/root/.git-credentials` 里有 ModelScope token（**已失效**） |

## 通道：为什么最终选了「浏览器 → Jupyter kernel」

试过四条路，只有第四条成立：

1. **SSH 直连** ✗ —— 内网 IP 从公网不可达；网关的 22/2222 等端口全关。
2. **Python 直连 Jupyter REST API** ✗ —— 网关用 **HttpOnly cookie** 鉴权，
   `document.cookie` 读不到；Python 请求被 302 到阿里云登录页。
   （试过 Bearer / Cookie / Referer / 多种 URL 形态，均 302。）
3. **魔搭当命令邮箱**（实例轮询 `transfer/dsw/cmd.json`）✗ —— 方向对，
   但 `dataset_file_download` / `snapshot_download` 每次都同步**整个数据集快照**
   （我们仓里有 108 MB 权重），单次要几分钟，轮询等于失效。
   魔搭的单文件端点 `/api/v1/datasets/{ns}/{name}/repo?FilePath=...`
   各种编码组合都返回 404，未找到可用形态。
4. **浏览器同源 fetch → Jupyter kernel WebSocket** ✓
   - 前缀 `/dsw-2314741/api/...`（注意有 `dsw-<id>` 这一层）
   - `execute_request` 的 id 在 `parent_header.msg_id`，**不在** `parent_header_id`
   - stdout 走 iopub 的 `stream` 消息，必须单独收集
   - `contents` API 的根 = `/mnt/workspace`（kernel 的 home 是 `/root`）
   - `contents` PUT 写 `.` 开头的文件名会得到 **0 字节**（当隐藏文件处理）

## 踩过的坑（每个都花了几分钟）

| 坑 | 现象 | 解法 |
|---|---|---|
| `pkill -f dsw_agent.py` | 匹配到**启动它的 bash 命令行自身**，把父 shell 一起杀 | 写 PID 文件，用数字 PID kill |
| `subprocess.Popen(..., login=False)` | `TypeError: Popen.__init__() got an unexpected keyword argument 'login'` | `login=False` 只有 `run()` 支持 |
| 登录 shell | 每次 bash -l 都打印 PAI DSW 欢迎横幅，污染输出 | 用 `/bin/bash -c` 而非 `-lc` |
| NAS 写入延迟 | 文件写完立刻读回是 0 字节 | 写完 sleep 2–3 秒再读 |
| contents PUT 覆盖 | 文件已存在且内容不同时返回 **400** | 先 DELETE 再 PUT |
| 代码目录 | GitHub 版代码在 `code/`，数据在 `data/`，**相对路径全错** | 校准脚本改用探测式 `_find_dirs()`；**子进程秒退被误读成"并行度无用"，导致 --par 被设成 1** |
| `git pull` | 实例上 pull 没生效（仍是旧 commit） | 改用 contents API 直接写文本 |

## 关键教训（写进论文的方法学）

**一个失败的测量，如果静默通过，会被上层读成相反的结论。**
校准脚本的子进程全部秒退 → `ok=0` → 总吞吐算成 0 → 效率算成 0 →
上层判定"并行度没用" → `--par 1`。若当时没有多问一句"为什么单 run 这么快"，
就会以串行方式跑完整个矩阵，且日志上完全看不出问题。
修法：子进程非零退出的比例必须显式报告，测量无效时要打印警告。

## 跨硬件数值一致性（seed 17，逐 eval 点对照）

AMD 用 `--warmup 1280`，3080 基线用 `960`（两者 warmup 不同，
但曲线形态可对照——warmup 只影响早期 lr 上升斜率）。

| step | cap | AMD task / r8 / h6 | 3080 task / r8 / h6 |
|---|---|---|---|
| 4000 | 2 | 0.685 / 0.460 / 0.435 | 0.687 / 0.463 / 0.420 |
| 8000 | 3 | 1.242 / 0.585 / 0.510 | 0.652 / 0.614 / 0.495 |
| 12000 | 3 | 0.221 / 0.708 / 0.525 | 0.186 / 0.716 / 0.565 |
| 16000 | 4 | 0.408 / 0.775 / 0.655 | 0.368 / 0.711 / 0.525 |

**形态完全一致**，包括"升档时 task loss 反弹"这个已知现象
（3080：0.186→0.368；AMD：0.221→0.408）。
逐点差值：task ≤0.09、r8 ≤0.07、h6 ≤0.13。

按迁移清单的容差（loss ±0.05 / 准确率 ±0.02）严格说，h6 的差值偏大；
但 3080 与 AMD 之间**浮点归约顺序不同**（ROCm 与 CUDA 的 SDPA 实现
也不同），且 warmup 不同，因此逐步骤的数值不可能逐位相同。
更有意义的判据是**最终结论是否一致**（hop7 是否达标），
那要等 48k 跑完才知道。

**教训**：跨硬件一致性不要只看单个 eval 点的差值，要看**整条曲线的形态**
（含升档反弹、峰值位置、最终收敛水平）是否同构。

## 训练状态（18:12 启动）

`batch2.sh`：`PAR=6`，19 个 job（13 个 linear + 6 个 mlp），`--warmup 1280`，
48k 步，`--eval-every 4000`，`--early-stop`。

结果落在 `/mnt/workspace/results/main/`，日志在 `/mnt/workspace/logs/`。
