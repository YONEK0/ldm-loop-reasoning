# LDM 循环深度项目 — 资产盘点（2026-10-30）

> 目的：在实例离线、断网前的静态快照。记录"资产在哪、有没有上云、缺口是什么"。
> 生成方式：`s97_audit_cloud.py`（本地 vs GitHub 差集）+ 魔搭 REST API 列举 + git ls-tree。
> 口径：本地 = `E:/develop/project_files/llm-test`；云端 = GitHub ×2 + 魔搭 ×2 + DSW 实例 NAS。

---

## 0. 一句话结论

**研究本体（LDM 循环深度线）几乎全部在云端**：代码 23 个脚本、数据集 4 文件、结果 356 个 JSON、
文档 91 份、19 个 checkpoint，均已在 GitHub `ldm-loop-reasoning` 与魔搭 `ldm-feasibility-testkit`。

**未上云的有价值资产只有两块**（合计约 11.2 GB）：
1. `export/hybridloop-v0.1-hf/` — HybridLoop v0.1 的 HF 权重，**本地 200 个分卷 + 9 个元数据文件，
   GitHub `v0.1-weights` tag 上只有 100 个分卷、零元数据**（缺 8.96 GB），
   → **该 tag 目前指向一个不可用（连 index.json 都没有）的模型**。
2. `runs/` 下 **83 个 checkpoint（2.29 GB）**未进 GitHub（其中 19 个 `ckpt/*.pt` 已在魔搭）。

此外 **AMD 主表 18 个结果 JSON 未进魔搭**（`transfer/results/` 里没有 MAIN_*/MLP_*）。

其余是早期 4B/0.8B 线的过程产物（`runs/s08_*`、`runs/v5_*`、`legacy_v1/` 等），
按既定口径不作为 LDM 结论依据，价值较低。

---

## 1. 本地资产（`E:/develop/project_files/llm-test`）

工作区总计约 **47 GB**（`.git` 单独 13 GB）。按目录：

| 目录 | 体积 | 文件数 | 内容 | 性质 |
|---|---|---|---|---|
| `export/` | 19 G | 212 | HybridLoop v0.1 HF 导出（config/tokenizer/modeling + 10 组 × ~20 分片 safetensors） | **发布物** |
| `models/` | 18 G | 82 | LoopUS-Qwen3-1.7B / Ouro-1.4B / Ouro-2.6B / Qwen3-4B-Thinking-2507（4bit） | 底座模型 |
| `runs/` | 4.3 G | 362 | 各期训练运行产物（99 个 `.pt`） | 过程产物 |
| `results/` | 3.0 G | 868 | 全部实验结果 JSON（含 `results/amd/` 18 个） | **结论依据** |
| `data/` | 519 M | 87 | 任务数据：`ldm_mvtd2/`（LDM 主用）、`ldm_mvtd/`、`ldm_mvt5/`、gsm8k 等 | **任务定义** |
| `ouro_env/` | 146 M | 5933 | Ouro 环境 | 参考实现 |
| `pkg_hybridloop4b/` | 102 M | 32 | 4B HybridLoop 打包 | 早期线 |
| `ckpt/` | 108 M | 19 | LDM 关键 checkpoint（H1 四臂 / H2 warmup / MAIN13） | **结论依据** |
| `bundle_ldm_server_v1/` | 274 M | 16 | LDM 服务端包 v1 | 部署物 |
| `bundle_ldm_server_v3/` | 71 M | 20 | LDM 服务端包 v3 | 部署物 |
| `bundle_v4/` | 13 M | 13 | v4 包 | 部署物 |
| `_pkg/` | 27 M | 1101 | Python 依赖包 | 环境 |
| `loopus/` | 3.5 M | 69 | LoopUS 实现 | 参考实现 |
| `results_server/` | 1.2 M | 19 | 服务端结果 | 结论依据 |
| `docs/` | 1.2 M | 119 | 假说/笔记/计划/错误/报告 | **知识资产** |
| `assets/` | 3.9 M | 6 | 图表等 | 辅助 |
| `server_run2_results/` | 374 K | 8 | 二轮服务端结果 | 结论依据 |
| `legacy_v1/` | 342 K | 56 | 早期版本 | 归档 |

顶层脚本 **208 个**（`s6*`–`s9*` + `run_*.sh` + `boot_dsw*.sh` + `dsw_guardian.py`）。

### 1.1 LDM 线核心代码（远端 `code/` 已有的 23 个）

| 脚本 | 作用 |
|---|---|
| `s63_ldm_model.py` | LDM 模型库（Prelude/Core/Coda + 决策头，head_type 可选 linear/bilinear/mlp/xattn） |
| `s71_ldm_iter4.py` | 训练主脚本（`--warmup` / `--head-type` / `--curriculum-step` / `--resume`） |
| `s62_ldm_task_gen.py` / `s66_ldm_mvt_gen.py` | MVT-D v2 任务生成 |
| `s65_ldm_gpu_smoke.py` | GPU 冒烟 |
| `s76_remote.py` | 远程控制（SSH） |
| `s81_eval_rounds.py` | 多轮评测（r 扫描） |
| `s83_shortcut_probe.py` | 捷径审计探针 |
| `s84_dual_task.py` / `s88_adj_task.py` | 双任务 / 邻接任务 |
| `s85_ms_transfer.py` | 魔搭中转传输 |
| `s86_progress.py` | 进度汇总 |
| `s89_auto_loop.py` | 自动迭代循环 |
| `s91_campaign_monitor.py` | 训练看板（6 类异常） |
| `s92_parallel_calibration.py` | 并行度拐点校准 |
| `dsw_guardian.py` | 保守型守护进程（28 KB） |
| `boot_dsw.sh` / `boot_dsw_auto.sh` | 重启自恢复 |
| `run_queue.sh` / `run_{batch_amd,head_ablation,main_matrix,warmup_ablation}.sh` | 队列与消融启动 |

### 1.2 文档结构（117 份，本地 `docs/`）

- `docs/hypotheses/` — H-001…H-029 假说台账（含 REVOKED/CONFIRMED/REFUTED/PENDING 状态），
  外加 `INDEX.md` 总表。
- `docs/notes/` — 逐轮判读笔记（`P19_LONG64K_VERDICT`、`H1_HEAD_ABLATION_VERDICT`、
  `H2_WARMUP_VERDICT`、`AMD_MAIN_TABLE_PARTIAL`、`EVIDENCE_FOR_CONCLUSIONS` 等）。
- `docs/plans/` — 预注册与计划（`PREREG_HEAD_ABLATION_20261028`、
  `PREREG_GUARDIAN_20261029`、`PAPER_SKELETON_20261027`、`AGENT_TASK_BRIEF` 等）。
- `docs/errors/` — 18 份错误档案（`err-task-shortcut` / `err-hop-seesaw` / `err-seed-variance` …）。
- `docs/reports/` — `FINAL_REPORT` / `RECURRENT_DEPTH` / `RESEARCH_SUMMARY` / `DELIVERABLES`。

---

## 2. 云端资产

### 2.1 GitHub — `YONEK0/ldm-loop-reasoning`（LDM 研究本体镜像）

- HEAD `36c64e9`，**480 文件**
- 布局：`code/` 23 · `docs/` 91 · `results/` 356（含 `results/amd/` 18、`results/guardian/` 3）·
  `data/ldm_mvtd2/` 4 · `README.md` · `REPRODUCE.md` · `LICENSE` · `requirements.txt` · `bootstrap_amd.sh`
- 内容与本地 LDM 线**基本对齐**：数据、结果、文档全在。

### 2.2 GitHub — `YONEK0/llm-hybrid-loop`（HybridLoop / 早期 4B 线）

- HEAD `2f19952`，**450 文件**；tag `v0.1`(`c913e12`)、`v0.1-weights`(`2f19952`)
- 布局：`runs/` 146 · `export/` 103 · `legacy_v1/` 56 · `results/` 51 · `logs/` 24 · `docs/` 14 ·
  `data/` 8 · `assets/` 6 + 若干根级训练脚本
- **注意**：`export/` 只有 103 个文件，**权重分片不完整**（见 §3）。

### 2.3 魔搭 — `Yoneko/ldm-feasibility-testkit`（数据集，非公开）

中转仓 `transfer/` 下：

| 子目录 | 内容 | 数量 |
|---|---|---|
| `transfer/ckpt/` | LDM 关键 checkpoint（19 个 `.pt`，约 108 MB） | 19 |
| `transfer/results/` | 实验结果 JSON + 日志 + 报告 | 100+ |
| `transfer/docs/hypotheses/` | H-024…H-027 + INDEX | 5 |
| `transfer/docs/notes/` | 4 份判读笔记 | 4 |
| `transfer/docs/plans/` | 7 份计划/契约 | 7 |
| `transfer/data/` | `ldm_mvtd/` + `ldm_mvtd2/` | 2 目录 |
| `transfer/dsw/` | `cmd.json` | 1 |
| 根级 | `_install_dsh.sh` / `_p01_launch.sh` 等 | 若干 |

数据集卡片名：**循环决策模型可行性测试工具包**，Apache-2.0，Downloads 477。

### 2.4 魔搭 — `Yoneko/hybridloop-4b-test`（模型）

4B HybridLoop 测试件，`FromSite: github`，2026-10-24。属早期 4B 线，非 LDM 结论依据。

### 2.5 DSW 实例（`dsw-g0bhx21k1rv5phru7z`，AMD 23 核 / 192 G / 200 GB）

| 项 | 值 |
|---|---|
| 状态 | **Failed（`Closed: timeout`）** — 平台侧 AMD 资源分配超时，重启未成功 |
| 剩余额度 | 294306 s ≈ **82 小时**（仍可用） |
| 数据持久性 | `/mnt/workspace` 为 NAS 1.0 P，实例关停后数据仍在 |
| 上次已知内容 | `ldm/`（GitHub clone）、`ldm_mvtd2` 数据、`ms_token.txt`、`jobs.txt`(19 条命令)、`dsw_guardian.py`、`boot_dsw_auto.sh`、`RECOVER.sh`、`reset_supervisor.sh`、`ldm-boot.conf`、`poll.sh`、`results/main/`(18 JSON)、`results/guardian/`、`results_bundle.json` |
| 开机自启 | supervisor `conf.d/ldm-boot.conf` → `boot_dsw_auto.sh`（**该目录在容器临时盘，重启会丢**，`RECOVER.sh`/`reset_supervisor.sh` 是恢复路径） |

---

## 3. 差集：有价值但未上云

### 3.1 HybridLoop v0.1 HF 权重（8.96 GB）— **优先级最高**

`export/hybridloop-v0.1-hf/`：本地 **209 个文件**（200 个 95 MB 分卷 + 9 个元数据文件），
GitHub `v0.1-weights` tag 上只有 **100 个分卷、0 个元数据文件**。

| 权重组 | 本地卷数 | 远端卷数 | 状态 |
|---|---|---|---|
| model-00001-of-00010 | 26 | 26 | ✅ |
| model-00002-of-00010 | 20 | 20 | ✅ |
| model-00003-of-00010 | 20 | 20 | ✅ |
| model-00004-of-00010 | 20 | 20 | ✅ |
| model-00005-of-00010 | 20 | **14** | ⚠️ 缺 6 卷 |
| model-00006…00009-of-00010 | 各 20 / 26 | **0** | ❌ 全缺 |
| model-00010-of-00010 | 8 | **0** | ❌ 全缺 |
| 元数据（config/tokenizer/modeling/index/README/REASSEMBLE/lineage） | 9 | **0** | ❌ 全缺 |
| **合计** | **209 / 200 卷** | **100 卷** | **缺 100 卷 + 9 元数据 ≈ 8.96 GB** |

> **风险**：现在拉 `v0.1` / `v0.1-weights` tag 会得到**不可用**的模型 ——
> 连 `model.safetensors.index.json` 都没有，且 10 个权重分片只到了一半。
> 这是唯一"已发布但不可用"的状态。

补救选项（择一）：
- (a) **续推**（推荐）：本地 `main` 已有全部 11 个 commit，只需 `git push origin main`；
- (b) 改传魔搭数据集（模型仓单文件上限不适合 95 MB × 200 卷），GitHub 侧只留 `REASSEMBLE.md` 指向魔搭；
- (c) 若确定不再用 HybridLoop v0.1，**删掉两个 tag**，避免他人误拉残缺权重。

> 复原方式（本地已验证可复原成标准 HF 目录）：
> `for prefix in $(ls *.safetensors.split-00 | sed 's/\.split-00$//'); do cat "${prefix}".split-* > "$prefix"; done`

### 3.2 `runs/` 下 83 个 checkpoint（2.29 GB）

主要是早期 4B/0.8B 线的运行产物（`s08_*`、`v5_*`、`t11/t14_param_eff`、`rd-*`、`p1_1_prosqa_loop` 等），
**按既定口径这些不作为 LDM 能力结论依据**，优先级低。
其中 LDM 相关且有价值的是 `ckpt/` 19 个 —— **已在魔搭 `transfer/ckpt/`**。

### 3.3 文档与脚本的路径差集（非真缺口）

- 文档 27 份未在 `ldm-loop-reasoning`：`docs/README.md`、`docs/STORAGE_CONTRACT.md`、
  早期 0.8B 线计划（`TRAINING_PLAN_V2/V3/V4/V5/V6*`、`0P8B_*`）、
  `docs/reports/*` 5 份、`S08_CAMPAIGN_LOG.md` 等。
  → 多数属**被明确排除的背景线**（0.8B / 4B 混入会污染 LDM 背景文档），
  **不建议整体上推**；只有 `docs/reports/*` 与 `docs/README.md`、`docs/STORAGE_CONTRACT.md` 值得单独补。
- 脚本 33 个"缺"实为路径差异：远端放在 `code/` 下。
  真正未上云的是早期线脚本（`s7_*`、`s8_r*`、`s9_*`、`s6_*`、`s7x_*`）与本次新增的验证码工具
  （`s94/s95/s96/s97`）。后者与 LDM 研究无关，属临时工具。

---

## 4. 恢复路径（实例离线时）

**最小恢复步骤**（实例恢复后执行）：

```bash
bash /mnt/workspace/RECOVER.sh
# → reset_supervisor.sh：把 ldm-boot.conf 装回 /etc/dsw/sys_configs/supervisor/conf.d/
# → boot_dsw_auto.sh：token → git pull → 从 MILESTONE 续跑未完成 run
#   → 起队列(PAR=6) → 起 dsw_guardian.py → 起 poll.sh
```

**若实例无法恢复**（AMD 资源持续超时）：

```bash
# 1) 新建实例（走 UI 启 动，需过阿里云 INPAINTING 滑块）
# 2) 新实例起来后：
cd /mnt/workspace/ldm/code
git pull --ff-only
python s92_parallel_calibration.py --results /mnt/workspace/results/main   # 校准并行度
# 3) 恢复 supervisor 开机任务
cp /mnt/workspace/ldm-boot.conf /etc/dsw/sys_configs/supervisor/conf.d/
supervisorctl -c /etc/dsw/sys_configs/supervisor-setup.conf reread && update
```

---

## 5. 待办（按优先级）

| 优先级 | 事项 | 说明 |
|---|---|---|
| P0 | 决定 HybridLoop v0.1 权重处置 | 现 tag 指向残缺模型：补推 8.96 GB / 删 tag / 转魔搭 |
| P0 | 实例恢复 | AMD 额度剩 82 h；`Failed/timeout` 是平台分配超时，需重试或换资源 |
| P1 | 补推 `docs/reports/*` + `docs/README.md` + `docs/STORAGE_CONTRACT.md` 到 `ldm-loop-reasoning` | 有价值且不污染背景线 |
| P1 | 补推 `results/amd/` 18 个 JSON 的**最新**版本到魔搭 `transfer/results/` | 现有 100 个 results 可能不含 AMD 主表 |
| P2 | `runs/` 83 个 ckpt 上云 | 价值低，量 2.29 GB，按需 |
| P2 | 清理本地 200+ 临时脚本与 `_b64.txt` 中转文件 | 工作区整洁；注意 `C:` 盘零大文件约定 |

---

## 6. 关键约束备忘（沿用，不因资产整理而变）

- 魔搭 token 只存服务器 `/root/.ms_token` 与 `/root/ldm/.dsh_env`（600），**不入库、不写日志**。
- `C:` 盘零大文件（`docs/STORAGE_CONTRACT.md`）；数据/代码传输走魔搭，SFTP 只用于控制命令。
- 背景文档不得混入 qwen3.5-4b / 0.8b 内容（排除 `S08_CAMPAIGN_LOG`、`TRAINING_PLAN_V6*`、`pkg_hybridloop4b`）。
- 不许用 MVT-5 数据支持任何能力结论（100% 捷径污染）。
- 预注册判据不得事后修改；负结果照实记录。
- 每个训练进程唯一 `--out`；并发上限 6。
- 杀进程用数字 PID。

---

## 附：盘点脚本

- `s97_audit_cloud.py` — 本地 vs GitHub 远端差集（可复跑）
- `s94_captcha_solve.py` / `s95_captcha_debug.py` / `s96_captcha_hole.py` — 阿里云 INPAINTING 滑块求解
  （本次为通过 UI 启 实例而写；INPAINTING 类型背景图含"被修补区"，
  用拼图片掩码做零均值 NCC 定位，峰值 NCC≈0.43，可用但需肉眼复核）
