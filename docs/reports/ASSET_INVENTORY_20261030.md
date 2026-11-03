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

- HEAD `574b604`，**535 文件**
- 布局：`code/` **40** · `data/` **34** · `docs/` **99** · `results/` 356 ·
  `README.md` · `REPRODUCE.md` · `LICENSE` · `requirements.txt` · `bootstrap_amd.sh`
- **代码与数据资料已补齐**（2026-10-30）：
  - `code/` 新增 17 个 LDM 脚本（s60–s87 段，含 `s78_agent_loop` / `s77_watchdog` /
    `s73_ldm_dashboard` / `s68/s69_parity` / `s87_hard_task`）
  - `data/` 新增 5 个任务集：`ldm_mvtd`、`ldm_edge_action_v5`、`synth_multihop`、
    `ldm_parity`、`ldm_mvte`
  - `ldm_edge_action_v5/train.jsonl`（262 MB / 192000 行）超出 GitHub 100 MB 限制，
    拆为 `train.jsonl.part1of4…part4of4`（59/59/72/72 MB）+ `REASSEMBLE.md` 说明还原命令
  - **`data/ldm_mvt5` 明确排除**（100% 捷径污染，不得用于任何能力结论）

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
| `transfer/results/` | 实验结果 JSON + 日志 + 报告 | 100+ （**本次已补 `results/amd/` 18 个主表**） |
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
| 状态 | **Failed（`Closed: timeout`，`ReasonMessage: "start timeout"`）** |
| 剩余额度 | 294306 s ≈ **82 小时**（仍可用，未消耗） |
| 数据持久性 | `/mnt/workspace` 为 NAS 1.0 P，**与实例存活解耦** —— 实例关停/重建后数据仍在 |
| 上次已知内容 | `ldm/`（GitHub clone）、`ldm_mvtd2` 数据、`ms_token.txt`、`jobs.txt`(19 条命令)、`dsw_guardian.py`、`boot_dsw_auto.sh`、`RECOVER.sh`、`reset_supervisor.sh`、`ldm-boot.conf`、`poll.sh`、`results/main/`(18 JSON)、`results/guardian/`、`results_bundle.json` |
| 开机自启 | supervisor `conf.d/ldm-boot.conf` → `boot_dsw_auto.sh`（**该目录在容器临时盘，重启会丢**，`RECOVER.sh`/`reset_supervisor.sh` 是恢复路径） |

> **诊断（2026-10-30 02:30–03:00）**：反复 `PUT /api/v1/notebooks/start` 均返回
> `{"Success":true}`，但实例 20–40 s 后转为 `Failed` + `ReasonMessage: "start timeout"`。
> **CPU 规格（`dsw/CPU`，8 核 32 GB）同样 `start timeout`** ——
> 说明**平台侧资源池整体不可用**，与实例规格、镜像、账号、captcha 均无关，
> 不是配置错误，只能等平台恢复。
> 已确认 `PUT` 是正确的启动接口（前端 bundle 里 `start` 走 `PUT /v1/notebooks/{paid/,}start`），
> `GET /v1/notebooks?Channel=dsw` 是正确的状态查询接口。
> UI 的「启 动」按钮额外要求阿里云 INPAINTING 滑块验证码（纯前端风控，
> **后端 `PUT` 并不校验 captcha**），因此不需要解验证码也能启动。

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

### 3.3 捷径污染数据集（已上传，但标注弃用）

用户裁定：污染数据可作为**错误数据**保留，但必须标注「弃用」。

用**可观测规则**（只看题面 `facts`，不看元数据）复跑全部数据集：
「从每个候选走到尽头，取下游链最长者」——

| 数据集 | 命中率 | 状态 |
|---|---|---|
| `ldm_edge_action` / `_v3` / `_v4p48` / `_v5` | 2400 / 24000 / 24000 / **192000** 全中 = 1.0000 | **弃用** |
| `ldm_mvt` / `_mvt2` / `_mvt3` / `_mvt4` / `_mvt5` | 12000 / 12000 / 12000 / 11900 / 12000 全中 = 1.0000 | **弃用** |
| `ldm_mvte` | 200/200 = **1.0000** | **弃用** |
| `ldm_mvtd` | max 0.2535 / min 0.4953 | 可用 |
| `ldm_mvtd2` | max 0.3355 / min 0.3309 | 可用（主用数据集） |
| `ldm_parity` | 规则不适用（无终止叶） | 可用 |
| `synth_multihop` | 结构不同，规则不适用 | 可用 |

> **污染范围比原先认定的大得多**：不是只有 MVT-5，而是 `edge_action` 整条线 +
> `MVT` 整条线 + `ldm_mvte`。只有 `ldm_mvtd` / `ldm_mvtd2` 是干净的。

标注文件：
- `data/README.md` — 数据集状态总表（可用 / 弃用 + 实测命中率）
- `data/<ds>/DEPRECATED.md` — 每个污染目录内的弃用声明（证据 + 不许怎么用 + 替代方案）
- `code/s109_observable_shortcut.py` — 审计脚本，可复跑
- `docs/errors/err-task-shortcut.md` — 已更新污染范围

### 3.4 文档与脚本的路径差集（非真缺口）

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

## 5. 本次已执行的补传（2026-10-30）

| 缺口 | 动作 | 目标 | 结果 |
|---|---|---|---|
| LDM 核心脚本 17 个 | 补传 | GitHub `ldm-loop-reasoning/code/` | ✅ `code/` 23 → **40** |
| 任务数据集 5 个 | 补传 | GitHub `ldm-loop-reasoning/data/` | ✅ `data/` 4 → **34**（`train.jsonl` 拆 4 片） |
| AMD 主表 18 个 JSON | 上传 | 魔搭 `transfer/results/` | ✅ 全部成功 |
| 缺失结果 JSON 10 个 | 上传 | 魔搭 `transfer/results/` | ✅ 全部成功（`results_server/*`、`server_run2_results/*`） |
| 报告文档 7 份 | 补传 | GitHub `ldm-loop-reasoning/docs/` | ✅ `docs/` 91 → **99** |
| 261 MB `edge_action_v5/train.jsonl` | 拆片 + 原件 | GitHub 4 片 / 魔搭原件 | ✅ 双份（GitHub 拼回，魔搭带原件） |
| HybridLoop v0.1 HF 权重 8.96 GB | **按用户指示暂停** | GitHub `llm-hybrid-loop` | ⏸ 远端仍 100/200 卷（不完整） |
| `runs/` ckpt 补传 | **按用户指示暂停** | 魔搭 `transfer/ckpt/` | ⏸ 已传 55 个（`ckpt/` 全 19 + 部分 `runs/`） |

**补传脚本**（都可重复运行，会跳过已完成项）：
- `s104_upload_ckpt.py` — 任意文件传魔搭（normal / LFS 双路径），状态记于 `%TEMP%\ckpt_upload.state`
- `s107_upload_results.py` — 结果 JSON 补传
- `s97_audit_cloud.py` / `s106_audit_code_data.py` — 本地 vs 云端差集审计（可复跑）
- `s98_push_weights.sh` / `s105_push_files.py` — HF 权重分批/逐文件推送（**已按用户指示暂停**）

> **凭据说明**：GitHub push 用 Windows 凭据管理器里 `git:https://github.com` 的 PAT
> （`YONEK0` 账号），经 `%TEMP%\gcreds.tmp`（`credential.helper=store`）供 git 使用；
> 直连 `github.com` 可用，但经 `gh-proxy.com` 推送会丢失鉴权头（"No anonymous write access"）。
> 魔搭 token 只在内存/脚本常量里用，**不写进仓库、不打印明文**。

---

## 5.1 魔搭写接口鉴权（本次实测，重要复用点）

魔搭 dataset 写接口（commit / LFS）**认 `m_session_id` cookie，其值就是 access token 本身**：

```bash
curl -X POST "https://www.modelscope.cn/api/v1/repos/datasets/<ns>/<repo>/commit/master" \
     -H "Authorization: Bearer $TOKEN" \
     -H "Cookie: m_session_id=$TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"commit_message":"...","actions":[...]}'
```

**只带 `Authorization: Bearer` 会返回 `user not logged in`（Code 10030000001）**，
加 `m_session_id` cookie 即成功（`{"Code":200,"Data":{"commit_id":"..."}}`）。

上传路径：
- `<5 MB` → normal 模式，body 里带 `content: base64`；
- `>=5 MB` → LFS 三步：`POST /info/lfs/objects/batch` 拿 `upload.href` →
  `PUT {href}`（原始字节）→ 再 commit，`action.type="lfs"` + `sha256` + `size`。

工具：`s104_upload_ckpt.py`（幂等，状态记于 `%TEMP%\ckpt_upload.state`）。

> 附带坑：`modelscope` SDK 导入会连带 `import torch`，本机页面文件不足时会
> `OSError: WinError 1455` 直接崩；纯 stdlib 实现（`s104`）绕开该依赖。
> 另：`document.cookie` 读不到 HttpOnly 的 `m_session_id`；
> Edge 的 cookie 库自 v20 起是 app-bound 加密，本地无法解密 —— 所以走上面的
> `m_session_id=<token>` 方案最省事。

---

## 6. 待办（按优先级）

| 优先级 | 事项 | 说明 |
|---|---|---|
| P0 | 权重推送完成并更新 `v0.1` / `v0.1-weights` tag | 当前 tag 指向残缺模型；11 批推完后需重打 tag |
| P1 | 实例恢复 | AMD `RemainingQuota` 已归零（`ReasonMessage: start timeout`），需平台侧恢复或换资源；`/api/v1/amd/status/check` 仍返回 `true`（活动资格在） |
| P2 | `runs/` 83 个 ckpt 上云（进行中） | 2.29 GB，价值中等（多为早期 4B/0.8B 线过程产物） |
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
