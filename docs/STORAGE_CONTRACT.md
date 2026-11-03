# 存储契约：数据集与权重文件禁止写入 C 盘

- **生效日期**：2026-10-24
- **约束范围**：本项目（HybridLoop / 循环深度线）的全部脚本、实验、交接 AI 与自动化流程
- **违反后果**：C 盘曾因 HF 默认缓存被填至 95%（仅模型缓存即 17GB）；此后任何写入 C 盘的权重/数据集都必须迁移清理

---

## 1. 存储布局（现状，已迁移并验证）

| 内容 | 位置 | 说明 |
|---|---|---|
| **HF 全局缓存（hub + datasets）** | `E:\develop\project_files\llm-lab\hf_cache\` | 17GB，已从 C 盘迁入；含 Qwen3.5-0.8B-Base(1.7G)、Qwen3.5-4B-Base(8.8G)、Qwen3-1.7B(3.8G)、gpt2 等全部底座 |
| **全局环境变量（已 setx）** | `HF_HOME = E:\develop\project_files\llm-lab\hf_cache`<br>`HF_HUB_CACHE = E:\develop\project_files\llm-lab\hf_cache\hub` | 用户级持久变量；**新开的终端/进程自动生效**；旧终端需重开 |
| **兼容连接点（Junction）** | `C:\Users\li287\.cache\huggingface\hub` → E 盘<br>`C:\Users\li287\.cache\huggingface\datasets` → E 盘 | 覆盖项目内硬编码路径（如 `s08_arch.py` 的 `SNAP`）；C 盘侧不占空间 |

**迁移后验证（2026-10-24，全部通过）**：① 旧路径经 junction 加载 0.8B tokenizer（eos=248044）；② `HF_HOME` 指向新缓存离线解析 4B 快照；③ C 盘剩余空间 18G→35G。

## 2. 契约条款（必须遵守）

1. **禁止**将模型权重、数据集、检查点、生成语料写入 C 盘任何位置——包括但不限于 `~/.cache/huggingface`、`~/.cache/modelscope`、临时目录下的持久文件。
2. **下载/训练/评测脚本**必须满足以下任一：
   - 运行环境已继承 `HF_HOME`（新终端自动满足）；或
   - 命令行显式携带 `HF_HOME=E:\develop\project_files\llm-lab\hf_cache`；或
   - 权重/数据路径直接指向 E 盘绝对路径。
3. **新增大文件（>100MB）的默认落盘位置**：`E:\develop\project_files\llm-lab\` 下按项目建目录。
4. **交接 AI / 新脚本**在上服务器或本机前，先读本文档；任何 `from_pretrained`、`snapshot_download`、`load_dataset` 调用不得假设 C 盘缓存可写。
5. **例外**：C 盘只允许临时小文件（<100MB）与系统缓存；若某工具强行写 C 盘（如旧版库忽略 HF_HOME），须在事后迁移并在本文档附录登记。
6. **连接点（Junction）不得删除**：它是旧硬编码路径的兼容层；删除会导致 `s08_arch.py` 等脚本立刻找不到底座。
7. 检查点（`runs/*/ckpt_*.pt`）继续放项目盘 `E:\develop\project_files\llm-test\runs\`（本就在 E 盘，维持现状）。

## 3. 快速自查命令

```bash
# 环境变量是否生效（新终端）
echo $HF_HOME          # 应输出 E:\develop\project_files\llm-lab\hf_cache

# C 盘 HF 缓存实际占用（应只显示 junction 元数据 ~KB 级）
du -sh /c/Users/li287/.cache/huggingface

# 大文件落盘审计：找出 C 盘用户目录下 >100MB 的文件（排除 junction 穿透）
find /c/Users/li287 -xdev -type f -size +100M 2>/dev/null | head
```

## 4. 附录：迁移记录

- 2026-10-24：`~/.cache/huggingface/{hub,datasets}`（17GB）整体迁至
  `E:\develop\project_files\llm-lab\hf_cache\`；原路径建 Junction 兼容；
  `setx HF_HOME / HF_HUB_CACHE` 已设；双路径加载验证通过。
- 待办（无期限约束）：发现新的大文件落 C 盘时在此登记并迁移。
