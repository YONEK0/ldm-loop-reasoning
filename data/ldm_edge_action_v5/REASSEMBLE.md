# train.jsonl 分片还原

GitHub 单文件上限 100 MB，`train.jsonl`（262 MB / 192000 行）拆成 4 片存放。

还原：

```bash
cd data/ldm_edge_action_v5
cat train.jsonl.part*of4 > train.jsonl
wc -l train.jsonl   # 期望 192000
```

原始未拆分文件同时存于 ModelScope 数据集 `Yoneko/ldm-feasibility-testkit`
的 `transfer/data/ldm_edge_action_v5/train.jsonl`。
