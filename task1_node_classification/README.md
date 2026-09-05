# 任务一：节点分类

支持 Cora、Citeseer、Flickr，模型为 GCN、GAT、GraphSAGE、GIN；`--mode both` 会比较全图训练和 PyG `NeighborLoader` 邻居采样训练。

```bash
python task1_node_classification/code/train.py --dataset cora --model all --mode both --epochs 100
python task1_node_classification/code/train.py --dataset citeseer --model gcn --mode full --epochs 100
python task1_node_classification/code/train.py --dataset flickr --model sage --mode sampled --epochs 30 --batch-size 2048
```

结果保存为 `task1_node_classification/results.json`，含验证/测试准确率、总时间和每轮时间。可分别调整 `--lr`、`--layers`、`--hidden` 进行参数对照。
