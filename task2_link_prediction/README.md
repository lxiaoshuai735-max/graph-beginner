# 任务二：链路预测

在 Cora、Citeseer、Flickr 上通过 `RandomLinkSplit` 构造训练/验证/测试边和负样本，比较 GCN、GAT、GraphSAGE、GIN 编码器；边解码器使用节点向量点积。采样模式使用 PyG `LinkNeighborLoader`。

```bash
python task2_link_prediction/code/train.py --dataset cora --model all --mode both --epochs 100
python task2_link_prediction/code/train.py --dataset flickr --model sage --mode sampled --epochs 30
```

结果保存到 `task2_link_prediction/results.json`，评价指标为验证集和测试集 ROC-AUC，并记录超参数与训练耗时。CUDA 计时在训练循环前后显式同步。
