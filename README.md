# 图神经网络 Beginner：PyTorch Geometric 实现

本仓库使用 PyTorch Geometric 完成四项练习，统一支持固定随机种子、GPU/CPU 自动选择、JSON 指标输出和快速自检。

## 环境

```bash
conda activate graph
pip install -r requirements.txt
python self_check.py
```

当前环境基于 PyTorch 2.7.1、CUDA 12.8 和 PyG 2.8。`pyg-lib`、`torch-scatter`、`torch-sparse` 用于 NeighborLoader/LinkNeighborLoader 采样。

## 目录

- `task1_node_classification/`：Cora、Citeseer、Flickr 节点分类；GCN/GAT/GraphSAGE/GIN；全图与邻居采样。
- `task2_link_prediction/`：三种数据集上的链路预测；随机边划分、负采样、全图与 LinkNeighborLoader。
- `task3_graph_classification/`：TUDataset/ZINC；四种 GNN 与 Avg/Max/MinPooling。
- `task4_knowledge_graph/`：TransE、RotatE、ConvE；负采样和 filtered MRR/Hits@K。
- `REPORT.md`：实现说明、快速实验结果和分析。

## 快速运行

```bash
bash run_quick_experiments.sh
```

学习率与网络层数消融：

```bash
python run_ablation.py --epochs 30
```

完整训练命令、数据集选择和输出说明见各任务目录的 README。快速脚本用于验证所有模型、采样器、池化和知识图谱模型均可运行；正式对比时应增加 epochs，并对 Cora/Citeseer/Flickr、TUDataset/ZINC 分别执行。
