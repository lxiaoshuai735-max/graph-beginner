# 图神经网络 Beginner：PyTorch Geometric 实现

本仓库使用 PyTorch Geometric 完成四项练习，统一支持固定随机种子、GPU/CPU 自动选择、JSON 指标输出和快速自检。

## 环境

```bash
conda activate graph
pip install -r requirements.txt
python self_check.py
python self_check.py --device cpu
```

当前环境基于 PyTorch 2.7.1、CUDA 12.8 和 PyG 2.8。`pyg-lib`、`torch-scatter`、`torch-sparse` 用于 NeighborLoader/LinkNeighborLoader 采样。

## 目录

- `task1_node_classification/`：Cora、Citeseer、Flickr 节点分类；GCN/GAT/GraphSAGE/GIN；全图与邻居采样。
- `task2_link_prediction/`：三种数据集上的链路预测；随机边划分、负采样、全图与 LinkNeighborLoader。
- `task3_graph_classification/`：TUDataset/ZINC；四种 GNN 与 Avg/Max/MinPooling。
- `task4_knowledge_graph/`：TransE、RotatE、ConvE；负采样和 filtered MRR/Hits@K。
- `REPORT.md`：实现说明、快速实验结果和分析。

其中 ConvE 按原论文的核心结构实现：实体与关系嵌入分别重排为二维网格，沿高度拼接，经 `Conv2d` 和全连接投影得到查询向量，再与候选尾实体嵌入打分。代码会自动为任意正整数嵌入维度选择合法的二维形状。

## 快速运行

```bash
bash run_quick_experiments.sh
```

学习率与网络层数消融：

```bash
python run_ablation.py --epochs 30
```

任务三批量大小与耗时对照：

```bash
python run_batch_ablation.py --dataset mutag --model gin --pool max --batch-sizes 16,32,64,128 --epochs 30
```

训练耗时只覆盖训练循环，不含验证/测试；CPU 直接计时，CUDA 在计时前后显式同步。首次运行仍可能包含算子缓存初始化开销，严谨测速建议先做一次不记录的预热运行，再重复多次报告均值与标准差。

## 已验证的数据覆盖

2026-09-05 实际完成：Cora 的节点分类与链路预测正式对比、MUTAG 图分类正式对比、toy KG 知识图谱补全，以及 Citeseer、PROTEINS 的单轮数据加载/训练自检。Flickr 与 ZINC 因当时 AutoDL 无法连接其上游下载源而未实跑；仓库提供了加载和训练入口，但不把它们列为已完成实验。

完整训练命令、数据集选择和输出说明见各任务目录的 README。快速脚本用于验证所有模型、采样器、池化和知识图谱模型均可运行；正式对比时应增加 epochs，并对 Cora/Citeseer/Flickr、TUDataset/ZINC 分别执行。
