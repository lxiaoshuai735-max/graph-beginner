# 任务三：图分类

支持 TUDataset（默认 MUTAG，也可用 PROTEINS）和 ZINC subset，比较 GCN、GAT、GraphSAGE、GIN 与 Avg/Max/Min 三种全局池化。TUDataset 使用准确率，ZINC 图回归使用 MAE。

```bash
python task3_graph_classification/code/train.py --dataset mutag --model all --pool all --epochs 100
python task3_graph_classification/code/train.py --dataset proteins --model gin --pool avg --epochs 100
python task3_graph_classification/code/train.py --dataset zinc --model gcn --pool all --epochs 50
```

结果保存到 `task3_graph_classification/results.json`，包含批量大小、超参数和经过设备同步的训练耗时。使用 `--lr`、`--layers` 做参数消融；在仓库根目录运行 `python run_batch_ablation.py` 可比较不同批量大小的性能与耗时。
