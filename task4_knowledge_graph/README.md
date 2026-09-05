# 任务四：知识图谱补全

实现 TransE、RotatE、ConvE 三种知识图谱嵌入模型，使用头/尾实体负采样训练，并对头实体预测与尾实体预测共同计算过滤后的 MRR、Mean Rank、Hits@1/3/10。输入是三个制表符分隔文件：`train.tsv`、`valid.tsv`、`test.tsv`。

```bash
python task4_knowledge_graph/data/download.py
python task4_knowledge_graph/code/train.py --model all --epochs 300
python task4_knowledge_graph/code/train.py --data-dir /path/to/your/kg --model rotate --epochs 1000
```

内置 toy KG 用于自检；替换 TSV 文件即可训练其他知识图谱数据集。结果保存到 `task4_knowledge_graph/results.json`。
