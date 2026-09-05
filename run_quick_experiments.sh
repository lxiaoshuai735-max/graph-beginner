#!/usr/bin/env bash
set -euo pipefail

python self_check.py
python task1_node_classification/code/train.py --dataset cora --model all --mode both --epochs 5 --output task1_node_classification/results_quick.json
python task2_link_prediction/code/train.py --dataset cora --model all --mode both --epochs 3 --output task2_link_prediction/results_quick.json
python task3_graph_classification/code/train.py --dataset mutag --model all --pool all --epochs 3 --output task3_graph_classification/results_quick.json
python task4_knowledge_graph/data/download.py
python task4_knowledge_graph/code/train.py --model all --epochs 50 --output task4_knowledge_graph/results_quick.json

echo "All quick experiments completed."
