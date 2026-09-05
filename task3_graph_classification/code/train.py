"""Graph classification/regression with model and pooling comparisons."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common import GNNEncoder, resolve_device, set_seed  # noqa: E402


class GraphModel(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, model: str, layers: int, dropout: float, pool: str, categorical: bool) -> None:
        super().__init__()
        self.categorical = categorical
        self.pool_name = pool
        self.embedding = nn.Embedding(64, hidden) if categorical else None
        encoder_input = hidden if categorical else max(1, in_channels)
        self.encoder = GNNEncoder(encoder_input, hidden, hidden, model=model, num_layers=layers, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_channels))

    def forward(self, data):
        if data.x is None:
            x = torch.ones((data.num_nodes, 1), device=data.edge_index.device)
        elif self.categorical:
            x = self.embedding(data.x.view(-1).long().clamp(0, 63))
        else:
            x = data.x.float()
        x = self.encoder(x, data.edge_index)
        if self.pool_name == "avg":
            graph_x = global_mean_pool(x, data.batch)
        elif self.pool_name == "max":
            graph_x = global_max_pool(x, data.batch)
        elif self.pool_name == "min":
            graph_x = -global_max_pool(-x, data.batch)
        else:
            raise ValueError(self.pool_name)
        return self.head(graph_x)


def load_dataset(name: str):
    data_root = ROOT / "task3_graph_classification" / "data"
    if name.lower() == "zinc":
        train = ZINC(str(data_root / "ZINC"), subset=True, split="train")
        val = ZINC(str(data_root / "ZINC"), subset=True, split="val")
        test = ZINC(str(data_root / "ZINC"), subset=True, split="test")
        return train, val, test, "regression", 1, True
    dataset = TUDataset(str(data_root / "TU"), name=name.upper()).shuffle()
    train_end = max(1, int(len(dataset) * 0.8))
    val_end = max(train_end + 1, int(len(dataset) * 0.9))
    train, val, test = dataset[:train_end], dataset[train_end:val_end], dataset[val_end:]
    categorical = bool(dataset[0].x is not None and not dataset[0].x.is_floating_point())
    return train, val, test, "classification", dataset.num_classes, categorical


def run_epoch(model, loader, optimizer, device, task_type: str) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        if task_type == "classification":
            loss = F.cross_entropy(output, batch.y.view(-1))
        else:
            loss = F.l1_loss(output.view(-1), batch.y.float().view(-1))
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
    return total / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, device, task_type: str) -> float:
    model.eval()
    correct, total, absolute_error = 0, 0, 0.0
    for batch in loader:
        batch = batch.to(device)
        output = model(batch)
        if task_type == "classification":
            target = batch.y.view(-1)
            correct += int((output.argmax(dim=-1) == target).sum())
            total += target.numel()
        else:
            target = batch.y.float().view(-1)
            absolute_error += float((output.view(-1) - target).abs().sum())
            total += target.numel()
    return (correct / max(1, total)) if task_type == "classification" else (absolute_error / max(1, total))


def run(args) -> list[dict]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_set, val_set, test_set, task_type, out_channels, categorical = load_dataset(args.dataset)
    loaders = [
        DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
        DataLoader(val_set, batch_size=args.batch_size),
        DataLoader(test_set, batch_size=args.batch_size),
    ]
    sample = train_set[0]
    in_channels = sample.num_node_features if sample.x is not None else 1
    models = ["gcn", "gat", "sage", "gin"] if args.model == "all" else [args.model]
    pools = ["avg", "max", "min"] if args.pool == "all" else [args.pool]
    results = []
    for model_name in models:
        for pool in pools:
            set_seed(args.seed)
            model = GraphModel(in_channels, args.hidden, out_channels, model_name, args.layers, args.dropout, pool, categorical).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            started = time.perf_counter()
            final_loss = 0.0
            for _ in range(args.epochs):
                final_loss = run_epoch(model, loaders[0], optimizer, device, task_type)
            elapsed = time.perf_counter() - started
            row = {
                "task": "graph_classification" if task_type == "classification" else "graph_regression",
                "dataset": args.dataset,
                "model": model_name,
                "pool": pool,
                "device": str(device),
                "train_loss": final_loss,
                "train_seconds": elapsed,
                "seconds_per_epoch": elapsed / args.epochs,
            }
            metric = "accuracy" if task_type == "classification" else "mae"
            row[f"val_{metric}"] = evaluate(model, loaders[1], device, task_type)
            row[f"test_{metric}"] = evaluate(model, loaders[2], device, task_type)
            results.append(row)
            print(json.dumps(row, ensure_ascii=False))
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mutag", "proteins", "zinc"], default="mutag")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "gin", "all"], default="all")
    parser.add_argument("--pool", choices=["avg", "max", "min", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="task3_graph_classification/results.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    output = ROOT / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}")
