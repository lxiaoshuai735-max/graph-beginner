"""Full-graph and neighbor-sampled node classification."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common import GNNEncoder, load_node_dataset, resolve_device, set_seed  # noqa: E402


@torch.no_grad()
def accuracy(model, data, mask, device) -> float:
    model.eval()
    data = data.to(device)
    prediction = model(data.x, data.edge_index).argmax(dim=-1)
    return float((prediction[mask] == data.y[mask]).float().mean())


def train_full(model, data, epochs: int, lr: float, weight_decay: float, device) -> dict:
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    return {
        "mode": "full",
        "train_seconds": elapsed,
        "seconds_per_epoch": elapsed / epochs,
        "val_accuracy": accuracy(model, data, data.val_mask, device),
        "test_accuracy": accuracy(model, data, data.test_mask, device),
    }


def train_sampled(model, data, epochs: int, lr: float, weight_decay: float, device, batch_size: int, fanout: int, layers: int) -> dict:
    loader = NeighborLoader(
        data,
        input_nodes=data.train_mask,
        num_neighbors=[fanout] * layers,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    started = time.perf_counter()
    batches = 0
    for _ in range(epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)[: batch.batch_size]
            loss = F.cross_entropy(logits, batch.y[: batch.batch_size])
            loss.backward()
            optimizer.step()
            batches += 1
    elapsed = time.perf_counter() - started
    return {
        "mode": "sampled",
        "train_seconds": elapsed,
        "seconds_per_epoch": elapsed / epochs,
        "batches": batches,
        "val_accuracy": accuracy(model, data, data.val_mask, device),
        "test_accuracy": accuracy(model, data, data.test_mask, device),
    }


def run(args) -> list[dict]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = load_node_dataset(args.dataset, ROOT / "task1_node_classification" / "data")
    data = dataset[0]
    models = ["gcn", "gat", "sage", "gin"] if args.model == "all" else [args.model]
    modes = ["full", "sampled"] if args.mode == "both" else [args.mode]
    results = []
    for model_name in models:
        for mode in modes:
            set_seed(args.seed)
            model = GNNEncoder(
                dataset.num_features,
                args.hidden,
                dataset.num_classes,
                model=model_name,
                num_layers=args.layers,
                dropout=args.dropout,
            ).to(device)
            if mode == "full":
                metrics = train_full(model, data, args.epochs, args.lr, args.weight_decay, device)
            else:
                metrics = train_sampled(
                    model, data, args.epochs, args.lr, args.weight_decay, device,
                    args.batch_size, args.fanout, args.layers,
                )
            row = {"task": "node_classification", "dataset": args.dataset, "model": model_name, "device": str(device), **metrics}
            results.append(row)
            print(json.dumps(row, ensure_ascii=False))
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cora", "citeseer", "flickr"], default="cora")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "gin", "all"], default="all")
    parser.add_argument("--mode", choices=["full", "sampled", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="task1_node_classification/results.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    output = ROOT / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}")
