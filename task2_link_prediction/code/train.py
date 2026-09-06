"""Link prediction with full-graph and LinkNeighborLoader training."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    GNNEncoder,
    load_node_dataset,
    resolve_device,
    set_seed,
    synchronize_device,
)


class LinkPredictor(nn.Module):
    def __init__(self, in_channels: int, hidden: int, model: str, layers: int, dropout: float) -> None:
        super().__init__()
        self.encoder = GNNEncoder(in_channels, hidden, hidden, model=model, num_layers=layers, dropout=dropout)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    @staticmethod
    def decode(z, edge_label_index):
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


@torch.no_grad()
def evaluate(model, data, device) -> float:
    model.eval()
    # Keep the original split on CPU for a later LinkNeighborLoader run.
    data = data.clone().to(device)
    z = model.encode(data.x, data.edge_index)
    logits = model.decode(z, data.edge_label_index)
    return float(roc_auc_score(data.edge_label.cpu().numpy(), logits.sigmoid().cpu().numpy()))


def train_full(model, train_data, val_data, test_data, args, device) -> dict:
    train_data = train_data.clone().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    synchronize_device(device)
    started = time.perf_counter()
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        logits = model.decode(z, train_data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, train_data.edge_label.float())
        loss.backward()
        optimizer.step()
    synchronize_device(device)
    elapsed = time.perf_counter() - started
    return {
        "mode": "full",
        "train_seconds": elapsed,
        "seconds_per_epoch": elapsed / args.epochs,
        "val_auc": evaluate(model, val_data, device),
        "test_auc": evaluate(model, test_data, device),
    }


def train_sampled(model, train_data, val_data, test_data, args, device) -> dict:
    loader = LinkNeighborLoader(
        train_data,
        num_neighbors=[args.fanout] * args.layers,
        edge_label_index=train_data.edge_label_index,
        edge_label=train_data.edge_label,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    synchronize_device(device)
    started = time.perf_counter()
    batches = 0
    for _ in range(args.epochs):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = model.encode(batch.x, batch.edge_index)
            logits = model.decode(z, batch.edge_label_index)
            loss = F.binary_cross_entropy_with_logits(logits, batch.edge_label.float())
            loss.backward()
            optimizer.step()
            batches += 1
    synchronize_device(device)
    elapsed = time.perf_counter() - started
    return {
        "mode": "sampled",
        "batches": batches,
        "train_seconds": elapsed,
        "seconds_per_epoch": elapsed / args.epochs,
        "val_auc": evaluate(model, val_data, device),
        "test_auc": evaluate(model, test_data, device),
    }


def run(args) -> list[dict]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset = load_node_dataset(args.dataset, ROOT / "task2_link_prediction" / "data")
    splitter = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        split_labels=False,
    )
    train_data, val_data, test_data = splitter(dataset[0])
    models = ["gcn", "gat", "sage", "gin"] if args.model == "all" else [args.model]
    modes = ["full", "sampled"] if args.mode == "both" else [args.mode]
    results = []
    for model_name in models:
        for mode in modes:
            set_seed(args.seed)
            model = LinkPredictor(dataset.num_features, args.hidden, model_name, args.layers, args.dropout).to(device)
            if mode == "full":
                metrics = train_full(model, train_data, val_data, test_data, args, device)
            else:
                metrics = train_sampled(model, train_data, val_data, test_data, args, device)
            row = {
                "task": "link_prediction",
                "dataset": args.dataset,
                "model": model_name,
                "device": str(device),
                "epochs": args.epochs,
                "hidden_channels": args.hidden,
                "layers": args.layers,
                "learning_rate": args.lr,
                "batch_size": args.batch_size if mode == "sampled" else None,
                "fanout": args.fanout if mode == "sampled" else None,
                "timing_protocol": "device_synchronized_training_only",
                **metrics,
            }
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
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="task2_link_prediction/results.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    output = ROOT / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}")
