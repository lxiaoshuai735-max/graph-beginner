"""Small Cora ablation for learning rate and GNN depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from task1_node_classification.code.train import run


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", default="experiments/task1_ablation.json")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    results = []
    for learning_rate in [0.005, 0.01, 0.02]:
        for layers in [2, 3]:
            args = SimpleNamespace(
                seed=42,
                device=cli.device,
                dataset="cora",
                model="gcn",
                mode="full",
                epochs=cli.epochs,
                hidden=64,
                layers=layers,
                dropout=0.5,
                lr=learning_rate,
                weight_decay=5e-4,
                batch_size=1024,
                fanout=10,
            )
            row = run(args)[0]
            row.update({"learning_rate": learning_rate, "layers": layers, "epochs": cli.epochs})
            results.append(row)

    output = ROOT / cli.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
