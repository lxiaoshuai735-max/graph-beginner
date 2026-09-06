"""Compare task-3 batch sizes and preserve each run's metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive comma-separated integers")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mutag", "proteins", "zinc"], default="mutag")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "gin"], default="gin")
    parser.add_argument("--pool", choices=["avg", "max", "min"], default="max")
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=parse_batch_sizes("16,32,64,128"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/task3_batch_ablation.json")
    args = parser.parse_args()

    temporary_dir = ROOT / "experiments" / "tmp"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch_size in args.batch_sizes:
        temporary_output = temporary_dir / f"task3_{args.dataset}_{args.model}_{args.pool}_batch{batch_size}.json"
        command = [
            sys.executable,
            str(ROOT / "task3_graph_classification" / "code" / "train.py"),
            "--dataset", args.dataset,
            "--model", args.model,
            "--pool", args.pool,
            "--batch-size", str(batch_size),
            "--epochs", str(args.epochs),
            "--device", args.device,
            "--seed", str(args.seed),
            "--output", str(temporary_output),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        run_rows = json.loads(temporary_output.read_text(encoding="utf-8"))
        if len(run_rows) != 1:
            raise RuntimeError(f"expected one result for batch_size={batch_size}, got {len(run_rows)}")
        rows.append(run_rows[0])

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
