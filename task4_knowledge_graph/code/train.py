"""Train TransE, RotatE, or ConvE for knowledge-graph completion."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common import resolve_device, set_seed, synchronize_device  # noqa: E402


def read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 3:
            triples.append(tuple(parts))
    return triples


def load_data(data_dir: Path):
    paths = [data_dir / name for name in ["train.tsv", "valid.tsv", "test.tsv"]]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError(f"missing TSV files under {data_dir}; run data/download.py")
    raw_splits = [read_triples(path) for path in paths]
    all_raw = sum(raw_splits, [])
    entities = sorted({value for h, _, t in all_raw for value in [h, t]})
    relations = sorted({r for _, r, _ in all_raw})
    entity_to_id = {value: index for index, value in enumerate(entities)}
    relation_to_id = {value: index for index, value in enumerate(relations)}
    encode = lambda triples: torch.tensor(
        [[entity_to_id[h], relation_to_id[r], entity_to_id[t]] for h, r, t in triples], dtype=torch.long
    )
    return [encode(split) for split in raw_splits], entity_to_id, relation_to_id


class TransE(nn.Module):
    def __init__(self, entities: int, relations: int, dim: int, gamma: float) -> None:
        super().__init__()
        self.entity = nn.Embedding(entities, dim)
        self.relation = nn.Embedding(relations, dim)
        self.gamma = gamma
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def score(self, h, r, t):
        return self.gamma - (self.entity(h) + self.relation(r) - self.entity(t)).norm(p=1, dim=-1)


class RotatE(nn.Module):
    def __init__(self, entities: int, relations: int, dim: int, gamma: float) -> None:
        super().__init__()
        self.entity = nn.Embedding(entities, dim * 2)
        self.phase = nn.Embedding(relations, dim)
        self.gamma = gamma
        nn.init.uniform_(self.entity.weight, -0.5, 0.5)
        nn.init.uniform_(self.phase.weight, -math.pi, math.pi)

    def score(self, h, r, t):
        h_re, h_im = self.entity(h).chunk(2, dim=-1)
        t_re, t_im = self.entity(t).chunk(2, dim=-1)
        phase = self.phase(r)
        r_re, r_im = torch.cos(phase), torch.sin(phase)
        rotated_re = h_re * r_re - h_im * r_im
        rotated_im = h_re * r_im + h_im * r_re
        distance = torch.stack([rotated_re - t_re, rotated_im - t_im], dim=0).norm(dim=0).sum(dim=-1)
        return self.gamma - distance


class ConvE(nn.Module):
    """ConvE interaction using a 2-D head/relation embedding stack."""

    def __init__(self, entities: int, relations: int, dim: int, gamma: float) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("ConvE requires dim >= 1")
        # Choose the most square factorisation available. Prime dimensions safely
        # fall back to (1, dim), while the adaptive kernel remains valid.
        self.embedding_height = math.isqrt(dim)
        while dim % self.embedding_height:
            self.embedding_height -= 1
        self.embedding_width = dim // self.embedding_height
        stacked_height = 2 * self.embedding_height
        kernel_height = min(3, stacked_height)
        kernel_width = min(3, self.embedding_width)
        feature_height = stacked_height - kernel_height + 1
        feature_width = self.embedding_width - kernel_width + 1

        self.entity = nn.Embedding(entities, dim)
        self.relation = nn.Embedding(relations, dim)
        self.input_dropout = nn.Dropout(0.2)
        self.conv = nn.Conv2d(1, 32, kernel_size=(kernel_height, kernel_width))
        self.feature_dropout = nn.Dropout2d(0.2)
        self.projection = nn.Linear(32 * feature_height * feature_width, dim)
        self.hidden_dropout = nn.Dropout(0.3)
        self.bias = nn.Parameter(torch.zeros(entities))
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def encode_query(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Encode ``(head, relation)`` pairs into tail-query vectors."""
        batch_shape = h.shape
        head = self.entity(h).reshape(-1, 1, self.embedding_height, self.embedding_width)
        relation = self.relation(r).reshape(-1, 1, self.embedding_height, self.embedding_width)
        stacked = torch.cat([head, relation], dim=2)
        features = F.relu(self.conv(self.input_dropout(stacked)))
        features = self.feature_dropout(features).flatten(start_dim=1)
        query = self.hidden_dropout(F.relu(self.projection(features)))
        return query.reshape(*batch_shape, -1)

    def score(self, h, r, t):
        query = self.encode_query(h, r)
        return (query * self.entity(t)).sum(dim=-1) + self.bias[t]

    def score_all_tails(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Score every entity as a tail for each ``(head, relation)`` pair."""
        query = self.encode_query(h, r)
        return query @ self.entity.weight.transpose(0, 1) + self.bias


def build_model(name: str, entities: int, relations: int, dim: int, gamma: float):
    classes = {"transe": TransE, "rotate": RotatE, "conve": ConvE}
    return classes[name](entities, relations, dim, gamma)


def train_model(model, triples: torch.Tensor, entities: int, args, device) -> tuple[float, float]:
    loader = DataLoader(TensorDataset(triples), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    synchronize_device(device)
    started = time.perf_counter()
    final_loss = 0.0
    for _ in range(args.epochs):
        for (positive,) in loader:
            positive = positive.to(device)
            negative = positive.clone()
            corrupt_head = torch.rand(len(negative), device=device) < 0.5
            random_entities = torch.randint(entities, (len(negative),), device=device)
            negative[corrupt_head, 0] = random_entities[corrupt_head]
            negative[~corrupt_head, 2] = random_entities[~corrupt_head]
            optimizer.zero_grad()
            positive_score = model.score(positive[:, 0], positive[:, 1], positive[:, 2])
            negative_score = model.score(negative[:, 0], negative[:, 1], negative[:, 2])
            loss = F.softplus(-positive_score).mean() + F.softplus(negative_score).mean()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    synchronize_device(device)
    return final_loss, time.perf_counter() - started


@torch.no_grad()
def filtered_metrics(model, test: torch.Tensor, all_triples: torch.Tensor, entities: int, device) -> dict:
    model.eval()
    known_tails = defaultdict(set)
    known_heads = defaultdict(set)
    for h, r, t in all_triples.tolist():
        known_tails[(h, r)].add(t)
        known_heads[(r, t)].add(h)
    ranks = []
    candidates = torch.arange(entities, device=device)
    for h, r, target in test.tolist():
        # Tail prediction: (h, r, ?)
        heads = torch.full((entities,), h, dtype=torch.long, device=device)
        relations = torch.full((entities,), r, dtype=torch.long, device=device)
        if hasattr(model, "score_all_tails"):
            scores = model.score_all_tails(heads[:1], relations[:1]).squeeze(0)
        else:
            scores = model.score(heads, relations, candidates)
        for other in known_tails[(h, r)] - {target}:
            scores[other] = -torch.inf
        target_score = scores[target]
        ranks.append(int((scores > target_score).sum()) + 1)

        # Head prediction: (?, r, t)
        tails = torch.full((entities,), target, dtype=torch.long, device=device)
        scores = model.score(candidates, relations, tails)
        for other in known_heads[(r, target)] - {h}:
            scores[other] = -torch.inf
        target_score = scores[h]
        ranks.append(int((scores > target_score).sum()) + 1)
    rank_array = np.asarray(ranks, dtype=np.float64)
    return {
        "mrr": float(np.mean(1.0 / rank_array)),
        "hits@1": float(np.mean(rank_array <= 1)),
        "hits@3": float(np.mean(rank_array <= 3)),
        "hits@10": float(np.mean(rank_array <= 10)),
        "mean_rank": float(np.mean(rank_array)),
    }


def run(args) -> list[dict]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    splits, entity_map, relation_map = load_data(Path(args.data_dir))
    train, valid, test = splits
    all_triples = torch.cat(splits, dim=0)
    names = ["transe", "rotate", "conve"] if args.model == "all" else [args.model]
    results = []
    for name in names:
        set_seed(args.seed)
        model = build_model(name, len(entity_map), len(relation_map), args.dim, args.gamma).to(device)
        loss, elapsed = train_model(model, train, len(entity_map), args, device)
        row = {
            "task": "knowledge_graph_completion",
            "model": name,
            "entities": len(entity_map),
            "relations": len(relation_map),
            "train_triples": len(train),
            "evaluation_queries": len(test) * 2,
            "epochs": args.epochs,
            "embedding_dim": args.dim,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "train_loss": loss,
            "train_seconds": elapsed,
            "device": str(device),
            "timing_protocol": "device_synchronized_training_only",
            **filtered_metrics(model, test, all_triples, len(entity_map), device),
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "task4_knowledge_graph" / "data"))
    parser.add_argument("--model", choices=["transe", "rotate", "conve", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="task4_knowledge_graph/results.json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    output = ROOT / arguments.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}")
