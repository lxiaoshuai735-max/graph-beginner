"""Shared PyG models and dataset helpers."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Flickr, Planetoid
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.transforms import NormalizeFeatures


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str = "auto") -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def synchronize_device(device: torch.device) -> None:
    """Wait for queued CUDA work so wall-clock timings are comparable."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_node_dataset(name: str, root: str | Path):
    name = name.lower()
    root = Path(root)
    if name in {"cora", "citeseer"}:
        return Planetoid(root=str(root / name), name=name.capitalize(), transform=NormalizeFeatures())
    if name == "flickr":
        return Flickr(root=str(root / name), transform=NormalizeFeatures())
    raise ValueError(f"unknown dataset: {name}; choose cora, citeseer, or flickr")


class GNNEncoder(nn.Module):
    """A uniform GCN/GAT/GraphSAGE/GIN stack."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        model: str = "gcn",
        num_layers: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        self.model_name = model.lower()
        self.dropout = dropout
        dimensions = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            [self._make_conv(dimensions[i], dimensions[i + 1]) for i in range(num_layers)]
        )

    def _make_conv(self, in_channels: int, out_channels: int):
        if self.model_name == "gcn":
            return GCNConv(in_channels, out_channels)
        if self.model_name == "gat":
            return GATConv(in_channels, out_channels, heads=4, concat=False, dropout=self.dropout)
        if self.model_name in {"sage", "graphsage"}:
            return SAGEConv(in_channels, out_channels)
        if self.model_name == "gin":
            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            return GINConv(mlp, train_eps=True)
        raise ValueError(f"unknown model: {self.model_name}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for index, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if index != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
