"""Shared utilities for the graph beginner tasks."""

from .gnn import GNNEncoder, load_node_dataset, resolve_device, set_seed

__all__ = ["GNNEncoder", "load_node_dataset", "resolve_device", "set_seed"]
