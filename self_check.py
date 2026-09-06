"""Offline structural and forward/backward checks for all four tasks."""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader

from common import GNNEncoder, resolve_device, set_seed
from task3_graph_classification.code.train import GraphModel
from task4_knowledge_graph.code.train import build_model, filtered_metrics


def main(device_name: str = "auto") -> None:
    set_seed(7)
    device = resolve_device(device_name)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0, 0, 2], [1, 0, 2, 1, 3, 2, 0, 3, 2, 0]],
        dtype=torch.long,
    )
    x = torch.randn(4, 8)
    y = torch.tensor([0, 1, 0, 1])
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=torch.tensor([1, 1, 1, 0], dtype=torch.bool))
    results = {"device": str(device), "gnn": {}, "samplers": {}, "pools": {}, "kge": {}}

    for name in ["gcn", "gat", "sage", "gin"]:
        model = GNNEncoder(8, 16, 2, model=name, num_layers=2).to(device)
        output = model(x.to(device), edge_index.to(device))
        torch.nn.functional.cross_entropy(output, y.to(device)).backward()
        results["gnn"][name] = list(output.shape)

    neighbor_batch = next(iter(NeighborLoader(data, input_nodes=data.train_mask, num_neighbors=[2, 2], batch_size=2)))
    data.edge_label_index = edge_index[:, :6]
    data.edge_label = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float)
    link_batch = next(iter(LinkNeighborLoader(data, num_neighbors=[2, 2], edge_label_index=data.edge_label_index, edge_label=data.edge_label, batch_size=3)))
    results["samplers"] = {
        "NeighborLoader": int(neighbor_batch.num_nodes),
        "LinkNeighborLoader": int(link_batch.num_nodes),
    }

    graph_data = Data(
        x=x,
        edge_index=edge_index,
        batch=torch.tensor([0, 0, 1, 1]),
        y=torch.tensor([0, 1]),
    ).to(device)
    for pool in ["avg", "max", "min"]:
        model = GraphModel(8, 16, 2, "gcn", 2, 0.1, pool, False).to(device)
        output = model(graph_data)
        output.sum().backward()
        results["pools"][pool] = list(output.shape)

    triples = torch.tensor([[0, 0, 1], [1, 0, 2]], device=device)
    for name in ["transe", "rotate", "conve"]:
        model = build_model(name, entities=4, relations=2, dim=16, gamma=8.0).to(device)
        score = model.score(triples[:, 0], triples[:, 1], triples[:, 2])
        score.sum().backward()
        results["kge"][name] = list(score.shape)

    # Exercise square, composite, prime, and minimal embedding dimensions. This
    # catches invalid ConvE reshape/kernel assumptions without a dataset download.
    conve_dimensions = {}
    for dim in [1, 16, 17, 100]:
        model = build_model("conve", entities=4, relations=2, dim=dim, gamma=8.0).to(device)
        if not isinstance(model.conv, nn.Conv2d):
            raise AssertionError("ConvE must use Conv2d")
        if model.embedding_height * model.embedding_width != dim:
            raise AssertionError(f"invalid ConvE reshape for dim={dim}")
        score = model.score(triples[:, 0], triples[:, 1], triples[:, 2])
        if score.shape != (2,) or not bool(torch.isfinite(score).all()):
            raise AssertionError(f"invalid ConvE scores for dim={dim}")
        all_tail_scores = model.score_all_tails(triples[:, 0], triples[:, 1])
        if all_tail_scores.shape != (2, 4) or not bool(torch.isfinite(all_tail_scores).all()):
            raise AssertionError(f"invalid ConvE all-tail scores for dim={dim}")
        score.sum().backward()
        conve_dimensions[str(dim)] = [model.embedding_height, model.embedding_width]
    results["kge"]["conve_reshape_checks"] = conve_dimensions

    ranking_model = build_model("conve", entities=4, relations=2, dim=16, gamma=8.0).to(device)
    ranking = filtered_metrics(ranking_model, triples, triples, entities=4, device=device)
    if set(ranking) != {"mrr", "hits@1", "hits@3", "hits@10", "mean_rank"}:
        raise AssertionError("filtered ranking returned an unexpected metric schema")
    results["kge"]["conve_filtered_ranking"] = ranking

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("SELF_CHECK_OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    main(arguments.device)
