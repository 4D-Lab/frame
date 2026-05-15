import os
import json
import uuid
import shutil
import argparse
from pathlib import Path

import yaml
import torch
import joblib
import numpy as np

from frame.source import datasets

device = "cuda" if torch.cuda.is_available() else "cpu"


def _graph_size_stats(dataset):
    """Return mean/std of node and edge counts across a dataset."""
    n_nodes = []
    n_edges = []
    for data in dataset:
        n_nodes.append(int(data.x.shape[0]))
        n_edges.append(int(data.edge_index.shape[1]))
    if not n_nodes:
        return {"n_graphs": 0,
                "nodes": {"mean": 0.0, "std": 0.0},
                "edges": {"mean": 0.0, "std": 0.0}}
    return {"n_graphs": len(n_nodes),
            "nodes": {"mean": float(np.mean(n_nodes)),
                      "std": float(np.std(n_nodes)),
                      "min": int(np.min(n_nodes)),
                      "max": int(np.max(n_nodes))},
            "edges": {"mean": float(np.mean(n_edges)),
                      "std": float(np.std(n_edges)),
                      "min": int(np.min(n_edges)),
                      "max": int(np.max(n_edges))}}


def _write_dataset_stats(dataset, loader: str, path_csv: str,
                         project_dir: Path):
    """Persist graph-size and BRICS-exclusion stats to dataset_stats.json."""
    stats = {"loader": loader,
             "source_csv": str(path_csv),
             "graph_size": _graph_size_stats(dataset)}
    if loader == "decompose" and hasattr(dataset, "exclusion_summary"):
        stats["brics_exclusion"] = dataset.exclusion_summary()
    with open(project_dir / "dataset_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-c", "--config", dest="config", required=True)
    args = args_parser.parse_args()
    with open(args.config) as stream:
        params = yaml.safe_load(stream)

    # Get params
    name = params["Data"].get("name", None)
    path_csv = params["Data"].get("path_csv", None)
    loader = params["Data"].get("loader", "default").lower()

    if name.lower() == "none":
        name = str(uuid.uuid4()).split("-")[0]

    # * Initialize
    cwd = Path(os.getcwd())
    project_dir = cwd / "output" / name
    os.makedirs(project_dir, exist_ok=True)

    # * Create dataset
    if loader == "default":
        dataset = datasets.MolecularDataset(path_csv)
    elif loader == "decompose":
        dataset = datasets.DecomposeDataset(path_csv)
    else:
        raise NotImplementedError("Loader not available")

    # * Export
    task = params["Data"].get("task", "classification").lower()
    if task == "classification" and sum(dataset.y) > 0:
        bce_weight = (len(dataset.y) - sum(dataset.y)) / sum(dataset.y)
    else:
        bce_weight = torch.tensor(1.0)
    metadata = {"feat_size": dataset.num_node_features,
                "edge_dim": dataset.num_edge_features,
                "bce_weight": bce_weight,
                "loader": loader,
                "project_dir": project_dir}

    dump_data = {"dataset": dataset, "metadata": metadata}
    joblib.dump(dump_data, project_dir / "data.joblib")

    _write_dataset_stats(dataset, loader, path_csv, project_dir)

    if os.path.isdir(cwd / "???"):
        shutil.rmtree(cwd / "???")
