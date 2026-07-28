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
    """Persist graph-size and exclusion stats to dataset_stats.json."""
    stats = {"loader": loader,
             "source_csv": str(path_csv),
             "graph_size": _graph_size_stats(dataset)}
    if loader == "decompose" and hasattr(dataset, "exclusion_summary"):
        stats["decomposition_exclusion"] = dataset.exclusion_summary()
    with open(project_dir / "dataset_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)


def _degree_histogram(dataset, max_bins: int = 512):
    """Train-split node in-degree histogram for PNA degree scalers.

    Accumulates, over the training graphs (data.set == "train"),
    how many nodes have each in-degree. The node count comes from
    data.x (the definitive atom set) rather than num_nodes,
    and destination indices are clamped into range, so each graph
    allocates only per-graph-sized tensors. Per-node degrees are
    capped at max_bins - 1 to bound the histogram length. Edgeless
    graphs (the (2, 0) fallback) are skipped. Falls back to the
    full dataset if no graph is labelled "train".

    Args:
        dataset: The in-memory dataset of PyG Data objects.
        max_bins: Hard cap on the histogram length; degrees at or
            above it fall in the final bin. Defaults to 512.
    """
    graphs = [d for d in dataset if getattr(d, "set", None) == "train"]
    if not graphs:
        graphs = list(dataset)
    deg = torch.zeros(1, dtype=torch.long)
    for data in graphs:
        if data.edge_index.numel() == 0:
            continue
        n = int(data.x.size(0))
        dst = data.edge_index[1].clamp(min=0, max=n - 1)
        node_deg = torch.bincount(dst, minlength=n).clamp(max=max_bins - 1)
        counts = torch.bincount(node_deg, minlength=deg.numel())
        if counts.numel() > deg.numel():
            counts[:deg.numel()] += deg
            deg = counts
        else:
            deg += counts
    return deg


def _feature_scale(dataset, eps: float = 1e-6):
    """Per-column max absolute node-feature value over the training split.

    Max-abs is used rather than a z-score for two reasons. Most columns
    are one-hot indicators whose standard deviation is tiny for a rare
    element, so dividing by it amplifies that column by two orders of
    magnitude; dividing by the max leaves every indicator column exactly
    as it was. And it maps zero to zero, so the all-zero Integrated
    Gradients baseline keeps its meaning of "nothing present" instead of
    silently becoming "an average fragment".

    The scale comes from the training graphs only, so no test-set
    information reaches the transform.

    Args:
        dataset: In-memory dataset of PyG Data objects, each
            carrying a set attribute.
        eps: Value below which a column is treated as all-zero and left
            unscaled. Defaults to 1e-6.

    Returns:
        Tensor of shape (n_features,) of per-column divisors.
    """
    graphs = [d for d in dataset if getattr(d, "set", None) == "train"]
    if not graphs:
        graphs = list(dataset)
    x = torch.cat([d.x for d in graphs], dim=0).float()
    scale = x.abs().max(dim=0).values
    scale[scale < eps] = 1.0
    return scale


def _scale_features(dataset, scale):
    """Divide node features in place by the given per-column scale.

    Atom-level features are one-hot indicators bounded in [-1, 1] while
    fragment-level features are unbounded atom counts reaching 28, so
    without this the two encodings reach the model on scales an order of
    magnitude apart and the comparison confounds the representation with
    its input scale. Atom-level columns have a max of 1 and so pass
    through unchanged.

    Args:
        dataset: In-memory dataset whose collated store is rewritten.
        scale: Per-column divisors from _feature_scale.
    """
    store = getattr(dataset, "_data", None)
    if store is None:
        store = dataset.data
    store.x = store.x.float() / scale


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
                "project_dir": project_dir,
                "deg": _degree_histogram(dataset)}

    # Node-feature scaling, from the training split only.
    normalize = params["Data"].get("normalize_features", True)
    feat_scale = _feature_scale(dataset) if normalize else None
    metadata["normalize_features"] = normalize
    metadata["feat_scale"] = feat_scale

    # Iterating the dataset (stats, degree histogram) fills PyG's
    # InMemoryDataset _data_list cache with per-graph tensor views
    # into the collated store; persisting that cache would bloat the
    # joblib ~N-fold. Reset it so only the compact store is serialized.
    dataset._data_list = None

    # Applied after the cache reset so the cached views, which point at
    # the pre-scaling tensor, cannot be handed out afterwards.
    if normalize:
        _scale_features(dataset, feat_scale)

    dump_data = {"dataset": dataset, "metadata": metadata}
    joblib.dump(dump_data, project_dir / "data.joblib")

    _write_dataset_stats(dataset, loader, path_csv, project_dir)

    if os.path.isdir(cwd / "???"):
        shutil.rmtree(cwd / "???")
