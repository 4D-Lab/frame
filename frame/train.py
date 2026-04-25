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
from torch_geometric.loader import DataLoader

from frame.source import models
from frame.source.train import runner

device = "cuda" if torch.cuda.is_available() else "cpu"


def _report_seed_stats(task, per_seed_results, project_dir):
    """Print mean ± std across seeds and dump per-seed metrics to JSON."""
    headline = "mcc" if task == "classification" else "ccc"
    values = [float(r[headline]) for r in per_seed_results]
    mean = float(np.mean(values))
    std = float(np.std(values))

    print(f"{headline.upper()}: {mean:.3f} ± {std:.3f} "
          f"(per-seed: {[round(v, 3) for v in values]})")

    summary = {"headline_metric": headline,
               "mean": round(mean, 4),
               "std": round(std, 4),
               "per_seed": values,
               "per_seed_full": per_seed_results}
    with open(project_dir / "seed_metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)


def run(params, dataset):
    epochs = params["Data"].get("epochs", 10)
    workers = params["Data"].get("workers", 4)
    size = params["Data"].get("batch_size", 32)
    patience = params["Data"].get("patience", 5)
    model_name = params["Data"].get("model", "gat").lower()
    task = params["Data"].get("task", "classification").lower()
    grad_clip = params["Data"].get("grad_clip_norm", 1.0)
    drop_edge_p = float(params["Data"].get("drop_edge_p", 0.0))
    mask_feat_p = float(params["Data"].get("mask_feat_p", 0.0))
    seeds = params["Data"].get("train_seeds", [8])
    if not seeds:
        raise ValueError("train_seeds must contain at least one seed")

    project_dir = params["Data"]["project_dir"]

    config = models.tune_fixed(params)
    config["feat_size"] = params["Data"]["feat_size"]
    config["edge_dim"] = params["Data"]["edge_dim"]
    config["bce_weight"] = params["Data"]["bce_weight"]
    config["task"] = task
    config["regression_loss"] = params["Data"].get("regression_loss", "mse")
    config["huber_delta"] = params["Data"].get("huber_delta", 1.0)
    params["Data"]["trial"] = None

    size = int(config.get("batch_size", size))

    # * Prepare valid loader (shared across seeds; train loader is built
    #   inside the runner so its shuffle is pinned to each seed)
    train_data = [data for data in dataset if data.set == "train"]
    valid_data = [data for data in dataset if data.set == "valid"]
    valid_loader = DataLoader(valid_data, batch_size=size,
                              num_workers=workers,
                              persistent_workers=True)

    # * Train one model per seed, keep the best-seed checkpoint
    per_seed_results = []
    best_state = None
    best_optim = -float("inf")
    for seed in seeds:
        state, results, _, _ = runner.train_one_seed(int(seed), train_data,
                                                     valid_loader,
                                                     model_name, config,
                                                     epochs, patience, task,
                                                     grad_clip, drop_edge_p,
                                                     mask_feat_p, size,
                                                     workers)
        per_seed_results.append(results)
        if results["optim"] > best_optim:
            best_optim = float(results["optim"])
            best_state = state

    os.makedirs(project_dir, exist_ok=True)
    torch.save(best_state, str(project_dir / "best_model.pt"))

    _report_seed_stats(task, per_seed_results, project_dir)


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-c", "--config", dest="config", required=True)
    args = args_parser.parse_args()
    with open(args.config) as stream:
        params = yaml.safe_load(stream)

    # * Initialize
    name = params["Data"]["name"]
    if name.lower() == "none":
        name = str(uuid.uuid4()).split("-")[0]
        params["Data"]["name"] = name

    cwd = Path(os.getcwd())
    project_dir = cwd / "output" / name
    os.makedirs(project_dir, exist_ok=True)

    # * Load dataset
    path_csv = Path(params["Data"]["path_joblib"])
    data = joblib.load(path_csv)

    dataset = data["dataset"]
    params["Data"]["feat_size"] = data["metadata"]["feat_size"]
    params["Data"]["edge_dim"] = data["metadata"]["edge_dim"]
    params["Data"]["bce_weight"] = data["metadata"]["bce_weight"]
    params["Data"]["project_dir"] = project_dir

    if os.path.isdir(cwd / "???"):
        shutil.rmtree(cwd / "???")

    # * Run
    run(params, dataset)
