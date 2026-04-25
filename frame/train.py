import os
import copy
import uuid
import shutil
import argparse
from pathlib import Path

import yaml
import torch
import joblib
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from frame.source import models, train

device = "cuda" if torch.cuda.is_available() else "cpu"


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

    # * Prepare dataloader
    train_data = [data for data in dataset if data.set == "train"]
    train_loader = DataLoader(train_data, batch_size=size,
                              shuffle=True, num_workers=workers,
                              persistent_workers=True)

    valid_data = [data for data in dataset if data.set == "valid"]
    valid_loader = DataLoader(valid_data, batch_size=size,
                              num_workers=workers,
                              persistent_workers=True)

    # * Get model
    model, optim, schdlr, lossfn = models.model_setup(model_name, config,
                                                      epochs=epochs)

    # * Train
    best_metric = -1.0
    patience_counter = 0
    best_model_state = None
    for epoch in tqdm(range(epochs), ncols=120, desc="Training"):
        _ = train.train_epoch(model, optim, lossfn, train_loader,
                              grad_clip_norm=grad_clip,
                              drop_edge_p=drop_edge_p,
                              mask_feat_p=mask_feat_p)
        val_metrics = train.valid_epoch(model, task, valid_loader)
        schdlr.step()

        # Early stopping check
        if val_metrics["optim"] > best_metric:
            patience_counter = 0
            best_metric = val_metrics["optim"]
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1

        # Check if we should stop early
        if patience_counter >= patience:
            break

    # Prepare best model
    model.load_state_dict(best_model_state)
    os.makedirs(project_dir, exist_ok=True)
    torch.save(best_model_state, str(project_dir / "best_model.pt"))
    results = train.valid_epoch(model, task, valid_loader)

    if task == "classification":
        print(f"MCC: {results['mcc']}")
    else:
        print(f"CCC: {results['ccc']}")


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
