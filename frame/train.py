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
    # Get static data from parameters
    epochs = params["Data"].get("epochs", 10)
    workers = params["Data"].get("workers", 4)
    size = params["Data"].get("batch_size", 32)
    patience = params["Data"].get("patience", 5)
    model_name = params["Data"].get("model", "gat").lower()

    project_dir = params["Data"]["project_dir"]

    # Get values
    config = {}
    for name, bounds in params["Tune"].items():
        if isinstance(bounds["value"], int):
            config[name] = int(bounds["value"])
        else:
            config[name] = float(bounds["value"])
    config["feat_size"] = params["Data"]["feat_size"]
    config["edge_dim"] = params["Data"]["edge_dim"]
    config["bce_weight"] = params["Data"]["bce_weight"]
    params["Data"]["trial"] = None

    # * Prepare dataloader
    train_data = [data for data in dataset if data.set == "train"]
    train_loader = DataLoader(train_data, batch_size=size,
                              shuffle=True, num_workers=workers,
                              persistent_workers=True)

    test_data = [data for data in dataset if data.set == "test"]
    test_loader = DataLoader(test_data, batch_size=size,
                             num_workers=workers,
                             persistent_workers=True)

    # * Get model
    model, optim, schdlr, lossfn = models.model_setup(model_name, config)

    # * Train
    best_metric = -1.0
    patience_counter = 0
    best_model_state = None
    for epoch in tqdm(range(epochs), ncols=120, desc="Training"):
        _ = train.train_epoch(model, optim, schdlr, lossfn, train_loader)
        val_metrics = train.test_epoch(model, test_loader)

        # Early stopping check
        if val_metrics["mcc"] > best_metric:
            patience_counter = 0
            best_metric = val_metrics["mcc"]
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
    results = train.test_epoch(model, test_loader)

    print(f"MCC: {results['mcc']}")


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-c", "--config", dest="config", required=True)
    args = args_parser.parse_args()
    with open(args.config) as stream:
        params = yaml.safe_load(stream)

    # * Initialize
    name = params["Data"]["name"]
    if name.lower() == "none":
        name = str(uuid.uuid4()).split["-"][0]
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
