import os
import time
import copy
import uuid
import shutil
import argparse
from pathlib import Path

import yaml
import torch
import joblib
import optuna
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
from torch_geometric.loader import DataLoader

from placeholder.source import models, train, utils

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = utils.get_logger("TUNE", log_level="INFO")


def objective(trial, params, dataset):
    # Get static data from parameters
    epochs = params["Data"].get("epochs", 10)
    workers = params["Data"].get("workers", 4)
    size = params["Data"].get("batch_size", 32)
    patience = params["Data"].get("patience", 5)
    model_name = params["Data"].get("model", "gat").lower()
    max_retries = params["Data"].get("max_retries", 5)

    project_dir = params["Data"]["project_dir"]

    # Get optuna suggestionns
    config = models.optuna_suggest(params, trial)
    config["feat_size"] = params["Data"]["feat_size"]
    config["edge_dim"] = params["Data"]["edge_dim"]
    config["bce_weight"] = params["Data"]["bce_weight"]
    params["Data"]["trial"] = trial

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
    retries = 0
    while retries < max_retries:
        try:
            best_metric = -1.0
            patience_counter = 0
            best_model_state = None

            for epoch in tqdm(range(epochs), ncols=120, desc="Training"):
                start = time.time()
                _ = train.train_epoch(model, optim, schdlr,
                                      lossfn, train_loader)
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

                fit_time = time.time() - start

            # Prepare best model
            model.load_state_dict(best_model_state)
            trial_dir = project_dir / f"trial_{trial.number}"
            os.makedirs(trial_dir, exist_ok=True)
            torch.save(best_model_state, str(trial_dir / "best_model.pt"))
            results = train.test_epoch(model, test_loader)

            #  Get model complexity
            n_params = filter(lambda p: p.requires_grad, model.parameters())
            sum_params = sum([np.prod(p.size()) for p in n_params])
            trial.set_user_attr("n_params", int(sum_params))

            # Report time and metrics
            trial.set_user_attr("fit_time", float(round(fit_time, 3)))
            trial.set_user_attr("metrics", results)

            return results["mcc"]

        except torch.cuda.OutOfMemoryError:
            retries += 1
            torch.cuda.empty_cache()

            if retries < max_retries:
                logger.info("OOM")
                time.sleep(90)
            else:
                raise optuna.exceptions.TrialPruned()


def get_dataframe(study):
    records = []
    for trial in study.trials:
        record = {"trial": trial.number,
                  "optim": trial.value}

        dummy = {"acc": np.nan, "acc_bal": np.nan, "f1": np.nan,
                 "prec": np.nan, "rec": np.nan, "mcc": np.nan,
                 "avg_prec": np.nan, "roc_auc": np.nan}

        # Get user attrs
        val_metrics = trial.user_attrs.get("metrics", dummy)
        n_params = trial.user_attrs.get("n_params", np.nan)
        fit_time = trial.user_attrs.get("fit_time", np.nan)

        # Update dict
        record.update(val_metrics)
        record.update(trial.params)
        record.update({"n_params": n_params})
        record.update({"fit_time": fit_time})

        records.append(record)

    df = pd.DataFrame(records)
    return df


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

    # * Optuna
    trials = params["Data"]["trials"]
    url_db = f"sqlite:///{project_dir / 'optuna_study.db'}"

    storage = optuna.storages.RDBStorage(url=url_db)
    study = optuna.create_study(study_name=name, direction="maximize",
                                storage=storage, load_if_exists=True)
    while len(study.trials) <= trials:
        study.optimize(lambda trial: objective(trial, params, dataset),
                       n_trials=1)

    if study.study_name == name:
        df = get_dataframe(study)
        df.to_csv(project_dir / f"{name}.csv", index=False)

    # plot parallel plot
    header = ["optim", "acc", "acc_bal", "f1",
              "prec", "rec", "mcc", "avg_prec",
              "roc_auc", "n_params", "fit_time"]
    feats = [col for col in list(df.columns) if col not in header]
    feats = feats + ["mcc"]

    dimensions = []
    for col in feats:
        col_values = df[col].values
        dim = dict(label=col, values=col_values,
                   range=[col_values.min(), col_values.max()])
        dimensions.append(dim)

    fig = go.Figure(data=go.Parcoords(line=dict(color=df["mcc"],
                                                colorscale="viridis",
                                                showscale=False),
                                      dimensions=dimensions))

    fig.update_layout(width=1600, height=500,
                      margin=dict(l=30, r=20, t=45, b=35))

    fig.write_html(project_dir / f"{name}.html", include_plotlyjs="cdn",
                   config={"displaylogo": False})
