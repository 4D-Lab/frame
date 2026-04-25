import os
import time
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
import plotly.graph_objects as go
from torch_geometric.loader import DataLoader

from frame.source import models, utils
from frame.source.train import runner

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = utils.get_logger("TUNE", log_level="INFO")


def _aggregate_seed_metrics(per_seed_results):
    """Mean each per-seed validation-metric dict, rounded to 3 dp."""
    keys = per_seed_results[0].keys()
    return {key: round(float(np.mean([r[key] for r in per_seed_results])), 3)
            for key in keys}


def _record_trial(trial, project_dir, best_state, per_seed_optim,
                  per_seed_results, total_time, n_params, best_seed):
    """Persist best-seed checkpoint and write seed-aware trial attrs."""
    trial_dir = project_dir / f"trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    torch.save(best_state, str(trial_dir / "best_model.pt"))

    trial.set_user_attr("n_params", int(n_params))
    trial.set_user_attr("fit_time", float(round(total_time, 3)))
    trial.set_user_attr("metrics", _aggregate_seed_metrics(per_seed_results))
    trial.set_user_attr("optim_mean",
                        round(float(np.mean(per_seed_optim)), 4))
    trial.set_user_attr("optim_std",
                        round(float(np.std(per_seed_optim)), 4))
    trial.set_user_attr("optim_per_seed",
                        [round(float(v), 4) for v in per_seed_optim])
    trial.set_user_attr("best_seed", int(best_seed))


def objective(trial, params, dataset):
    # Get static data from parameters
    epochs = params["Data"].get("epochs", 10)
    workers = params["Data"].get("workers", 4)
    size = params["Data"].get("batch_size", 32)
    patience = params["Data"].get("patience", 5)
    model_name = params["Data"].get("model", "gat").lower()
    max_retries = params["Data"].get("max_retries", 5)
    task = params["Data"].get("task", "classification").lower()
    grad_clip = params["Data"].get("grad_clip_norm", 1.0)
    drop_edge_p = float(params["Data"].get("drop_edge_p", 0.0))
    mask_feat_p = float(params["Data"].get("mask_feat_p", 0.0))
    seeds = params["Data"].get("tune_seeds", [8])
    if not seeds:
        raise ValueError("tune_seeds must contain at least one seed")

    project_dir = params["Data"]["project_dir"]

    # Get optuna suggestionns
    config = models.optuna_suggest(params, trial)
    config["feat_size"] = params["Data"]["feat_size"]
    config["edge_dim"] = params["Data"]["edge_dim"]
    config["bce_weight"] = params["Data"]["bce_weight"]
    config["task"] = task
    config["regression_loss"] = params["Data"].get("regression_loss", "mse")
    config["huber_delta"] = params["Data"].get("huber_delta", 1.0)
    params["Data"]["trial"] = trial
    size = int(config.get("batch_size", size))

    # * Prepare dataloader (valid is shared across seeds; train is rebuilt
    #   inside the runner so its shuffle is pinned to the seed)
    train_data = [data for data in dataset if data.set == "train"]
    valid_data = [data for data in dataset if data.set == "valid"]
    valid_loader = DataLoader(valid_data, batch_size=size,
                              num_workers=workers,
                              persistent_workers=True)

    # * Train one model per seed and aggregate
    retries = 0
    while retries < max_retries:
        try:
            per_seed_optim = []
            per_seed_results = []
            best_state = None
            best_optim = -float("inf")
            best_seed = int(seeds[0])
            total_time = 0.0
            n_params = 0
            for seed in seeds:
                state, results, fit_time, n_params = runner.train_one_seed(
                    int(seed), train_data, valid_loader, model_name,
                    config, epochs, patience, task, grad_clip,
                    drop_edge_p, mask_feat_p, size, workers)
                per_seed_optim.append(float(results["optim"]))
                per_seed_results.append(results)
                total_time += fit_time
                if results["optim"] > best_optim:
                    best_optim = float(results["optim"])
                    best_state = state
                    best_seed = int(seed)

            _record_trial(trial, project_dir, best_state, per_seed_optim,
                          per_seed_results, total_time, n_params,
                          best_seed)
            return float(np.mean(per_seed_optim))

        except torch.cuda.OutOfMemoryError:
            retries += 1
            torch.cuda.empty_cache()

            if retries < max_retries:
                logger.info("OOM")
                time.sleep(90)
            else:
                raise optuna.exceptions.TrialPruned()


def get_dataframe(study, task):
    records = []
    for trial in study.trials:
        record = {"trial": trial.number,
                  "optim": trial.value}

        if task == "classification":
            dummy = {"optim": np.nan, "acc": np.nan, "acc_bal": np.nan,
                     "f1": np.nan, "prec": np.nan, "rec": np.nan,
                     "mcc": np.nan, "avg_prec": np.nan, "roc_auc": np.nan}
        else:
            dummy = {"optim": np.nan, "r2": np.nan, "rmse": np.nan,
                     "mae": np.nan, "rto_r2": np.nan, "ccc": np.nan,
                     "roy_c": np.nan, "roy_c_inv": np.nan, "delta": np.nan}

        # Get user attrs
        val_metrics = trial.user_attrs.get("metrics", dummy)
        n_params = trial.user_attrs.get("n_params", np.nan)
        fit_time = trial.user_attrs.get("fit_time", np.nan)
        optim_std = trial.user_attrs.get("optim_std", np.nan)
        best_seed = trial.user_attrs.get("best_seed", np.nan)

        # Update dict
        record.update(val_metrics)
        record.update(trial.params)
        record.update({"n_params": n_params})
        record.update({"fit_time": fit_time})
        record.update({"optim_std": optim_std})
        record.update({"best_seed": best_seed})

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
    task = params["Data"]["task"]
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
        df = get_dataframe(study, task)
        df.to_csv(project_dir / f"{name}.csv", index=False)

    # plot parallel plot
    header = ["optim", "acc", "acc_bal", "f1", "prec", "rec",
              "mcc", "avg_prec", "roc_auc", "r2", "rmse",
              "mae", "rto_r2", "ccc", "roy_c", "roy_c_inv",
              "delta", "n_params", "fit_time", "optim_std",
              "best_seed"]
    feats = [col for col in list(df.columns) if col not in header]
    feats = feats + ["optim"]

    dimensions = []
    for col in feats:
        col_values = df[col].values
        dim = dict(label=col, values=col_values,
                   range=[col_values.min(), col_values.max()])
        dimensions.append(dim)

    fig = go.Figure(data=go.Parcoords(line=dict(color=df["optim"],
                                                colorscale="viridis",
                                                showscale=False),
                                      dimensions=dimensions))

    fig.update_layout(width=1600, height=500,
                      margin=dict(l=30, r=20, t=45, b=35))

    fig.write_html(project_dir / f"{name}.html", include_plotlyjs="cdn",
                   config={"displaylogo": False})
