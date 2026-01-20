import uuid
import argparse
from pathlib import Path

import yaml
import torch
import joblib
from tqdm import tqdm
from sklearn import metrics

from frame.source import models, train
from torch_geometric.loader import DataLoader
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-c", "--config", dest="config", required=True)
    args = args_parser.parse_args()
    with open(args.config) as stream:
        params = yaml.safe_load(stream)

    config = params["Data"]
    tune = {}
    for name, bounds in params["Tune"].items():
        if isinstance(bounds["value"], int):
            tune[name] = int(bounds["value"])
        else:
            tune[name] = float(bounds["value"])

    path_checkpoint = config["path_checkpoint"]
    model_name = config.get("model", "gat").lower()
    batch_size = config.get("batch_size", 64)
    task = config.get("task", "classification").lower()

    # * Initialize
    name = config["name"]
    if name.lower() == "none":
        name = str(uuid.uuid4()).split("-")[0]
        config["name"] = name

    # * Load dataset
    path_joblib = Path(config["path_joblib"])
    data = joblib.load(path_joblib)
    dataset = data["dataset"]
    tune["feat_size"] = data["metadata"]["feat_size"]
    tune["edge_dim"] = data["metadata"]["edge_dim"]
    tune["bce_weight"] = data["metadata"]["bce_weight"]

    # * Prepare dataloader
    test_data = [data for data in dataset if data.set == "test"]
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # * Get checkpoint and prepare Explainer
    model = models.select_model(model_name, tune)
    model.load_state_dict(torch.load(path_checkpoint))
    model.eval()

    agg_pred = []
    agg_lbl = []
    agg_true = []
    for data in tqdm(test_loader, ncols=120, desc="Explaining"):
        data.to(device)

        # * Make predictions
        model_out = model(x=data.x.float(),
                          edge_index=data.edge_index,
                          edge_attr=data.edge_attr.float(),
                          batch=data.batch)

        # * Read prediction values
        if task == "classification":
            detach = torch.sigmoid(model_out).cpu().detach()
            pred = list(torch.ravel(detach).cpu().detach().numpy())
            pred_lbl = (detach >= 0.5).int()
            agg_lbl += pred_lbl.flatten().tolist()
        else:
            detach = model_out.cpu().detach()
            pred = list(torch.ravel(detach).cpu().detach().numpy())
            pred_lbl = None

        # * Save prediction values
        agg_pred += pred
        agg_true += data.y.flatten().tolist()

    # * Get metrics
    if task == "classification":
        acc = metrics.accuracy_score(agg_true, agg_lbl)
        acc_bal = metrics.balanced_accuracy_score(agg_true, agg_lbl)
        f1 = metrics.f1_score(agg_true, agg_lbl, zero_division=0)
        prec = metrics.precision_score(agg_true, agg_lbl, zero_division=0)
        rec = metrics.recall_score(agg_true, agg_lbl, zero_division=0)
        mcc = metrics.matthews_corrcoef(agg_true, agg_lbl)
        roc_auc = metrics.roc_auc_score(agg_true, agg_pred)
        avg_prec = metrics.average_precision_score(agg_true, agg_pred)

        print(f"\n========= {name}"
              f"\n{'Accuracy:':<19}{round(acc, 3)}"
              f"\n{'Balanced Accuracy:':<19}{round(acc_bal, 3)}"
              f"\n{'F1:':<19}{round(f1, 3)}"
              f"\n{'MCC:':<19}{round(mcc, 3)}"
              f"\n{'Precision:':<19}{round(prec, 3)}"
              f"\n{'Recall:':<19}{round(rec, 3)}"
              f"\n{'Avg. Precision:':<19}{round(avg_prec, 3)}"
              f"\n{'ROC-AUC:':<19}{round(roc_auc, 3)}\n")

    else:
        r2 = metrics.r2_score(agg_true, agg_pred)
        rmse = metrics.root_mean_squared_error(agg_true, agg_pred)
        mae = metrics.mean_absolute_error(agg_true, agg_pred)

        rto_r2, _ = train.reg_through_origin(agg_true, agg_pred)
        ccc = train.concordance_correlation(agg_true, agg_pred)
        roy_c = train.roy_criteria(agg_true, agg_pred, inverse=False)
        roy_c_inv = train.roy_criteria(agg_true, agg_pred, inverse=True)
        delta = train.golbraikh_tropsha(agg_true, agg_pred)

        print(f"\n========= {name}"
              f"\n{'R²:':<19}{round(r2, 3)}"
              f"\n{'RMSE:':<19}{round(rmse, 3)}"
              f"\n{'MAE:':<19}{round(mae, 3)}"
              f"\n{'RTO R²:':<19}{round(rto_r2, 3)}"
              f"\n{'CCC:':<19}{round(ccc, 3)}"
              f"\n{'Roy Criteria:':<19}{round(roy_c, 3)}"
              f"\n{'Roy C. Inverse:':<19}{round(roy_c_inv, 3)}"
              f"\n{'Delta:':<19}{round(delta, 3)}\n")
