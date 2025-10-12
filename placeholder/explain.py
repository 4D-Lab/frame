import os
import uuid
import argparse
from pathlib import Path
from collections import Counter

import yaml
import torch
import joblib
from tqdm import tqdm
from torch_geometric.explain import Explainer, CaptumExplainer

from placeholder.source import explain, models
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

    # * Initialize
    name = config["name"]
    if name.lower() == "none":
        name = str(uuid.uuid4()).split["-"][0]
        config["name"] = name

    cwd = Path(os.getcwd())
    project_dir = cwd / "output" / name
    out = project_dir / "explain"
    os.makedirs(out, exist_ok=True)

    # * Load dataset
    path_joblib = Path(config["path_joblib"])
    data = joblib.load(path_joblib)
    dataset = data["dataset"]
    tune["feat_size"] = data["metadata"]["feat_size"]
    tune["edge_dim"] = data["metadata"]["edge_dim"]
    tune["bce_weight"] = data["metadata"]["bce_weight"]
    loader = data["metadata"]["loader"]

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=4,
                            persistent_workers=True)

    # * Get checkpoint and prepare Explainer
    model = models.select_model(model_name, tune)
    model.load_state_dict(torch.load(path_checkpoint))
    model.eval()

    explainer = Explainer(model=model,
                          algorithm=CaptumExplainer("IntegratedGradients"),
                          explanation_type="model",
                          edge_mask_type="object",
                          node_mask_type="attributes",
                          model_config=dict(mode="multiclass_classification",
                                            task_level="graph",
                                            return_type="raw"))

    # Create counters
    count_frag = {"0_0": {0: Counter(), 1: Counter()},
                  "0_1": {0: Counter(), 1: Counter()},
                  "1_1": {0: Counter(), 1: Counter()},
                  "1_0": {0: Counter(), 1: Counter()}}

    count_label = {"0_0": {0: Counter(), 1: Counter()},
                   "0_1": {0: Counter(), 1: Counter()},
                   "1_1": {0: Counter(), 1: Counter()},
                   "1_0": {0: Counter(), 1: Counter()}}

    for data in tqdm(dataloader, ncols=120, desc="Explaining"):
        data.to(device)

        # * Make predictions
        model_out = model(x=data.x.float(),
                          edge_index=data.edge_index,
                          edge_attr=data.edge_attr.float(),
                          batch=data.batch)

        # * Read prediction values
        detach = torch.sigmoid(model_out).cpu().detach()
        pred_lbl = (detach >= 0.5).int()
        pred = list(torch.ravel(detach).cpu().detach().numpy())

        # * Explain
        explanation = explainer(data.x.float(), data.edge_index,
                                edge_attr=data.edge_attr.float(),
                                batch=data.batch)

        mol_exp = explain.MolExplain(explanation, pred, pred_lbl, loader, out)
        mol_exp.retrieve_info(data, count_frag, count_label)
        mol_exp.plot_explanations(data)

    explain.plot_counters(count_frag, out, "frag")
    explain.plot_counters(count_label, out, "label")
