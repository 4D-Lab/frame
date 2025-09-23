import os
import uuid
import argparse
from pathlib import Path

import yaml
import torch
import joblib
from tqdm import tqdm
from torch_geometric.explain import Explainer, CaptumExplainer

from placeholder.source import explain, models

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

    checkpoint_path = config["checkpoint_path"]
    method = config.get("method", "ig").lower()
    model_name = config.get("model", "gat").lower()

    # * Initialize
    name = config["name"]
    if name.lower() == "none":
        name = str(uuid.uuid4()).split["-"][0]
        config["name"] = name

    cwd = Path(os.getcwd())
    project_dir = cwd / "output" / name
    image_dir = project_dir / "explain"
    os.makedirs(image_dir, exist_ok=True)

    # * Load dataset
    path_csv = Path(config["path_joblib"])
    data = joblib.load(path_csv)
    dataset = data["dataset"]
    tune["feat_size"] = data["metadata"]["feat_size"]
    tune["edge_dim"] = data["metadata"]["edge_dim"]
    tune["bce_weight"] = data["metadata"]["bce_weight"]

    # * Get checkpoint and prepare Explainer
    model = models.select_model(model_name, tune)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    mode = "multiclass_classification"
    if method.lower() == "ig":
        explainer = Explainer(model=model,
                              algorithm=CaptumExplainer("IntegratedGradients"),
                              explanation_type="model",
                              edge_mask_type="object",
                              node_mask_type="attributes",
                              model_config=dict(mode=mode,
                                                task_level="graph",
                                                return_type="raw"))

    elif method.lower() == "shapley":
        explainer = Explainer(model=model,
                              algorithm=CaptumExplainer(
                                  "ShapleyValueSampling"),
                              explanation_type="model",
                              edge_mask_type="object",
                              node_mask_type="attributes",
                              model_config=dict(mode=mode,
                                                task_level="graph",
                                                return_type="raw"))

    else:
        raise NotImplementedError("Method not availabe")

    for data in tqdm(dataset, ncols=120, desc="Explaining"):
        data.to(device)
        batch = torch.zeros(data.x.shape[0], dtype=int, device=device)

        # * Make predictions
        out = model(x=data.x.float(),
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr.float(),
                    batch=batch)

        # * Read prediction values
        detach = torch.sigmoid(out).cpu().detach()
        pred = list(torch.ravel(detach).cpu().detach().numpy())[0]

        # * Explain
        explanation = explainer(data.x.float(), data.edge_index,
                                edge_attr=data.edge_attr.float(),
                                batch=batch)

        explain.plot_explain(data, explanation, pred, image_dir, True)
