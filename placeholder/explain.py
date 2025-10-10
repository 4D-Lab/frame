import os
import uuid
import argparse
from pathlib import Path
from collections import defaultdict

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

    path_checkpoint = config["path_checkpoint"]
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
    model.load_state_dict(torch.load(path_checkpoint))
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

    top_k = defaultdict(float)
    bot_k = defaultdict(float)
    frag_top = defaultdict(float)
    frag_bot = defaultdict(float)
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
        node_mask = explanation.node_mask

        top, bot = explain.retrieve_info(data, node_mask, 10)
        for label, contrib in zip(top[0], top[1]):
            top_k[label] += abs(contrib)
        for label, contrib in zip(bot[0], bot[1]):
            bot_k[label] += abs(contrib)

        for label in top[3]:
            frag_top[label] += 1
        for label in bot[3]:
            frag_bot[label] += 1

        explain.plot_fragments(data, node_mask, image_dir, 10)
        explain.plot_importance(data, node_mask, image_dir, 10)
        explain.plot_explain(data, node_mask, pred, image_dir)

    # explain.plot_general(top_k, bot_k, image_dir)
    explain.plot_general(frag_top, frag_bot, image_dir)
