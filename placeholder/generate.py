import os
import uuid
import shutil
import argparse
from pathlib import Path

import yaml
import torch
import joblib

from placeholder.source import datasets

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    elif loader == "decompose_v2":
        dataset = datasets.DecomposeDataset_v2(path_csv)
    else:
        raise NotImplementedError("Loader not available")

    # * Export
    bce_weight = (len(dataset.y) - sum(dataset.y)) / sum(dataset.y)
    metadata = {"feat_size": dataset.num_node_features,
                "edge_dim": dataset.num_edge_features,
                "bce_weight": bce_weight,
                "loader": loader,
                "project_dir": project_dir}

    dump_data = {"dataset": dataset, "metadata": metadata}
    joblib.dump(dump_data, project_dir / "data.joblib")

    if os.path.isdir(cwd / "???"):
        shutil.rmtree(cwd / "???")
