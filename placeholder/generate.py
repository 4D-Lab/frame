import os
import shutil
import joblib
import argparse
from pathlib import Path

import yaml
import torch

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
    file_path = params["Data"].get("file_path", None)
    loader = params["Data"].get("loader", "default").lower()

    # * Initialize
    cwd = Path(os.getcwd())
    project_dir = cwd / "output" / name
    os.makedirs(project_dir, exist_ok=True)

    # * Create dataset
    if loader == "default":
        dataset = datasets.MolecularDataset(file_path)
    elif loader == "brics":
        dataset = datasets.MolecularBRICSDataset(file_path)
    else:
        raise NotImplementedError("Loader not available")

    # * Export
    bce_weight = (len(dataset.y) - sum(dataset.y)) / sum(dataset.y)
    metadata = {"feat_size": dataset.num_node_features,
                "edge_dim": dataset.num_edge_features,
                "bce_weight": bce_weight,
                "project_dir": project_dir}

    dump_data = {"dataset": dataset, "metadata": metadata}
    joblib.dump(dump_data, project_dir / "data.joblib")

    if os.path.isdir(cwd / "???"):
        shutil.rmtree(cwd / "???")
