import os
import uuid
import argparse
from pathlib import Path

import yaml
import torch
import joblib
from tqdm import tqdm
from torch_geometric.explain import (Explainer,
                                     CaptumExplainer,
                                     GNNExplainer)
from torch_geometric.loader import DataLoader

from frame.source import explain, models
from frame.source.explain import aggregate
from frame.source.explain import pharmacophores

device = "cuda" if torch.cuda.is_available() else "cpu"

ALGORITHMS = ("ig", "gnnex")
BASELINES = ("native", "aggregated")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config", required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="ig",
                        help="Attribution algorithm to use.")
    parser.add_argument("--baseline", choices=BASELINES, default="native",
                        help="`native` explains the loader in metadata; "
                             "`aggregated` runs the atom-level model and "
                             "aggregates attributions per BRICS fragment.")
    parser.add_argument("--pharmacophore", default=None,
                        choices=list(pharmacophores.CLASSIFIERS),
                        help="Case study for fragment hit rate.")
    parser.add_argument("--gnnex-epochs", type=int, default=100,
                        help="Epochs for GNNExplainer mask optimisation.")
    return parser.parse_args()


def _resolve_name(config: dict):
    name = config.get("name", "none")
    if name.lower() == "none":
        name = str(uuid.uuid4()).split("-")[0]
        config["name"] = name
    return name


def _build_algorithm(algorithm: str, epochs: int):
    if algorithm == "ig":
        return CaptumExplainer("IntegratedGradients")
    return GNNExplainer(epochs=epochs)


def _build_explainer(model, algorithm: str, task: str, epochs: int):
    mode = ("multiclass_classification"
            if task == "classification" else "regression")
    return Explainer(model=model,
                     algorithm=_build_algorithm(algorithm, epochs),
                     explanation_type="model",
                     edge_mask_type="object",
                     node_mask_type="attributes",
                     model_config=dict(mode=mode,
                                       task_level="graph",
                                       return_type="raw"))


def _load_artefacts(config: dict, joblib_key: str, checkpoint_key: str):
    """Load a dataset joblib + trained checkpoint by config key.

    Args:
        config: params["Data"] block.
        joblib_key: Key into config pointing to the dataset joblib.
        checkpoint_key: Key into config pointing to the model
            checkpoint.

    Returns:
        Dict with keys dataset, metadata, state_dict.

    Raises:
        KeyError: If either key is missing.
        FileNotFoundError: If either path does not exist.
    """
    joblib_path = Path(config[joblib_key])
    ckpt_path = Path(config[checkpoint_key])
    data = joblib.load(joblib_path)
    state = torch.load(ckpt_path, map_location=device)
    return {"dataset": data["dataset"],
            "metadata": data["metadata"],
            "state_dict": state}


def _model_config(tune: dict, metadata: dict, task: str, params: dict):
    """Merge the fixed tune config with dataset metadata for model init."""
    cfg = dict(tune)
    cfg["feat_size"] = metadata["feat_size"]
    cfg["edge_dim"] = metadata["edge_dim"]
    cfg["bce_weight"] = metadata["bce_weight"]
    cfg["task"] = task
    cfg["regression_loss"] = params["Data"].get("regression_loss", "mse")
    return cfg


def _read_predictions(task: str, model_out):
    """Return (logit_list, pred, pred_lbl) for a forward pass."""
    if task == "classification":
        logit = model_out.cpu().detach()
        logit_list = list(torch.ravel(logit).numpy())
        detach = torch.sigmoid(logit)
        pred = list(torch.ravel(detach).numpy())
        pred_lbl = (detach >= 0.5).int()
    else:
        detach = model_out.cpu().detach()
        pred = list(torch.ravel(detach).numpy())
        logit_list = pred
        pred_lbl = [None] * detach.shape[0]
    return logit_list, pred, pred_lbl


def _run_native(model, dataloader, explainer, task: str,
                mol_exp: explain.MolExplain):
    """Iterate batches, run explainer, hand off to MolExplain."""
    for batch in tqdm(dataloader, ncols=120, desc="Explaining"):
        batch.to(device)
        model_out = model(x=batch.x.float(),
                          edge_index=batch.edge_index,
                          edge_attr=batch.edge_attr.float(),
                          batch=batch.batch)
        logit_list, pred, pred_lbl = _read_predictions(task, model_out)
        explanation = explainer(batch.x.float(), batch.edge_index,
                                edge_attr=batch.edge_attr.float(),
                                batch=batch.batch)
        mol_exp.process_batch(explanation, logit_list, pred, pred_lbl,
                              batch.to_data_list())


def _run_aggregated(model, dataloader, explainer, task: str,
                    smiles_index: dict, mol_exp: explain.MolExplain):
    """Aggregate atom-level attributions into fragment-level scores."""
    for batch in tqdm(dataloader, ncols=120, desc="Aggregating"):
        batch.to(device)
        model_out = model(x=batch.x.float(),
                          edge_index=batch.edge_index,
                          edge_attr=batch.edge_attr.float(),
                          batch=batch.batch)
        logit_list, pred, pred_lbl = _read_predictions(task, model_out)
        explanation = explainer(batch.x.float(), batch.edge_index,
                                edge_attr=batch.edge_attr.float(),
                                batch=batch.batch)
        graphs = batch.to_data_list()
        for graph in graphs:
            record = smiles_index.get(getattr(graph, "smiles", None))
            if record is not None:
                graph.agg_atom_map = dict(enumerate(
                    record["atom_to_fragment"]))
        agg = aggregate.aggregated_batch_masks(explanation.node_mask,
                                               explanation.batch,
                                               graphs, smiles_index)
        mol_exp.process_aggregated_batch(agg, logit_list, pred,
                                         pred_lbl, graphs)


def main():
    args = _parse_args()
    with open(args.config) as stream:
        params = yaml.safe_load(stream)

    config = params["Data"]
    name = _resolve_name(config)
    cwd = Path(os.getcwd())
    out_root = cwd / "output" / name / "explain" / args.algorithm
    out_dir = out_root / args.baseline
    os.makedirs(out_dir, exist_ok=True)

    tune = models.tune_fixed(params)
    task = config.get("task", "classification").lower()
    batch_size = int(config.get("batch_size", 64))
    model_name = config.get("model", "gat").lower()

    primary = _load_artefacts(config, "path_joblib", "path_checkpoint")
    cfg = _model_config(tune, primary["metadata"], task, params)
    model = models.select_model(model_name, cfg)
    model.load_state_dict(primary["state_dict"])
    model.eval()

    explainer = _build_explainer(model, args.algorithm, task,
                                 args.gnnex_epochs)
    dataset = primary["dataset"]
    loader = primary["metadata"]["loader"]
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=4,
                            persistent_workers=True)

    if args.baseline == "aggregated":
        if loader != "default":
            raise ValueError("aggregated baseline requires path_joblib to "
                             "point at the atom-level dataset (loader="
                             "default); got loader=" + loader)
        frag_path = config.get("path_joblib_frag")
        if frag_path is None:
            raise ValueError("aggregated baseline requires "
                             "Data.path_joblib_frag in the config.")
        frag_data = joblib.load(Path(frag_path))
        smiles_index = aggregate.build_smiles_index(frag_data["dataset"])
        mol_exp = explain.MolExplain("decompose", out_dir,
                                     algorithm=args.algorithm)
        _run_aggregated(model, dataloader, explainer, task,
                        smiles_index, mol_exp)
    else:
        mol_exp = explain.MolExplain(loader, out_dir,
                                     algorithm=args.algorithm)
        _run_native(model, dataloader, explainer, task, mol_exp)

    classifier = None
    class_names = None
    if args.pharmacophore is not None:
        classifier = pharmacophores.get_classifier(args.pharmacophore)
        class_names = pharmacophores.get_class_names(args.pharmacophore)
    mol_exp.finalize(classifier=classifier, class_names=class_names)

    # Cross-explainer Spearman: if the *other* algorithm has already
    # produced records under the same baseline directory, write a
    # summary JSON next to this run.
    other = "gnnex" if args.algorithm == "ig" else "ig"
    other_dir = cwd / "output" / name / "explain" / other / args.baseline
    other_records = other_dir / "records.npz"
    if other_records.exists():
        peer = _load_records_npz(other_records)
        spearman_path = out_dir / "cross_explainer_spearman.json"
        explain.spearman_between_runs(mol_exp.records, peer,
                                      spearman_path)
        explain.spearman_between_runs(peer, mol_exp.records,
                                      other_dir
                                      / "cross_explainer_spearman.json")
    _dump_records_npz(mol_exp.records, out_dir / "records.npz")


def _dump_records_npz(records, path):
    """Persist accumulated per-molecule scores for later Spearman pairing.

    Stores SMILES, fragments, and the 1-D scores vector per molecule in a
    single .npz archive (one entry per molecule, plus a manifest list of
    SMILES). This is the minimum needed to recompute cross-explainer
    Spearman after the fact, without redoing the attribution run.
    """
    import numpy as np
    smiles = [r["smiles"] for r in records]
    payload = {"smiles": np.array(smiles, dtype=object)}
    for i, r in enumerate(records):
        payload[f"scores_{i}"] = np.asarray(r["scores"], dtype=float)
        payload[f"frags_{i}"] = np.array(r["fragments"], dtype=object)
        payload[f"top_{i}"] = np.array(
            "" if r["top_fragment"] is None else r["top_fragment"],
            dtype=object)
    np.savez(path, **payload)


def _load_records_npz(path):
    """Load records saved by `_dump_records_npz`."""
    import numpy as np
    arch = np.load(path, allow_pickle=True)
    smiles = list(arch["smiles"])
    records = []
    for i, smi in enumerate(smiles):
        records.append({"smiles": str(smi),
                        "scores": arch[f"scores_{i}"],
                        "fragments": list(arch[f"frags_{i}"]),
                        "top_fragment": (str(arch[f"top_{i}"])
                                         if str(arch[f"top_{i}"]) != ""
                                         else None)})
    return records
