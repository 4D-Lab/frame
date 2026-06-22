import json
from typing import Callable, Sequence

import torch
import numpy as np
from rdkit import Chem
import matplotlib as mpl
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize, TwoSlopeNorm

from frame.source.explain import metrics_explain
from frame.source.explain.metrics_explain import (fragment_scores,
                                                  fragment_hit_rate,
                                                  mean_gini,
                                                  spearman_cross_explainer,
                                                  top_fragment)

mpl.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V1 = ["atom_C", "atom_N", "atom_O", "atom_F", "atom_P",
      "atom_S", "atom_Cl", "atom_Br", "atom_I", "atom_R",
      "degree_0", "degree_1", "degree_2", "degree_3",
      "degree_4", "degree_5", "charge", "radical",
      "hybridization_S", "hybridization_SP", "hybridization_SP2",
      "hybridization_SP3", "hybridization_SP3D", "hybridization_SP3D2",
      "other", "aromaticity", "TotalNumHs_0", "TotalNumHs_1",
      "TotalNumHs_2", "TotalNumHs_3", "TotalNumHs_4", "chirality",
      "chirality_R", "chirality_S"]


class MolExplain:
    """Driver for per-molecule explanation artefacts and metrics.

    The constructor wires up output paths and metric accumulators.
    `process_batch` is called once per minibatch by the CLI driver
    and handles native attributions (atom-level or fragment-level).
    `process_aggregated_batch` consumes pre-aggregated fragment
    masks from the aggregated atom-level baseline. `finalize`
    writes explain_metrics.json.

    Args:
        loader: "default" for atom-level data or "decompose" for
            fragment-level data. Controls CSV header and visualisation
            branch.
        out_dir: Directory where predictions.csv, SVGs, and
            explain_metrics.json are written.
        algorithm: Label included in the metrics JSON for downstream
            cross-explainer comparison. Typically "ig" or
            "gnnex".
        k: Number of top features per fragment for the optional bar
            plot. Currently unused in CSV output.
    """

    def __init__(self, loader: str, out_dir, algorithm: str = "ig",
                 k: int = 10):
        self.loader = loader
        self.out = out_dir
        self.algorithm = algorithm
        self.k = k
        self.cut = k // 2
        self.labels = np.array(V1)
        self.records = []
        self._init_predictions_file()

    def _init_predictions_file(self):
        """Write the predictions CSV header (overwriting any prior file)."""
        if self.loader == "default":
            header = "id,smiles,real,pred_label,pred,fragment\n"
        else:
            labels = ",".join(V1)
            header = "id,smiles,real,pred_label,pred," + labels + "\n"
        with open(self.out / "predictions.csv", "w") as f:
            f.write(header)

    def process_batch(self, explanation, logit, pred, pred_lbl, graphs):
        """Consume one minibatch's native explanation.

        Splits the batched attribution by graph, writes one row per
        atom-level molecule or one row per fragment per fragment-level
        molecule to predictions.csv, and appends one record per
        fragment-level molecule to :attr:`records` for later metric
        aggregation. Visualisation is delegated to
        `plot_explanations`.

        Args:
            explanation: A PyG Explanation with attributes
                node_mask and batch.
            logit: Per-graph raw model output (list).
            pred: Per-graph post-sigmoid probability (classification)
                or raw output (regression).
            pred_lbl: Per-graph predicted hard label
                (classification) or list of None (regression).
            graphs: Iterable of Data objects in batch order.
        """
        mask = explanation.node_mask.detach().cpu()
        batch = explanation.batch.detach().cpu()
        if self.loader == "default":
            self._info_atom(mask, batch, logit, pred, pred_lbl, graphs)
        else:
            self._info_fragment(mask, batch, logit, pred, pred_lbl, graphs)
        self.plot_explanations(mask, batch, logit, pred, pred_lbl, graphs)

    def process_aggregated_batch(self, agg_results, logit, pred,
                                 pred_lbl, graphs):
        """Consume one minibatch's aggregated atom-level baseline.

        agg_results is the output of
        `frame.source.explain.aggregate.aggregated_batch_masks`,
        one dict (or None for skipped graphs) per atom-level graph.
        For each non-skipped graph this writes the same predictions row
        layout as `_info_fragment`, appends an aggregated record,
        and renders the SVG via the fragment-level visualiser.

        Args:
            agg_results: List from
                `aggregate.aggregated_batch_masks`.
            logit: Per-graph raw model output.
            pred: Per-graph probability (cls) or raw output (reg).
            pred_lbl: Per-graph predicted label or list of None.
            graphs: Atom-level Data objects in batch order.
        """
        for idx, record in enumerate(agg_results):
            if record is None:
                continue
            data = graphs[idx]
            mask = record["mask"]
            fragments = record["fragments"]
            label_pred = (pred_lbl[idx].numpy()[0]
                          if pred_lbl[idx] is not None else "")
            self._write_fragment_row(data, mask, fragments,
                                     pred[idx], label_pred)
            self._append_record(data, mask, fragments)
            self._plot_aggregated(data, mask, pred[idx], logit[idx],
                                  pred_lbl[idx])

    def _info_atom(self, mask, batch, logit, pred, pred_lbl, graphs):
        """Write one CSV row per atom-level molecule."""
        batch_num = batch.unique()
        masks = [mask[batch == b] for b in batch_num]
        for idx in range(len(masks)):
            data = graphs[idx]
            real_label = int(data.y.cpu().numpy()[0])
            label_pred = (pred_lbl[idx].numpy()[0]
                          if pred_lbl[idx] is not None else "")
            text = (f"{data.idx},{data.smiles},{real_label},"
                    f"{label_pred},{pred[idx]:.3f}\n")
            with open(self.out / "predictions.csv", "a") as f:
                f.writelines(text)

    def _info_fragment(self, mask, batch, logit, pred, pred_lbl, graphs):
        """Write one CSV row per fragment and accumulate scores."""
        batch_num = batch.unique()
        masks = [mask[batch == b] for b in batch_num]
        for idx, node_mask in enumerate(masks):
            data = graphs[idx]
            fragments = list(np.array(data.frag))
            label_pred = (pred_lbl[idx].numpy()[0]
                          if pred_lbl[idx] is not None else "")
            self._write_fragment_row(data, node_mask.cpu().numpy(),
                                     fragments, pred[idx], label_pred)
            self._append_record(data, node_mask.cpu().numpy(), fragments)

    def _write_fragment_row(self, data, mask_2d, fragments, pred_val,
                            label_pred):
        """Append one row per fragment to predictions.csv."""
        real_label = int(data.y.cpu().numpy()[0])
        mask_list = [[f"{m:.3f}" for m in row] for row in mask_2d]
        text = []
        for row, frag in zip(mask_list, fragments):
            contribs = ",".join(row)
            text.append(f"{data.idx},{data.smiles},{real_label},"
                        f"{label_pred},{pred_val:.3f},{frag},"
                        f"{contribs}\n")
        with open(self.out / "predictions.csv", "a") as f:
            f.writelines(text)

    def _append_record(self, data, mask_2d, fragments):
        """Accumulate one molecule's per-fragment scores for metrics."""
        scores = fragment_scores(np.asarray(mask_2d))
        self.records.append({"idx": str(data.idx),
                             "smiles": data.smiles,
                             "fragments": list(fragments),
                             "scores": scores,
                             "top_fragment": top_fragment(scores,
                                                          fragments)})

    def plot_explanations(self, mask, batch, logit, pred, pred_lbl,
                          graphs):
        """Render per-molecule SVGs for the current minibatch."""
        batch_num = batch.unique()
        masks = [mask[batch == b] for b in batch_num]
        for idx, node_mask in enumerate(masks):
            data = graphs[idx]
            name = data.idx
            label_pred = (pred_lbl[idx].numpy()[0]
                          if pred_lbl[idx] is not None else None)
            if self.loader == "default":
                self._explain_atom(data, node_mask, pred[idx],
                                   logit[idx], label_pred, name)
            else:
                self._explain_frag(data, node_mask, pred[idx],
                                   logit[idx], label_pred, name)

    def _plot_aggregated(self, data, mask_2d, pred_val, logit_val,
                         pred_lbl_t):
        """Render an SVG for the aggregated baseline on an atom-level graph.

        Uses the same fragment-level visualisation as the native model,
        but takes the atom-to-fragment map from the matching fragment
        record instead of data.atom_map.
        """
        atom_map = getattr(data, "agg_atom_map", None)
        if atom_map is None:
            return
        scores = fragment_scores(np.asarray(mask_2d))
        scores = self._rescale_mask(scores, logit_val)
        scores = np.round(scores, 3)
        label_pred = (pred_lbl_t.numpy()[0]
                      if pred_lbl_t is not None else None)
        self._draw_fragment_svg(data, scores, atom_map, pred_val,
                                logit_val, label_pred, data.idx)

    @staticmethod
    def _rescale_mask(mask_atom: np.ndarray, logit):
        """Rescale per-atom attributions so they sum to the raw logit."""
        current_sum = mask_atom.sum()
        if current_sum == 0:
            return mask_atom
        return mask_atom * (float(logit) / current_sum)

    def _explain_atom(self, data, node_mask, pred, logit,
                      pred_label, name):
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles)
        mask_atom = torch.sum(node_mask, dim=1).numpy()
        mask_atom = self._rescale_mask(mask_atom, logit)
        mask_atom = np.round(mask_atom, 3)
        cmap = self._make_cmap(mask_atom)

        highlight_node = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            rgb = cmap.to_rgba(mask_atom[idx])[:-1]
            highlight_node[idx] = [rgb]
            atom.SetProp("atomNote", str(mask_atom[idx]))

        legend = self._format_legend(data, smiles, pred, logit,
                                     pred_label, name)
        self._write_svg(mol, legend, highlight_node, name)

    def _explain_frag(self, data, node_mask, pred, logit, pred_label,
                      name):
        atom_map = dict(zip(data.atom_map[0], data.atom_map[1]))
        scores = fragment_scores(node_mask.numpy())
        scores = self._rescale_mask(scores, logit)
        scores = np.round(scores, 3)
        self._draw_fragment_svg(data, scores, atom_map, pred, logit,
                                pred_label, name)

    def _draw_fragment_svg(self, data, frag_scores, atom_map, pred,
                           logit, pred_label, name):
        """Draw a 2-D molecule with atoms coloured by parent fragment."""
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles)
        scores = np.round(np.asarray(frag_scores, dtype=float), 3)
        cmap = self._make_cmap(scores)

        highlight_node = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            frag_val = scores[atom_map[idx]]
            rgb = cmap.to_rgba(frag_val)[:-1]
            highlight_node[idx] = [rgb]
            atom.SetProp("atomNote", str(frag_val))

        legend = self._format_legend(data, smiles, pred, logit,
                                     pred_label, name)
        self._write_svg(mol, legend, highlight_node, name)

    @staticmethod
    def _make_cmap(values: np.ndarray):
        """Build a matplotlib colormap suited to the sign of values."""
        arr = np.asarray(values).ravel()
        if arr.size == 0:
            return mpl.cm.ScalarMappable(norm=Normalize(0, 1),
                                         cmap=mpl.cm.Blues)
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmin >= 0:
            return mpl.cm.ScalarMappable(norm=Normalize(vmin=0,
                                                        vmax=vmax * 1.3
                                                        or 1.0),
                                         cmap=mpl.cm.Blues)
        if vmax <= 0:
            return mpl.cm.ScalarMappable(norm=Normalize(vmin=vmin * 1.3,
                                                        vmax=0),
                                         cmap=mpl.cm.Oranges_r)
        pos_colors = plt.cm.Blues(np.linspace(0, 1, 128))
        neg_colors = plt.cm.Oranges_r(np.linspace(0, 1, 128))
        combined = np.vstack((neg_colors, pos_colors))
        color = LinearSegmentedColormap.from_list("OrBu", combined)
        norm = TwoSlopeNorm(vmin=vmin * 1.3, vcenter=0, vmax=vmax * 1.3)
        return mpl.cm.ScalarMappable(norm=norm, cmap=color)

    @staticmethod
    def _format_legend(data, smiles, pred, logit, pred_label, name):
        if pred_label is None:
            return (f"Graph ID: {name}\n{smiles}\n"
                    f"Prediction: {pred:.3f}\tTrue: {float(data.y):.3f}")
        return (f"Graph ID: {name}\n{smiles}\n"
                f"Prediction: {pred:.3f}\tLogits: {logit:.3f}\t|\t"
                f"Class: {pred_label}\tTrue: {int(data.y)}")

    def _write_svg(self, mol, legend, highlight_node, name):
        drawer = rdMolDraw2D.MolDraw2DSVG(1200, 800)
        opts = drawer.drawOptions()
        opts.fillHighlights = True
        opts.annotationFontScale = 0.5
        opts.legendFontSize = 25
        drawer.DrawMoleculeWithHighlights(mol, legend, highlight_node,
                                          {}, {}, {})
        drawer.FinishDrawing()
        with open(self.out / f"{name}.svg", "w") as f:
            f.write(drawer.GetDrawingText())

    def finalize(self, classifier: Callable = None,
                 class_names: Sequence[str] = None):
        """Write explain_metrics.json summarising accumulated records.

        Computes the mean Gini coefficient of fragment-importance
        distributions and, if a pharmacophore classifier and
        class_names are supplied, the fragment hit rate with a
        bootstrap 95% CI and a per-class breakdown.

        Args:
            classifier: Optional callable mapping fragment SMILES to a
                class name string (or None).
            class_names: Optional iterable of all valid class names for
                the per-class breakdown.

        Returns:
            The metrics dict that was written to disk.
        """
        score_vecs = [r["scores"] for r in self.records]
        tops = [r["top_fragment"] for r in self.records
                if r["top_fragment"] is not None]
        metrics = {"algorithm": self.algorithm,
                   "loader": self.loader,
                   "n_molecules": len(self.records),
                   "mean_gini": mean_gini(score_vecs)}
        if classifier is not None and class_names is not None:
            metrics["fragment_hit_rate"] = fragment_hit_rate(
                tops, classifier, class_names)
        with open(self.out / "explain_metrics.json", "w") as fh:
            json.dump(metrics, fh, indent=2)
        return metrics


def spearman_between_runs(records_a: Sequence[dict],
                          records_b: Sequence[dict],
                          out_path):
    """Compute per-molecule Spearman rho between two explainer runs.

    The two record lists must come from the same dataset (same SMILES
    set) so that fragment vectors can be aligned. Molecules present in
    only one run are skipped.

    Args:
        records_a: MolExplain.records from explainer A.
        records_b: MolExplain.records from explainer B.
        out_path: Destination JSON path. Parent directory must exist.

    Returns:
        The metrics dict that was written to disk.
    """
    by_smiles_b = {r["smiles"]: r for r in records_b}
    aligned_a, aligned_b = [], []
    for r in records_a:
        match = by_smiles_b.get(r["smiles"])
        if match is None:
            continue
        if len(match["scores"]) != len(r["scores"]):
            continue
        aligned_a.append(r["scores"])
        aligned_b.append(match["scores"])

    summary = spearman_cross_explainer(aligned_a, aligned_b)
    summary["n_overlap"] = len(aligned_a)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


__all__ = ["MolExplain",
           "spearman_between_runs",
           "metrics_explain",
           "V1"]
