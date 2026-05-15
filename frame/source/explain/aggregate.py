from typing import Sequence

import numpy as np
import torch
from torch_geometric.data.data import Data


def aggregate_atom_mask(atom_mask: np.ndarray,
                        atom_to_fragment: Sequence[int],
                        n_fragments: int):
    """Sum a per-atom attribution vector into per-fragment scores.

    Args:
        atom_mask: 2-D array of shape (n_atoms, n_features) or 1-D
            array of length n_atoms. The feature axis is preserved
            so the result is shape-compatible with native fragment-level
            attributions.
        atom_to_fragment: Sequence of length n_atoms where
            atom_to_fragment[i] is the fragment index of atom i.
        n_fragments: Total number of BRICS fragments.

    Returns:
        Numpy array of shape (n_fragments, n_features) if
        atom_mask was 2-D, otherwise shape (n_fragments,).

    Raises:
        ValueError: If the lengths of atom_mask and
            atom_to_fragment disagree.
    """
    arr = np.asarray(atom_mask, dtype=float)
    n_atoms = arr.shape[0]
    if n_atoms != len(atom_to_fragment):
        raise ValueError(f"atom_mask has {n_atoms} rows but "
                         f"atom_to_fragment has {len(atom_to_fragment)} "
                         "entries")

    if arr.ndim == 1:
        out = np.zeros(n_fragments, dtype=float)
        for atom_idx, frag_idx in enumerate(atom_to_fragment):
            out[frag_idx] += arr[atom_idx]
        return out

    out = np.zeros((n_fragments, arr.shape[1]), dtype=float)
    for atom_idx, frag_idx in enumerate(atom_to_fragment):
        out[frag_idx] += arr[atom_idx]
    return out


def build_smiles_index(decompose_dataset):
    """Build a SMILES → (atom_map, fragments) lookup over a dataset.

    Used by the aggregated-baseline pipeline to translate an atom-level
    Data object into the matching fragment-level structure. Matching is
    done on the canonical SMILES string stored on every Data object.

    Args:
        decompose_dataset: A :class:`DecomposeDataset` or any iterable
            of Data objects exposing smiles, atom_map, and
            frag attributes.

    Returns:
        Dict mapping SMILES string to
        {"atom_to_fragment": list[int], "fragments": list[str],
        "n_fragments": int}.
    """
    index = {}
    for data in decompose_dataset:
        smiles = getattr(data, "smiles", None)
        if smiles is None:
            continue
        atom_keys, frag_values = data.atom_map[0], data.atom_map[1]
        mapping = [0] * (max(atom_keys) + 1)
        for atom_idx, frag_idx in zip(atom_keys, frag_values):
            mapping[atom_idx] = int(frag_idx)
        index[smiles] = {"atom_to_fragment": mapping,
                         "fragments": list(data.frag),
                         "n_fragments": len(data.frag)}
    return index


def aggregated_batch_masks(node_mask: torch.Tensor,
                           batch: torch.Tensor,
                           atom_graphs: Sequence[Data],
                           smiles_index: dict):
    """Aggregate a batched atom-level mask into per-fragment masks.

    Splits node_mask by the PyG batch vector, looks each graph up by
    SMILES in smiles_index, and aggregates with `aggregate_atom_mask`.
    Graphs whose SMILES is missing from the index (e.g. molecules with
    no BRICS bonds, which are filtered out of the fragment dataset)
    are returned as None placeholders so the caller can skip them.

    Args:
        node_mask: 2-D tensor of shape (total_atoms, n_features).
        batch: 1-D long tensor of length total_atoms mapping each
            atom to its graph index inside the batch.
        atom_graphs: Atom-level Data objects in batch order; used to
            look up SMILES and atom counts.
        smiles_index: Output of `build_smiles_index`.

    Returns:
        List with one entry per graph in the batch. Each entry is a dict
        with keys mask (numpy array, shape (n_fragments,
        n_features)), fragments (list of SMILES), or None for
        skipped graphs.
    """
    mask_np = node_mask.detach().cpu().numpy()
    batch_np = batch.detach().cpu().numpy()
    out = []
    for graph_idx, graph in enumerate(atom_graphs):
        smiles = getattr(graph, "smiles", None)
        record = smiles_index.get(smiles)
        if record is None:
            out.append(None)
            continue

        atom_rows = mask_np[batch_np == graph_idx]
        agg = aggregate_atom_mask(atom_rows,
                                  record["atom_to_fragment"],
                                  record["n_fragments"])
        out.append({"mask": agg,
                    "fragments": record["fragments"]})
    return out
