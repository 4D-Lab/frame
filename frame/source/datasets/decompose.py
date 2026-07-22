import re
import random

import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

from frame.source.datasets.ertl import get_map_ertl
from frame.source.datasets.brics import get_map_brics
from frame.source.datasets.frag_edges import edge_dim
from frame.source.datasets.features import ATOM_FEAT_DIM, atom_features

random.seed(8)
np.random.seed(8)
torch.manual_seed(8)
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr,
                                      GlobalStorage, Data])

# Available decomposition backends
_DECOMPOSERS = {"brics": get_map_brics, "ertl": get_map_ertl}


class DecomposeDataset(InMemoryDataset):
    """Fragment-level molecular dataset with edge features.

    Fragments become graph nodes; edges carry the chemistry of the
    bond(s) connecting two fragments (see frag_edges.py). Two
    backends are available via method:

    - "ertl": ring-aware Ertl functional-group decomposition. Rings
      are never cut and functional groups keep their carbons, so
      fragments stay matchable against pharmacophore SMARTS, and
      molecules are essentially never excluded.
    - "brics": original BRICS retrosynthetic fragmentation;
      molecules with no BRICS-cleavable bond are skipped.

    The SMILES and ids of skipped molecules are kept on the dataset so
    frame.generate can report the exclusion rate.

    Attributes:
        excluded_smiles: SMILES strings of molecules that could not be
            decomposed by the chosen backend.
        excluded_ids: Aligned list of dataset ids.
        n_total: Total number of input rows (incl. excluded).
    """

    def __init__(self, path: str, method: str = "brics",
                 extended_edges: bool = True,
                 transform=None, pre_transform=None):
        self.path = path
        if method not in _DECOMPOSERS:
            raise ValueError(f"method must be one of {list(_DECOMPOSERS)}, "
                             f"got {method!r}")
        self.method = method
        self.extended_edges = extended_edges
        self._get_map = _DECOMPOSERS[method]
        self.edge_dim = edge_dim(extended_edges)
        self.excluded_smiles = []
        self.excluded_ids = []
        self.n_total = 0
        super().__init__(None, transform, pre_transform, log=False)

        data_list = self.process_data()
        self.data, self.slices = self.collate(data_list)

    def process_data(self):
        with open(self.path, "r") as f:
            dataset = f.read().split("\n")
            dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

        # * Get cols names
        cols = [col.lower() for col in dataset[0].split(",")]
        col_smiles = cols.index("smiles")
        col_labels = cols.index("label")
        col_set = cols.index("set")
        col_id = cols.index("id")

        dataset = dataset[1:-1]
        self.n_total = len(dataset)

        # * Iterate
        data_list = []
        for line in tqdm(dataset, ncols=120, desc="Creating graphs"):
            line = re.sub(r"\'.*?\'", "", line)  # Replace '...' strings.
            line = line.split(",")

            # Get label
            ys = line[col_labels]
            ys = ys if isinstance(ys, list) else [ys]
            ys = [float(y) if len(y) > 0 else float("NaN") for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            mol_set = line[col_set].lower()
            mol_smiles = line[col_smiles]
            mol_idx = line[col_id]

            # Create graph object
            frags, frag_map, atom_map, edge_feats = self._get_map(
                mol_smiles, extended=self.extended_edges)
            if frags is None:
                self.excluded_smiles.append(mol_smiles)
                self.excluded_ids.append(mol_idx)
                continue

            mol = Chem.MolFromSmiles(mol_smiles)
            x = _fragment_features(mol, atom_map, len(frags))

            mapping = [list(atom_map.keys()), list(atom_map.values())]

            # Bidirectional edges; each undirected pair keeps its feature
            # vector on both directions.
            edges = []
            feats = []
            for (u, v), feat in zip(frag_map, edge_feats):
                edges.append((u, v))
                edges.append((v, u))
                feats.append(feat)
                feats.append(feat)

            if len(edges) == 0:
                # Single-fragment molecule: no edges.
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, self.edge_dim),
                                        dtype=torch.float)
            else:
                edge_index = torch.tensor(edges, dtype=torch.long)
                edge_index = edge_index.t().contiguous()
                edge_attr = torch.tensor(feats, dtype=torch.float)

            data = Data(x=x, edge_index=edge_index,
                        edge_attr=edge_attr, y=y,
                        idx=mol_idx, set=mol_set,
                        frag=frags, atom_map=mapping,
                        smiles=mol_smiles)

            data_list.append(data)

        return data_list

    def exclusion_summary(self):
        """Return {n_total, n_excluded, fraction, excluded} dict."""
        n_excl = len(self.excluded_smiles)
        frac = (n_excl / self.n_total) if self.n_total else 0.0
        return {"n_total": self.n_total,
                "n_excluded": n_excl,
                "fraction_excluded": frac,
                "excluded_smiles": list(self.excluded_smiles),
                "excluded_ids": list(self.excluded_ids)}

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass


def _fragment_features(mol, atom_map: dict, n_frags: int):
    """Sum intact-molecule atom features over each fragment.

    Atoms are described as they exist in the parent molecule, so the
    bonds the decomposition severs do not change any atom's degree,
    hybridisation, hydrogen count, aromaticity or CIP code. Reading the
    features off a re-sanitised fragment instead would let RDKit refill
    the broken valence with hydrogen, so an amide nitrogen would be
    encoded as ammonia and an attachment point as a terminal atom.

    Args:
        mol: Parent RDKit Mol whose atom indices are the keys of
            atom_map.
        atom_map: Mapping of atom index to fragment index, as returned
            by the decomposition backend. Must cover every atom.
        n_frags: Number of fragments, giving the number of rows.

    Returns:
        torch.Tensor of shape (n_frags, ATOM_FEAT_DIM) holding
        one summed feature vector per fragment.

    Raises:
        ValueError: If atom_map does not cover every atom of
            mol, which would silently drop atoms from the sums.
    """
    if len(atom_map) != mol.GetNumAtoms():
        raise ValueError(f"atom_map covers {len(atom_map)} atoms but mol "
                         f"has {mol.GetNumAtoms()}")

    x = torch.zeros((n_frags, ATOM_FEAT_DIM), dtype=torch.float)
    for atom in mol.GetAtoms():
        x[atom_map[atom.GetIdx()]] += atom_features(atom)
    return x
