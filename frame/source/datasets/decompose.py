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

random.seed(8)
np.random.seed(8)
torch.manual_seed(8)
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr,
                                      GlobalStorage, Data])

HYBRD = [Chem.rdchem.HybridizationType.S,
         Chem.rdchem.HybridizationType.SP,
         Chem.rdchem.HybridizationType.SP2,
         Chem.rdchem.HybridizationType.SP3,
         Chem.rdchem.HybridizationType.SP3D,
         Chem.rdchem.HybridizationType.SP3D2,
         "other"]
STEREOS = [Chem.rdchem.BondStereo.STEREONONE,
           Chem.rdchem.BondStereo.STEREOANY,
           Chem.rdchem.BondStereo.STEREOZ,
           Chem.rdchem.BondStereo.STEREOE]
SYMBOLS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "R"]

# Available decomposition backends
_DECOMPOSERS = {"brics": get_map_brics, "ertl": get_map_ertl}


class DecomposeDataset(InMemoryDataset):
    """Fragment-level molecular dataset with edge features.

    Fragments become graph nodes; edges carry the chemistry of the
    bond(s) connecting two fragments (see frag_edges.py). Two backends
    are available via ``method``:

    - ``"ertl"`` (default): functional groups stay intact as nodes and
      molecules are essentially never excluded.
    - ``"brics"``: original BRICS retrosynthetic fragmentation;
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

            xs = []
            for frag in frags:
                xs.append(_gen_features(frag))
            x = torch.stack(xs, dim=0)

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


def _gen_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    xs = []
    for atom in mol.GetAtoms():
        symbol = [0.] * len(SYMBOLS)
        try:
            symbol[SYMBOLS.index(atom.GetSymbol())] = 1.
        except ValueError:
            symbol[SYMBOLS.index("R")] = 1.
        degree = [0.] * 6
        try:
            degree[atom.GetDegree()] = 1.
        except IndexError:
            degree[5] = 1.
        formal_charge = atom.GetFormalCharge()
        radical_electrons = atom.GetNumRadicalElectrons()
        hybridization = [0.] * len(HYBRD)
        hybridization[HYBRD.index(
            atom.GetHybridization())] = 1.
        aromaticity = 1. if atom.GetIsAromatic() else 0.
        hydrogens = [0.] * 5
        hydrogens[atom.GetTotalNumHs()] = 1.
        chirality = 1. if atom.HasProp("_ChiralityPossible") else 0.
        chirality_type = [0.] * 2
        if atom.HasProp("_CIPCode"):
            chirality_type[["R", "S"].index(atom.GetProp("_CIPCode"))] = 1.

        x = torch.tensor(symbol + degree + [formal_charge] +
                         [radical_electrons] + hybridization +
                         [aromaticity] + hydrogens + [chirality] +
                         chirality_type)
        xs.append(x)
    frag_x = torch.stack(xs, dim=0)

    agg_x = torch.sum(frag_x, dim=0)
    return agg_x
