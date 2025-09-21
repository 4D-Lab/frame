import re
import random

import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data.data import Data
from torch_geometric.utils import from_smiles
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

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


class MolecularDataset(InMemoryDataset):
    def __init__(self, path: str, transform=None, pre_transform=None):
        self.path = path
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

        # * Iterate
        data_list = []
        for line in tqdm(dataset, ncols=120, desc="Creating graphs"):
            line = re.sub(r"\'.*\'", "", line)  # Replace ".*" strings.
            line = line.split(",")

            # Get label
            ys = line[col_labels]
            ys = ys if isinstance(ys, list) else [ys]
            ys = [float(y) if len(y) > 0 else float("NaN") for y in ys]
            y = torch.tensor(ys, dtype=torch.float).view(1, -1)

            # Create graph object
            data = from_smiles(line[col_smiles])
            data = _gen_features(data)

            data.set = line[col_set].lower()
            data.idx = line[col_id]
            data.y = y

            data_list.append(data)

        return data_list

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


def _gen_features(data):
    mol = Chem.MolFromSmiles(data.smiles)

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

    data.x = torch.stack(xs, dim=0)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
        edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]

        bond_type = bond.GetBondType()
        single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
        double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
        triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
        aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
        conjugation = 1. if bond.GetIsConjugated() else 0.
        ring = 1. if bond.IsInRing() else 0.
        stereo = [0.] * 4
        stereo[STEREOS.index(bond.GetStereo())] = 1.

        edge_attr = torch.tensor(
            [single, double, triple, aromatic, conjugation, ring] + stereo)

        edge_attrs += [edge_attr, edge_attr]

    if len(edge_attrs) == 0:
        data.edge_index = torch.zeros((2, 0), dtype=torch.long)
        data.edge_attr = torch.zeros((0, 10), dtype=torch.float)
    else:
        data.edge_index = torch.tensor(edge_indices).t().contiguous()
        data.edge_attr = torch.stack(edge_attrs, dim=0)

    return data
