import re
import random

import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import BRICS
from torch_geometric.data.data import Data
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


class DecomposeDataset(InMemoryDataset):
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

            mol_set = line[col_set].lower()
            mol_smiles = line[col_smiles]
            mol_idx = line[col_id]

            # Create graph object
            frags, frag_map, atom_map = _get_map(mol_smiles)
            if frags is not None:
                xs = []
                for frag in frags:
                    xs.append(_gen_features(frag))
                x = torch.stack(xs, dim=0)

                mapping = [list(atom_map.keys()), list(atom_map.values())]

                edges = []
                for u, v in frag_map:
                    edges.append((u, v))
                    edges.append((v, u))
                edge_index = torch.tensor(edges, dtype=torch.long)
                edge_index = edge_index.t().contiguous()
                edge_attr = torch.ones(edge_index.size(1), 1)

                data = Data(x=x, edge_index=edge_index,
                            edge_attr=edge_attr, y=y,
                            idx=mol_idx, set=mol_set,
                            frag=frags, atom_map=mapping,
                            smiles=mol_smiles)

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


def _get_map(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    connections = [bond[0] for bond in brics_bonds]

    if len(brics_bonds) == 0:
        return None, None, None

    # Get the bond object (BRICS) between the two atoms
    bond_idx = []
    atom_pairs = [bond[0] for bond in brics_bonds]
    for atom_pair in atom_pairs:
        bond = mol.GetBondBetweenAtoms(atom_pair[0], atom_pair[1])
        if bond:
            bond_idx.append(bond.GetIdx())

    # Break the molecule at bond indices and get fragments
    broken_mol = Chem.FragmentOnBonds(mol, bond_idx, addDummies=False)
    frag_idx = Chem.GetMolFrags(broken_mol)

    # Create a map from atom index to fragment index
    atom_map = {}  # Atom → Fragment
    for i, frag in enumerate(frag_idx):
        for atom_idx in frag:
            atom_map[atom_idx] = i

    # Get fragments SMILES
    frag_mols = Chem.GetMolFrags(broken_mol, asMols=True)
    fragments = [Chem.MolToSmiles(frag) for frag in frag_mols]

    # Create fragment connection map
    frag_map = []  # Fragment → Fragment
    for conn in connections:
        frag_0 = atom_map[conn[0]]
        frag_1 = atom_map[conn[1]]
        frag_map.append((frag_0, frag_1))

    return fragments, frag_map, atom_map


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

    # edge_attrs = []
    # for bond in mol.GetBonds():
    #     bond_type = bond.GetBondType()
    #     single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
    #     double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
    #     triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
    #     aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
    #     conjugation = 1. if bond.GetIsConjugated() else 0.
    #     ring = 1. if bond.IsInRing() else 0.
    #     stereo = [0.] * 4
    #     stereo[STEREOS.index(bond.GetStereo())] = 1.

    #     edge_attr = torch.tensor(
    #         [single, double, triple, aromatic, conjugation, ring] + stereo)

    #     edge_attrs += [edge_attr, edge_attr]
    #     frag_edge_attr = torch.stack(edge_attrs, dim=0)

    agg_x = torch.sum(frag_x, dim=0)
    return agg_x
