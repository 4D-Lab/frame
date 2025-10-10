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

from rdkit.Chem import (AllChem, rdMolDescriptors, Descriptors,
                        Lipinski, GraphDescriptors)

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


class DecomposeDataset_v2(InMemoryDataset):
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
                # TODO: Add edges attributes?
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
    mol = Chem.AddHs(mol)
    attrs = []

    attrs.append(_wiener_index(mol))
    attrs.append(_randic_index(mol))

    attrs.append(Descriptors.ExactMolWt(mol))
    attrs.append(mol.GetNumAtoms())
    attrs.append(mol.GetNumBonds())
    attrs.append(mol.GetNumHeavyAtoms())

    attrs.append(Lipinski.NumHeteroatoms(mol))
    attrs.append(Lipinski.NumHDonors(mol))
    attrs.append(Lipinski.NumHAcceptors(mol))
    attrs.append(Lipinski.NumRotatableBonds(mol))
    attrs.append(Lipinski.NumAromaticRings(mol))
    attrs.append(Lipinski.NumAliphaticRings(mol))
    attrs.append(Lipinski.NumAmideBonds(mol))
    attrs.append(Lipinski.NumAtomStereoCenters(mol))
    attrs.append(Lipinski.NHOHCount(mol))
    attrs.append(Lipinski.NOCount(mol))
    attrs.append(Lipinski.FractionCSP3(mol))

    attrs.append(GraphDescriptors.BalabanJ(mol))
    attrs.append(GraphDescriptors.BertzCT(mol))

    attrs.append(rdMolDescriptors.CalcKappa1(mol))
    attrs.append(rdMolDescriptors.CalcKappa2(mol))
    attrs.append(rdMolDescriptors.CalcKappa3(mol))
    attrs.append(rdMolDescriptors.CalcTPSA(mol))

    AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()

    attrs.append(rdMolDescriptors.CalcLabuteASA(mol))
    attrs.append(_radius_gyration(conf))

    AllChem.ComputeGasteigerCharges(mol)

    charges = []
    for a in mol.GetAtoms():
        ch = a.GetProp("_GasteigerCharge")
        charges.append(float(ch))
    qarr = np.array([q for q in charges if not np.isnan(q)])

    attrs.append(sum(charges))
    attrs.append(float(np.mean(qarr)))
    attrs.append(float(np.std(qarr)))
    attrs.append(float(np.min(qarr)))
    attrs.append(float(np.max(qarr)))

    dipole, dipvec = _dipole_from_charges(conf, charges)

    attrs.append(dipole)
    attrs.append(dipvec[0])
    attrs.append(dipvec[1])
    attrs.append(dipvec[2])

    x = torch.tensor(attrs)
    return x


def _wiener_index(mol):
    """Wiener index is a topological index of a molecule, defined as the
    sum of thenlengths of the shortest paths between all pairs of vertices
    in the chemical graph representing the non-hydrogen atoms in the molecule.

    Args:
        mol (Chem.MolFromSmiles): RDKit molecule

    Returns:
        float: Wiener index
    """
    no_h = Chem.RemoveHs(mol)

    mat = Chem.GetDistanceMatrix(no_h)
    wiener = np.sum(np.triu(mat, k=1))
    result = float(wiener)

    return result


def _randic_index(mol):
    """The Randić index of a graph is the sum of bond contributions of the
    sum over all edges of (1/sqrt(deg(i)*deg(j))) where deg(i) and deg(j)
    are degrees of connected atoms

    Args:
        mol (Chem.MolFromSmiles): RDKit molecule

    Returns:
        float: Randić index
    """
    degrees = [atom.GetDegree() for atom in mol.GetAtoms()]

    randic_index = 0.0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        deg_i = degrees[i]
        deg_j = degrees[j]
        randic_index += 1.0 / ((deg_i * deg_j) ** 0.5)

    return randic_index


def _radius_gyration(conf):
    coords = np.array([list(conf.GetAtomPosition(i))
                       for i in range(conf.GetNumAtoms())])
    masses = np.array([conf.GetOwningMol().GetAtomWithIdx(i).GetMass()
                       for i in range(conf.GetNumAtoms())])
    com = np.sum(coords * masses[:, None], axis=0) / np.sum(masses)
    rg2 = np.sum(masses * np.sum((coords - com)**2, axis=1)) / np.sum(masses)

    return float(np.sqrt(rg2))


def _dipole_from_charges(conf, charges):
    # charges: list of floats (e)
    # positions in Angstroms,
    # dipole (Debye) = 4.80320427 * sum(q_i * r_i) magnitude
    coords = np.array([list(conf.GetAtomPosition(i))
                       for i in range(conf.GetNumAtoms())])
    q = np.array(charges)
    vec = np.sum((q[:, None] * coords), axis=0)  # e * Å
    mag_eA = np.linalg.norm(vec)
    # convert e·Å to Debye: 1 e·Å = 4.80320427 Debye
    debye = float(mag_eA * 4.80320427)

    return debye, vec.tolist()
