import torch
from rdkit import Chem

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

# 10 symbol + 6 degree + charge + radicals + 7 hybridisation +
# aromaticity + 5 hydrogens + chirality + 2 CIP codes
ATOM_FEAT_DIM = (len(SYMBOLS) + 6 + 1 + 1 + len(HYBRD) + 1 + 5 + 1 + 2)


def atom_features(atom):
    """Encode one RDKit atom as a 34-dimensional feature vector.

    Elements outside SYMBOLS fall into the catch-all "R" slot and
    hybridisations outside HYBRD into its "other" slot, so an
    unusual atom degrades gracefully instead of raising.

    Args:
        atom: RDKit Atom. Read from the molecule it belongs to, so
            degree, hydrogen count and aromaticity reflect that
            molecule's bonding.

    Returns:
        torch.Tensor of shape (ATOM_FEAT_DIM,) and dtype float32.

    Example:
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> atom_features(mol.GetAtomWithIdx(2)).shape
        torch.Size([34])
    """
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
    try:
        hybridization[HYBRD.index(atom.GetHybridization())] = 1.
    except ValueError:
        hybridization[HYBRD.index("other")] = 1.

    aromaticity = 1. if atom.GetIsAromatic() else 0.

    hydrogens = [0.] * 5
    try:
        hydrogens[atom.GetTotalNumHs()] = 1.
    except IndexError:
        hydrogens[4] = 1.

    chirality = 1. if atom.HasProp("_ChiralityPossible") else 0.
    chirality_type = [0.] * 2
    if atom.HasProp("_CIPCode"):
        chirality_type[["R", "S"].index(atom.GetProp("_CIPCode"))] = 1.

    return torch.tensor(symbol + degree + [formal_charge] +
                        [radical_electrons] + hybridization +
                        [aromaticity] + hydrogens + [chirality] +
                        chirality_type, dtype=torch.float)
