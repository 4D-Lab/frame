from rdkit import Chem
from rdkit.Chem import BRICS


def get_map_brics(smiles: str):
    """BRICS atom-to-fragment map for one molecule.

    Args:
        smiles: Input molecule SMILES.

    Returns:
        Tuple (fragments, frag_map, atom_map) where fragments is a
        list of canonical fragment SMILES, frag_map a list of
        (u, v) undirected fragment-fragment pairs (one per
        BRICS-cleaved bond), and atom_map a dict of atom index to
        fragment index. Returns (None, None, None) when the SMILES
        cannot be parsed or has no BRICS-cleavable bond.

    Example:
        >>> frags, edges, amap = get_map_brics("CCOC(=O)c1ccccc1")
        >>> len(frags) == len(set(amap.values()))
        True
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None

    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if len(brics_bonds) == 0:
        return None, None, None

    # Bond indices of the BRICS-cleavable bonds.
    bond_idx = []
    for atom_pair, _ in brics_bonds:
        bond = mol.GetBondBetweenAtoms(atom_pair[0], atom_pair[1])
        if bond:
            bond_idx.append(bond.GetIdx())

    # Break at those bonds; addDummies=False keeps original atom
    # indices, so atom_map keys align with the intact mol.
    broken = Chem.FragmentOnBonds(mol, bond_idx, addDummies=False)
    frag_idx = Chem.GetMolFrags(broken)

    atom_map = {}
    for i, frag in enumerate(frag_idx):
        for a in frag:
            atom_map[a] = i

    frag_mols = Chem.GetMolFrags(broken, asMols=True)
    fragments = [Chem.MolToSmiles(f) for f in frag_mols]

    # One edge per cleaved bond, taken from the atoms BRICS separated.
    frag_map = [(atom_map[pair[0]], atom_map[pair[1]])
                for pair, _ in brics_bonds]

    return fragments, frag_map, atom_map
