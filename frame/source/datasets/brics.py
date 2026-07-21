from rdkit import Chem
from rdkit.Chem import BRICS

from frame.source.datasets.frag_edges import build_edges


def get_map_brics(smiles: str, extended: bool = True):
    """BRICS atom->fragment map with edge features.

    Args:
        smiles: Input molecule SMILES.
        extended: Passed to build_edges (14-dim vs 10-dim edges).

    Returns:
        fragments:  list[str] canonical SMILES, one per fragment.
        frag_map:   list[(u, v)] undirected fragment-fragment edges.
        atom_map:   dict[atom_idx -> fragment_idx].
        edge_feats: list[list[float]] aligned with frag_map.
        Returns (None, None, None, None) if unparseable or no BRICS bond.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None

    brics_bonds = list(BRICS.FindBRICSBonds(mol))
    if len(brics_bonds) == 0:
        return None, None, None, None

    # Bond indices of the BRICS-cleavable bonds.
    bond_idx = []
    for atom_pair, _ in brics_bonds:
        bond = mol.GetBondBetweenAtoms(atom_pair[0], atom_pair[1])
        if bond:
            bond_idx.append(bond.GetIdx())

    # Break at those bonds; addDummies=False keeps original atom indices,
    # so atom_map keys align with the intact `mol` used for edge feats.
    broken = Chem.FragmentOnBonds(mol, bond_idx, addDummies=False)
    frag_idx = Chem.GetMolFrags(broken)

    atom_map = {}
    for i, frag in enumerate(frag_idx):
        for a in frag:
            atom_map[a] = i

    frag_mols = Chem.GetMolFrags(broken, asMols=True)
    fragments = [Chem.MolToSmiles(f) for f in frag_mols]

    # Edges + edge features computed on the INTACT mol (the broken bonds
    # are exactly the fragment-crossing bonds, so their real chemistry is
    # recovered here rather than discarded).
    frag_map, edge_feats = build_edges(mol, atom_map, extended=extended)

    return fragments, frag_map, atom_map, edge_feats
