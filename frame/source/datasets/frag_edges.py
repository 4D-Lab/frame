from collections import defaultdict

from rdkit import Chem

BOND_TYPES = [Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]
STEREOS = [Chem.rdchem.BondStereo.STEREONONE,
           Chem.rdchem.BondStereo.STEREOANY,
           Chem.rdchem.BondStereo.STEREOZ,
           Chem.rdchem.BondStereo.STEREOE]

EDGE_DIM_BASE = 10
EDGE_DIM_EXTENDED = 14


def edge_dim(extended: bool = True):
    """Feature width produced by build_edges for the given mode."""
    return EDGE_DIM_EXTENDED if extended else EDGE_DIM_BASE


def _bond_feats(bonds, extended: bool):
    """Aggregate a list of RDKit Bond objects into one feature vector."""
    bt = [0.] * 4
    stereo = [0.] * 4
    conj = inring = rot = 0.
    for b in bonds:
        t = b.GetBondType()
        if t in BOND_TYPES:
            bt[BOND_TYPES.index(t)] = 1.
        s = b.GetStereo()
        if s in STEREOS:
            stereo[STEREOS.index(s)] = 1.
        if b.GetIsConjugated():
            conj = 1.
        if b.IsInRing():
            inring = 1.
        if t == Chem.rdchem.BondType.SINGLE and not b.IsInRing():
            rot = 1.
    base = bt + [conj, inring] + stereo          # 10 dims
    if not extended:
        return base
    n = len(bonds)
    nbucket = [1. if n == 1 else 0.,
               1. if n == 2 else 0.,
               1. if n >= 3 else 0.]
    return base + [rot] + nbucket                # 14 dims


def build_edges(mol, node_of, extended: bool = True):
    """Fragment-graph edges + aligned edge features.

    Args:
        mol: RDKit Mol whose atom indices match node_of.
        node_of: sequence or dict mapping atom_idx -> node(fragment) id.
        extended: include rotatable flag + n-crossing-bonds bucket.

    Returns:
        frag_map:   list[(u, v)] one entry per connected fragment pair.
        edge_feats: list[list[float]] aligned with frag_map; each vector
                    has length edge_dim(extended).
    """
    pair_bonds = defaultdict(list)
    for b in mol.GetBonds():
        u = node_of[b.GetBeginAtomIdx()]
        v = node_of[b.GetEndAtomIdx()]
        if u != v:
            pair_bonds[(min(u, v), max(u, v))].append(b)

    frag_map, edge_feats = [], []
    for pair, bonds in pair_bonds.items():
        frag_map.append(pair)
        edge_feats.append(_bond_feats(bonds, extended))
    return frag_map, edge_feats
