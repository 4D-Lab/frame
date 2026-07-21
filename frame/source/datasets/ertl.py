from rdkit import Chem
from rdkit.Contrib.IFG import ifg

from frame.source.datasets.frag_edges import build_edges


def get_map_ertl(smiles: str, extended: bool = True):
    """Functional-group-preserving atom->fragment map with edge feats.

    Args:
        smiles: Input molecule SMILES.
        extended: Passed to build_edges (14-dim vs 10-dim edges).

    Returns:
        fragments:  list[str] canonical SMILES, one per node.
        frag_map:   list[(u, v)] undirected fragment-fragment edges.
        atom_map:   dict[atom_idx -> node_idx], covering every atom.
        edge_feats: list[list[float]] aligned with frag_map.
        Returns (None, None, None, None) only if SMILES cannot parse.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None

    n = mol.GetNumAtoms()

    # Functional groups: each Ertl group is one node.
    fgs = ifg.identify_functional_groups(mol)
    node_of = [-1] * n
    for node_id, fg in enumerate(fgs):
        for a in fg.atomIds:
            node_of[a] = node_id
    next_node = len(fgs)

    #  Scaffold atoms: connected components among non-FG atoms
    #  (union-find over scaffold-scaffold bonds only).
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if node_of[i] == -1 and node_of[j] == -1:
            parent[find(i)] = find(j)

    root_to_node = {}
    for a in range(n):
        if node_of[a] == -1:
            r = find(a)
            if r not in root_to_node:
                root_to_node[r] = next_node
                next_node += 1
            node_of[a] = root_to_node[r]

    atom_map = {a: node_of[a] for a in range(n)}

    # Fragment SMILES per node
    fragments = []
    for node_id in range(next_node):
        atoms = [a for a in range(n) if node_of[a] == node_id]
        fragments.append(
            Chem.MolFragmentToSmiles(mol, atomsToUse=atoms, canonical=True))

    # Edges and edge features
    frag_map, edge_feats = build_edges(mol, node_of, extended=extended)

    return fragments, frag_map, atom_map, edge_feats
