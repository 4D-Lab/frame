from rdkit import Chem
from rdkit.Contrib.IFG import ifg

from frame.source.datasets.frag_edges import build_edges


def _assign_scaffold_nodes(mol, node_of: list[int], next_node: int):
    """Group the non-functional-group atoms into connected nodes.

    Runs union-find over scaffold-scaffold bonds only, so each connected
    run of non-FG atoms becomes a single node. ``node_of`` is filled in
    place for every atom still marked ``-1``.

    Args:
        mol: Parent molecule.
        node_of: Atom-indexed node ids; ``-1`` marks scaffold atoms.
        next_node: First free node id (i.e. number of FG nodes).

    Returns:
        Total number of nodes after the scaffold ones are assigned.
    """
    parent = list(range(mol.GetNumAtoms()))

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
    for a in range(mol.GetNumAtoms()):
        if node_of[a] != -1:
            continue
        r = find(a)
        if r not in root_to_node:
            root_to_node[r] = next_node
            next_node += 1
        node_of[a] = root_to_node[r]

    return next_node


def _fragment_smiles(kek_mol, atoms: list[int]):
    """Return a parseable canonical SMILES for a subset of atoms.

    Ertl functional groups may cut through rings, so writing the subset
    straight from the aromatic molecule yields SMILES such as ``"cc"``
    or ``"n"`` that RDKit cannot parse back. Writing from a kekulized
    copy keeps every fragment readable; re-canonicalising restores
    aromaticity for fragments that are aromatic on their own (e.g. an
    intact benzene ring).

    Args:
        kek_mol: Kekulized copy of the parent molecule.
        atoms: Atom indices belonging to this fragment.

    Returns:
        Canonical SMILES string that ``Chem.MolFromSmiles`` accepts.
    """
    smiles = Chem.MolFragmentToSmiles(kek_mol, atomsToUse=atoms,
                                      kekuleSmiles=True, canonical=True)
    frag = Chem.MolFromSmiles(smiles)
    if frag is None:
        raise ValueError(f"unparseable fragment SMILES {smiles!r}")
    return Chem.MolToSmiles(frag)


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

    Raises:
        ValueError: If a fragment cannot be written as parseable SMILES.
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

    next_node = _assign_scaffold_nodes(mol, node_of, next_node)

    atom_map = {a: node_of[a] for a in range(n)}

    # Fragment SMILES per node
    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    fragments = []
    for node_id in range(next_node):
        atoms = [a for a in range(n) if node_of[a] == node_id]
        fragments.append(_fragment_smiles(kek, atoms))

    # Edges and edge features
    frag_map, edge_feats = build_edges(mol, node_of, extended=extended)

    return fragments, frag_map, atom_map, edge_feats
