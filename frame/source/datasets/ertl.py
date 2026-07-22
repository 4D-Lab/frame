"""Ring-aware Ertl functional-group decomposition.

Ertl's algorithm marks functional-group atoms only, which leaves the
carbon skeleton to fragment into scraps: roughly half the resulting
nodes are bare heteroatoms and about half of all rings are cut. Both
are fatal when fragment-level explanations are checked against
pharmacophore SMARTS, so this module repairs them:

* Fused ring systems are seeded as whole nodes before functional groups
  are assigned, so no ring is ever cut. A functional group reaching
  into a ring donates its exocyclic atoms to that ring node, which
  keeps ring-borne motifs (lactams, pyridones) matchable. Because every
  ring atom is pre-assigned, the leftover scaffold is purely acyclic and
  the union-find pass can no longer merge whole ring systems into one
  oversized node.
* An acyclic functional group absorbs its neighbouring carbons, so
  hydroxyls and amines become ``CO`` / ``CN`` nodes instead of bare
  ``O`` / ``N`` heteroatoms.

Fragment SMILES are written in aromatic form when possible and fall
back to a kekulized copy only for subsets RDKit cannot re-parse.
"""

from rdkit import Chem
from rdkit.Contrib.IFG import ifg

from frame.source.datasets.frag_edges import build_edges


class _Union:
    """Minimal union-find over atom indices."""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, x: int):
        """Return the representative of ``x``."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        """Merge the sets containing ``a`` and ``b``."""
        self.parent[self.find(a)] = self.find(b)


def _seed_ring_nodes(mol, node_of: list[int]):
    """Assign every ring atom to its fused-ring-system node.

    Atoms joined by a ring bond belong to the same system, so fused and
    spiro systems become a single node.

    Args:
        mol: Parent molecule.
        node_of: Atom-indexed node ids, filled in place.

    Returns:
        Number of ring nodes created (the first free node id).
    """
    union = _Union(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        if bond.IsInRing():
            union.union(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    root_to_node = {}
    for atom in mol.GetAtoms():
        if not atom.IsInRing():
            continue
        root = union.find(atom.GetIdx())
        if root not in root_to_node:
            root_to_node[root] = len(root_to_node)
        node_of[atom.GetIdx()] = root_to_node[root]
    return len(root_to_node)


def _place_functional_group(mol, group, node_of: list[int],
                            next_node: int):
    """Assign one Ertl functional group to a node.

    A group overlapping a ring donates its still-unassigned (exocyclic)
    atoms to that ring node; a fully acyclic group becomes a new node
    and absorbs its neighbouring carbons.

    Args:
        mol: Parent molecule.
        group: One ``rdkit.Contrib.IFG.ifg`` result.
        node_of: Atom-indexed node ids, filled in place.
        next_node: First free node id.

    Returns:
        The next free node id after this group.
    """
    if any(node_of[a] != -1 for a in group.atomIds):
        host = next(node_of[a] for a in group.atomIds
                    if node_of[a] != -1)
        for a in group.atomIds:
            if node_of[a] == -1:
                node_of[a] = host
        return next_node

    for a in group.atomIds:
        node_of[a] = next_node
    for a in group.atomIds:
        for nbr in mol.GetAtomWithIdx(a).GetNeighbors():
            if node_of[nbr.GetIdx()] == -1 and nbr.GetSymbol() == "C":
                node_of[nbr.GetIdx()] = next_node
    return next_node + 1


def _merge_leftover_atoms(mol, node_of: list[int], next_node: int):
    """Group the still-unassigned atoms into connected linker nodes.

    Runs union-find over bonds whose both endpoints are unassigned, so
    each connected run of leftover atoms becomes one node. All such
    atoms are acyclic once the ring nodes have been seeded.

    Args:
        mol: Parent molecule.
        node_of: Atom-indexed node ids; ``-1`` marks leftover atoms.
        next_node: First free node id.

    Returns:
        Total number of nodes after the leftovers are assigned.
    """
    union = _Union(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if node_of[i] == -1 and node_of[j] == -1:
            union.union(i, j)

    root_to_node = {}
    for a in range(mol.GetNumAtoms()):
        if node_of[a] != -1:
            continue
        root = union.find(a)
        if root not in root_to_node:
            root_to_node[root] = next_node
            next_node += 1
        node_of[a] = root_to_node[root]
    return next_node


def _fragment_smiles(mol, kek_mol, atoms: list[int]):
    """Return a parseable canonical SMILES for a subset of atoms.

    The aromatic form is tried first so intact rings stay aromatic; a
    kekulized copy is the fallback for subsets that RDKit writes as
    unparseable aromatic SMILES (e.g. ``"cc"``).

    Args:
        mol: Parent molecule.
        kek_mol: Kekulized copy of the parent molecule.
        atoms: Atom indices belonging to this fragment.

    Returns:
        Canonical SMILES string that ``Chem.MolFromSmiles`` accepts.

    Raises:
        ValueError: If neither form can be parsed back.
    """
    smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=atoms,
                                      canonical=True)
    frag = Chem.MolFromSmiles(smiles)
    if frag is None:
        smiles = Chem.MolFragmentToSmiles(kek_mol, atomsToUse=atoms,
                                          kekuleSmiles=True,
                                          canonical=True)
        frag = Chem.MolFromSmiles(smiles)
    if frag is None:
        raise ValueError(f"unparseable fragment SMILES {smiles!r}")
    return Chem.MolToSmiles(frag)


def get_map_ertl(smiles: str, extended: bool = True):
    """Ring-aware functional-group atom->fragment map with edge feats.

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

    Example:
        >>> get_map_ertl("O=C1NCCC1CO")[0]
        ['O=C1CCCN1', 'CO']
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None, None

    n = mol.GetNumAtoms()
    node_of = [-1] * n

    next_node = _seed_ring_nodes(mol, node_of)
    for group in ifg.identify_functional_groups(mol):
        next_node = _place_functional_group(mol, group, node_of,
                                            next_node)
    next_node = _merge_leftover_atoms(mol, node_of, next_node)

    atom_map = {a: node_of[a] for a in range(n)}

    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    fragments = []
    for node_id in range(next_node):
        atoms = [a for a in range(n) if node_of[a] == node_id]
        fragments.append(_fragment_smiles(mol, kek, atoms))

    frag_map, edge_feats = build_edges(mol, node_of, extended=extended)

    return fragments, frag_map, atom_map, edge_feats
