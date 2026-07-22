from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


BACE_PATTERNS = (("transition_state_mimic",
                  "[CX4]([OX2H])[CX4][NX3]"),
                 ("transition_state_mimic",
                  "[CX4]([OX2H])[CX4]=[CX3]"),
                 ("basic_amine",
                  "[NX3;H2,H1;!$(N-C=[!#6])]"),
                 ("s2_aromatic_hydrophobic",
                  "c1ccc(cc1)[CX4,CX3]"),
                 ("s1_hydrophobic",
                  "[CX4;H2,H3;!$(C~[!#6;!#1])]"
                  "[CX4;H2,H3;!$(C~[!#6;!#1])]"
                  "[CX4;H2,H3;!$(C~[!#6;!#1])]"
                  "[CX4;H2,H3;!$(C~[!#6;!#1])]"))

MPRO_PATTERNS = (("warhead",
                  "C#N"),
                 ("warhead",
                  "C=CC(=O)[!O]"),
                 ("warhead",
                  "C(=O)C(=O)N"),
                 ("s1_lactam_pyridone",
                  "O=C1NCCC1"),
                 ("s1_lactam_pyridone",
                  "O=C1NCCCC1"),
                 ("s1_lactam_pyridone",
                  "O=c1cccc[nH]1"),
                 ("s2_hydrophobic",
                  "C1CC1"),
                 ("s2_hydrophobic",
                  "[CX4;H1,H0]([CX4;H3])([CX4;H3])[CX4;H3]"))

BBBP_TPSA_THRESHOLD = 30.0


_PATTERNS = {"bace": BACE_PATTERNS, "mpro": MPRO_PATTERNS}

# SMARTS are compiled once; matching runs per fragment per molecule.
_QUERIES = {name: tuple((class_name, Chem.MolFromSmarts(smarts))
                        for class_name, smarts in patterns
                        if Chem.MolFromSmarts(smarts) is not None)
            for name, patterns in _PATTERNS.items()}

MATCH_MODES = ("strict", "anchored")


def get_queries(name: str):
    """Return the compiled (class_name, query) pairs for a registry.

    Args:
        name: ``"bace"`` or ``"mpro"`` (case-insensitive).

    Returns:
        Tuple of ``(class_name, rdkit.Chem.Mol)`` query pairs.

    Raises:
        ValueError: If the registry has no SMARTS definition. BBBP is
            physicochemical; use :func:`classify_bbbp_fragment`.
    """
    key = name.lower()
    if key not in _QUERIES:
        raise ValueError(f"No SMARTS registry for {name!r}. "
                         f"Choose from {list(_QUERIES)}.")
    return _QUERIES[key]


def pharmacophore_instances(mol, name: str):
    """Find every pharmacophore occurrence in an intact molecule.

    Args:
        mol: RDKit molecule (the parent, not a fragment).
        name: Registry name, ``"bace"`` or ``"mpro"``.

    Returns:
        List of ``(class_name, frozenset_of_atom_indices)`` pairs, one
        per substructure match. A molecule may yield several matches of
        the same class.

    Raises:
        ValueError: If the registry has no SMARTS definition.
    """
    found = []
    for class_name, query in get_queries(name):
        for match in mol.GetSubstructMatches(query):
            found.append((class_name, frozenset(match)))
    return found


def classify_fragment(mol, atoms, name: str, mode: str = "strict"):
    """Classify one fragment using parent-molecule SMARTS matches.

    The fragment is described by its atom indices in ``mol``, so every
    atom keeps the hydrogen count, aromaticity and substitution it has
    in the real molecule.

    Args:
        mol: Parent RDKit molecule.
        atoms: Atom indices belonging to the fragment.
        name: Registry name, ``"bace"`` or ``"mpro"``.
        mode: ``"strict"`` requires every atom of a match to lie inside
            the fragment. ``"anchored"`` also accepts a match whose
            majority of atoms lie inside, which tolerates motifs that
            straddle a cut. Defaults to ``"strict"``.

    Returns:
        Set of class names carried by the fragment; empty if none.

    Raises:
        ValueError: If ``mode`` is not a member of ``MATCH_MODES``.

    Example:
        >>> mol = Chem.MolFromSmiles("O=C1NCCC1CO")
        >>> sorted(classify_fragment(mol, range(6), "mpro"))
        ['s1_lactam_pyridone']
    """
    if mode not in MATCH_MODES:
        raise ValueError(f"mode must be one of {list(MATCH_MODES)}, "
                         f"got {mode!r}")
    inside = set(atoms)
    found = set()
    for class_name, match in pharmacophore_instances(mol, name):
        covered = len(match & inside)
        if covered == len(match):
            found.add(class_name)
        elif mode == "anchored" and covered * 2 > len(match):
            found.add(class_name)
    return found


def fragment_tpsa(mol, atoms):
    """Sum the parent molecule's per-atom TPSA over a fragment.

    Using RDKit's per-atom contributions keeps the value consistent
    with the intact molecule; recomputing TPSA from a cut fragment's
    SMILES would count hydrogens the atoms do not really carry.

    Args:
        mol: Parent RDKit molecule.
        atoms: Atom indices belonging to the fragment.

    Returns:
        TPSA contribution of the fragment in Angstrom^2.
    """
    contribs = rdMolDescriptors._CalcTPSAContribs(mol)
    return float(sum(contribs[a] for a in atoms))


def classify_bbbp_fragment(mol, atoms,
                           threshold: float = BBBP_TPSA_THRESHOLD):
    """Classify a fragment by its share of the parent molecule's TPSA.

    Args:
        mol: Parent RDKit molecule.
        atoms: Atom indices belonging to the fragment.
        threshold: TPSA cutoff in Angstrom^2. Defaults to 30.0.

    Returns:
        ``"low_tpsa"`` or ``"high_tpsa"``.
    """
    if fragment_tpsa(mol, atoms) < threshold:
        return "low_tpsa"
    return "high_tpsa"


def _classify_by_smarts(fragment_smiles: str, patterns: tuple):
    """Return the first SMARTS class name matching the fragment.

    Deprecated: matches the fragment in isolation, which misreads
    hydrogen counts. Use :func:`classify_fragment`.

    Args:
        fragment_smiles: Canonical SMILES of a single BRICS fragment.
        patterns: Iterable of (class_name, smarts) pairs evaluated
            in order; the first match wins.

    Returns:
        Class name string, or None if no pattern matches or the
        SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return None

    for class_name, smarts in patterns:
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            continue
        if mol.HasSubstructMatch(query):
            return class_name
    return None


def classify_bace(fragment_smiles: str):
    """Classify a fragment against BACE-1 inhibitor pharmacophores.

    Classes (in priority order): transition_state_mimic (catalytic
    Asp32/Asp228 contact), basic_amine (S3 recognition),
    s2_aromatic_hydrophobic (S2 pocket), s1_hydrophobic (S1
    pocket).

    Args:
        fragment_smiles: Canonical SMILES of one BRICS fragment.

    Returns:
        Class name string or None.
    """
    return _classify_by_smarts(fragment_smiles, BACE_PATTERNS)


def classify_mpro(fragment_smiles: str):
    """Classify a fragment against SARS-CoV-2 MPro inhibitor pharmacophores.

    Classes (in priority order): warhead (nitrile, Michael
    acceptor, alpha-ketoamide), s1_lactam_pyridone (gamma-lactam,
    delta-lactam, 2-pyridone), s2_hydrophobic (cyclopropyl or
    branched leucine-mimetic).

    Args:
        fragment_smiles: Canonical SMILES of one BRICS fragment.

    Returns:
        Class name string or None.
    """
    return _classify_by_smarts(fragment_smiles, MPRO_PATTERNS)


def classify_bbbp(fragment_smiles: str,
                  threshold: float = BBBP_TPSA_THRESHOLD):
    """Classify a fragment by topological polar surface area.

    BBB permeation is governed by global physicochemistry rather than
    discrete binding-site motifs, so the BBBP registry partitions
    fragments by RDKit TPSA. Fragments with TPSA < threshold are
    expected to favour BBB+ predictions; fragments above are expected
    to favour BBB- predictions.

    Args:
        fragment_smiles: Canonical SMILES of one BRICS fragment.
        threshold: TPSA cutoff in Angstrom^2. Defaults to 30.0.

    Returns:
        "low_tpsa" or "high_tpsa"; None if the SMILES is
        invalid.
    """
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return None

    tpsa = Descriptors.TPSA(mol)
    if tpsa < threshold:
        return "low_tpsa"
    return "high_tpsa"


CLASSIFIERS = {"bace": classify_bace,
               "mpro": classify_mpro,
               "bbbp": classify_bbbp}

CLASS_NAMES = {"bace": ("transition_state_mimic",
                        "basic_amine",
                        "s2_aromatic_hydrophobic",
                        "s1_hydrophobic"),
               "mpro": ("warhead",
                        "s1_lactam_pyridone",
                        "s2_hydrophobic"),
               "bbbp": ("low_tpsa",
                        "high_tpsa")}


def get_classifier(name: str):
    """Return the classify function for a case study by name.

    Args:
        name: One of "bace", "mpro", "bbbp" (case-insensitive).

    Returns:
        Callable fragment_smiles -> Optional[str].

    Raises:
        ValueError: If name is not a registered case study.
    """
    key = name.lower()
    if key not in CLASSIFIERS:
        raise ValueError(f"Unknown pharmacophore registry: {name}. "
                         f"Choose from {list(CLASSIFIERS)}.")
    return CLASSIFIERS[key]


def get_class_names(name: str):
    """Return the tuple of class names for a case study.

    Args:
        name: One of "bace", "mpro", "bbbp" (case-insensitive).

    Returns:
        Tuple of class-name strings.

    Raises:
        ValueError: If name is not a registered case study.
    """
    key = name.lower()
    if key not in CLASS_NAMES:
        raise ValueError(f"Unknown pharmacophore registry: {name}. "
                         f"Choose from {list(CLASS_NAMES)}.")
    return CLASS_NAMES[key]
