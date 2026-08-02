from rdkit import Chem
from rdkit.Chem import Descriptors


BACE_PATTERNS = (("transition_state_mimic",
                  "[CX4]([OX2H])[CX4][NX3]"),
                 ("transition_state_mimic",
                  "[CX4]([OX2H])[CX4]=[CX3]"),
                 ("basic_amine",
                  "[NX3;H2,H1;!$(N-C=[!#6])]"),
                 ("s2_aromatic_hydrophobic",
                  "c1ccc(cc1)[CX4,CX3]"),
                 ("s1_hydrophobic",
                  "[CX4;H2,H3][CX4;H2,H3][CX4;H2,H3][CX4;H2,H3]"))

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


def _classify_by_smarts(fragment_smiles: str, patterns: tuple):
    """Return the first SMARTS class name matching the fragment.

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
