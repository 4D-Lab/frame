from collections import defaultdict

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def _scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality)


def scaffold_split(smiles_list, fracs=(0.8, 0.1, 0.1),
                   include_chirality=False):
    """Murcko-scaffold split.

    Largest scaffold groups go to train; smaller ones fill valid then test.
    Molecules with the same scaffold never cross splits, which gives a more
    realistic generalization signal than a random split for drug discovery.

    Args:
        smiles_list: list of SMILES strings.
        fracs: (train, valid, test) fractions; must sum to ~1.0.
        include_chirality: pass-through to MurckoScaffoldSmiles.

    Returns:
        list of "train" / "valid" / "test", aligned with smiles_list.
    """
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"fracs must sum to 1.0, got {fracs}")

    n = len(smiles_list)
    train_target = int(round(fracs[0] * n))
    valid_target = int(round(fracs[1] * n))

    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        groups[_scaffold(smi, include_chirality)].append(i)

    # Largest groups first; tiebreak on the scaffold key for determinism.
    sorted_groups = sorted(groups.items(),
                           key=lambda kv: (-len(kv[1]), kv[0]))

    sets = ["test"] * n
    train_count = 0
    valid_count = 0
    for _, indices in sorted_groups:
        if train_count + len(indices) <= train_target:
            for i in indices:
                sets[i] = "train"
            train_count += len(indices)

        elif valid_count + len(indices) <= valid_target:
            for i in indices:
                sets[i] = "valid"
            valid_count += len(indices)

    return sets
