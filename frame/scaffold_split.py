import json
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from frame.source.datasets import scaffold_split


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def _murcko(smiles: str, include_chirality: bool):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality)


def _scaffold_stats(smiles_list: list, sets: list, include_chirality: bool):
    """Summarise scaffold counts and split sizes."""
    scaffolds = [_murcko(s, include_chirality) for s in smiles_list]
    group_sizes = Counter(scaffolds)
    set_counts = Counter(sets)
    return {"n_molecules": len(smiles_list),
            "n_scaffolds": len(group_sizes),
            "n_singleton_scaffolds": sum(1 for v in group_sizes.values()
                                         if v == 1),
            "largest_scaffold_group": max(group_sizes.values(),
                                          default=0),
            "split_sizes": {"train": set_counts.get("train", 0),
                            "valid": set_counts.get("valid", 0),
                            "test": set_counts.get("test", 0)}}


def main():
    parser = argparse.ArgumentParser(
        description=("Rewrite the `set` column of a CSV using a Murcko "
                     "scaffold split. Run before frame_gen."))
    parser.add_argument("-i", "--input", required=True,
                        help="Path to input CSV with id/smiles/label/set.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to output CSV.")
    parser.add_argument("--fracs", nargs=3, type=float,
                        default=[0.8, 0.1, 0.1],
                        metavar=("TRAIN", "VALID", "TEST"),
                        help="Split fractions (default: 0.8 0.1 0.1).")
    parser.add_argument("--chirality", action="store_true",
                        help="Include chirality in scaffold definition.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "smiles" not in df.columns:
        raise ValueError("Input CSV must have a `smiles` column.")

    smiles_list = df["smiles"].tolist()
    sets = scaffold_split(smiles_list,
                          fracs=tuple(args.fracs),
                          include_chirality=args.chirality)
    df["set"] = sets

    counts = df["set"].value_counts().to_dict()
    print(f"Scaffold split: {counts}")

    df.to_csv(args.output, index=False)

    stats = _scaffold_stats(smiles_list, sets, args.chirality)
    stats_path = Path(args.output).with_name("scaffold_stats.json")
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
