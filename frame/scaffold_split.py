import argparse

import pandas as pd

from frame.source.datasets import scaffold_split


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

    sets = scaffold_split(df["smiles"].tolist(),
                          fracs=tuple(args.fracs),
                          include_chirality=args.chirality)
    df["set"] = sets

    counts = df["set"].value_counts().to_dict()
    print(f"Scaffold split: {counts}")

    df.to_csv(args.output, index=False)
