import os

import torch
import numpy as np
from rdkit import Chem
import matplotlib as mpl
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import rdMolDraw2D

mpl.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ["WienerIndex", "RandicIndex", "MolWt", "NumAtoms",
          "NumBonds", "NumHeavyAtoms", "NumHeteroatoms", "NumHDonors",
          "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings",
          "NumAliphaticRings", "NumAmideBonds", "NumAtomStereoCenters",
          "NHOHCount", "NOCount", "FractionCSP3", "BalabanJ", "BertzCT",
          "Kappa1", "Kappa2", "Kappa3", "TPSA", "LabuteASA", "RadiusGyration",
          "GeneralCharge", "Charge_mean", "Charge_std", "Charge_min",
          "Charge_max", "Dipole_Debye", "DipoleVec_X", "DipoleVec_Y",
          "DipoleVec_Z"]


def plot_importance(data, explanation, cwd, k=10):
    labels = np.array(LABELS)
    node_mask = explanation.node_mask

    mask_node = torch.sum(node_mask, dim=0).tolist()
    mask_node = np.array([round(x, 3) for x in mask_node])

    top_idx = np.argsort(mask_node)[-k:][::-1]
    top_mask = mask_node[top_idx]
    top_labels = labels[top_idx]

    plt.barh(top_labels, top_mask)
    plt.xlim(0, top_mask.max() * 1.1)
    for i, v in enumerate(top_mask):
        plt.text(v + 0.01, i, str(v), va="center")

    plt.title(f"Top {k} features - {data.idx}")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # * Save image
    out_path = cwd / "explain"
    os.makedirs(out_path, exist_ok=True)

    file = out_path / f"{data.idx}.png"
    plt.savefig(file, format="png", dpi=200)
    plt.close("all")


def plot_explain(data, explanation, pred, out, note=True):
    highlight_node = {}
    highlight_edge = {}
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)

    # * Prepare molecule data
    smiles = data.smiles
    atom_map = dict(zip(data.atom_map[0], data.atom_map[1]))
    mol = Chem.MolFromSmiles(smiles)

    # * Node mask
    mask_node = torch.sum(explanation.node_mask, dim=1).tolist()
    min_val = min(mask_node)
    max_val = max(mask_node)
    mask_node = [(x - min_val) / (max_val - min_val) for x in mask_node]
    mask_node = [round(x, 3) for x in mask_node]

    # Annotate values and set highlight color
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        try:
            frag_idx = atom_map[atom_idx]
            frag_val = mask_node[frag_idx]
        except KeyError:
            frag_val = -1

        rgb = cmap.to_rgba(frag_val)[:-1]
        highlight_node[atom_idx] = [rgb]

        if note:
            atom.SetProp("atomNote", str(frag_val))

    # * Draw mol
    legend = (f"Graph ID: {data.idx}\n"
              f"{smiles}\n"
              f"Prediction: {pred:.3f}\tTrue: {float(data.y)}")

    drawer = rdMolDraw2D.MolDraw2DSVG(1200, 800)
    drawer.drawOptions().fillHighlights = True
    drawer.drawOptions().bondLineWidth = 2
    drawer.drawOptions().annotationFontScale = 0.5
    drawer.drawOptions().clearBackground = True
    drawer.drawOptions().legendFontSize = 25
    drawer.DrawMoleculeWithHighlights(mol, legend,
                                      highlight_node,
                                      highlight_edge,
                                      {}, {})
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # * Save image
    file = open(out / f"{data.idx}.svg", 'w')
    file.write(svg)
    file.close()
