from io import BytesIO

import torch
import numpy as np
from rdkit import Chem
import matplotlib as mpl
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import rdMolDraw2D, MolDraw2DCairo

mpl.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V2 = ["WienerIndex", "RandicIndex", "MolWt", "NumAtoms",
      "NumBonds", "NumHeavyAtoms", "NumHeteroatoms", "NumHDonors",
      "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings",
      "NumAliphaticRings", "NumAmideBonds", "NumAtomStereoCenters",
      "NHOHCount", "NOCount", "FractionCSP3", "BalabanJ", "BertzCT",
      "Kappa1", "Kappa2", "Kappa3", "TPSA", "LabuteASA", "RadiusGyration",
      "GeneralCharge", "Charge_mean", "Charge_std", "Charge_min",
      "Charge_max", "Dipole_Debye", "DipoleVec_X", "DipoleVec_Y",
      "DipoleVec_Z"]

ATOM = ["atom_C", "atom_N", "atom_O", "atom_F", "atom_P",
        "atom_S", "atom_Cl", "atom_Br", "atom_I", "atom_R"]
DEGREE = ["degree_0", "degree_1", "degree_2",
          "degree_3", "degree_4", "degree_5"]
HYBRD = ["hybridization_S", "hybridization_SP", "hybridization_SP2",
         "hybridization_SP3", "hybridization_SP3D", "hybridization_SP3D2",
         "other"]
HS = ["TotalNumHs_0", "TotalNumHs_1", "TotalNumHs_2",
      "TotalNumHs_3", "TotalNumHs_4"]
C_TYPE = ["chirality_R", "chirality_S"]
V1 = (ATOM + DEGREE + ["charge"] + ["radical"]
      + HYBRD + ["aromaticity"] + HS
      + ["chirality"] + C_TYPE)


def retrieve_info(node_mask, k=10):
    labels = np.array(V1)
    cut = int(k/2)

    mask_node = torch.sum(node_mask, dim=0).tolist()
    mask_node = np.array([round(x, 3) for x in mask_node])

    top_idx = np.argsort(mask_node)[-cut:][::-1]
    top_mask = mask_node[top_idx]
    top_labels = labels[top_idx]

    bot_idx = np.argsort(mask_node)[:cut][::-1]
    bot_mask = mask_node[bot_idx]
    bot_labels = labels[bot_idx]

    return (top_labels, top_mask), (bot_labels, bot_mask)


def plot_general(top_k, bot_k, out):
    top = dict(sorted(top_k.items(), key=lambda x: x[1], reverse=True))
    bot = dict(sorted(bot_k.items(), key=lambda x: x[1], reverse=True))
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Top plot
    axs[0].bar(top.keys(), top.values(), color='blue')
    axs[0].set_title('Top K')
    axs[0].set_ylabel('Contribution')
    axs[0].tick_params(axis='x', rotation=75)

    # Bottom plot
    axs[1].bar(bot.keys(), bot.values(), color='red')
    axs[1].set_title('Bottom K')
    axs[1].tick_params(axis='x', rotation=75)

    plt.tight_layout()

    # * Save image
    file = out / "0.png"
    plt.savefig(file, format="png", dpi=200)
    plt.close("all")


def plot_importance(data, node_mask, out, k=10):
    labels = np.array(V1)
    cut = int(k/2)

    mask_node = torch.sum(node_mask, dim=0).tolist()
    mask_node = np.array([round(x, 3) for x in mask_node])

    top_idx = np.argsort(mask_node)[-cut:][::-1]
    top_mask = mask_node[top_idx]
    top_labels = labels[top_idx]

    bot_idx = np.argsort(mask_node)[:cut][::-1]
    bot_mask = mask_node[bot_idx]
    bot_labels = labels[bot_idx]

    plot_mask = np.append(top_mask, bot_mask)
    plot_labels = np.append(top_labels, bot_labels)
    color = ["blue"] * cut + ["red"] * cut

    plt.barh(plot_labels, plot_mask, color=color)
    plt.xlim(plot_mask.min() * 1.6, plot_mask.max() * 1.15)
    for i, v in enumerate(plot_mask):
        if i < cut:
            plt.text(v + 0.01, i, str(v), va="center")
        else:
            plt.text(v - 0.45, i, str(v), va="center")

    plt.title(f"Top {k} features - {data.idx}")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # * Save image
    file = out / f"{data.idx}.png"
    plt.savefig(file, format="png", dpi=200)
    plt.close("all")


def plot_explain(data, node_mask, pred, out):
    highlight_node = {}
    highlight_edge = {}

    # * Prepare molecule data
    smiles = data.smiles
    atom_map = dict(zip(data.atom_map[0], data.atom_map[1]))
    mol = Chem.MolFromSmiles(smiles)

    # * Node mask
    mask_node = torch.sum(node_mask, dim=1).tolist()
    mask_node = [round(x, 3) for x in mask_node]

    middle = 0
    min_val = min(mask_node)
    max_val = max(mask_node)
    if (max_val < 0) or (min_val > 0):
        middle = (max_val + min_val) / 2

    # Annotate values and set highlight color
    norm = mpl.colors.TwoSlopeNorm(vmin=min_val, vcenter=middle, vmax=max_val)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.RdBu)

    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        frag_idx = atom_map[atom_idx]
        frag_val = mask_node[frag_idx]

        rgb = cmap.to_rgba(frag_val)[:-1]
        highlight_node[atom_idx] = [rgb]

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


def plot_fragments(data, node_mask, out, k=10):
    # * Prepare molecule data
    fragments = data.frag
    labels = np.array(V1)
    cut = int(k/2)

    # * Get node mask information
    node_mask = node_mask.tolist()

    values = []
    for idx in range(len(fragments)):
        mask = node_mask[idx]
        mask = np.round(mask, 3)

        top_idx = np.argsort(mask)[-cut:][::-1]
        top_mask = mask[top_idx]
        top_labels = labels[top_idx]

        bot_idx = np.argsort(mask)[:cut][::-1]
        bot_mask = mask[bot_idx]
        bot_labels = labels[bot_idx]

        msk = np.append(top_mask, bot_mask)
        lbl = np.append(top_labels, bot_labels)
        parts = [f"{x[0]}: {x[1]}" for x in zip(lbl, msk)]

        txt = ""
        for i, part in enumerate(parts):
            txt += part
            if i < len(parts) - 1:
                txt += ", "
            if (i + 1) % 3 == 0 and i < len(parts) - 1:
                txt += "\n"
        values.append(txt)

    # * Generate fragments images
    frag_imgs = []
    for fragment in fragments:
        frag = Chem.MolFromSmiles(fragment)
        img_data = _draw_rdkit(frag, (350, 200))
        frag_imgs.append(img_data)

    n_axes = int(np.ceil(len(frag_imgs) / 3))
    _, axes = plt.subplots(n_axes, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, img_data in enumerate(frag_imgs):
        if i < len(axes):
            img = plt.imread(BytesIO(img_data))
            axes[i].imshow(img)

            title = f"Frag {i} - {fragments[i]}"
            legend = f"{values[i]}"

            axes[i].set_title(title, fontsize=10)
            axes[i].set_xlabel(legend, fontsize=10)
            # axes[i].axis("off")

    for i in range(len(fragments), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(out / f"{data.idx}_frag.svg")
    plt.close("all")


def _draw_rdkit(mol, size=(300, 300)):
    drawer = MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().dummiesAreAttachments = False
    drawer.drawOptions().padding = 0.0
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    return drawer.GetDrawingText()
