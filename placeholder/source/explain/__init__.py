from io import BytesIO

import torch
import numpy as np
from rdkit import Chem
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from rdkit.Chem.Draw import rdMolDraw2D, MolDraw2DCairo

mpl.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V1 = ["atom_C", "atom_N", "atom_O", "atom_F", "atom_P",
      "atom_S", "atom_Cl", "atom_Br", "atom_I", "atom_R",
      "degree_0", "degree_1", "degree_2", "degree_3",
      "degree_4", "degree_5", "charge", "radical",
      "hybridization_S", "hybridization_SP", "hybridization_SP2",
      "hybridization_SP3", "hybridization_SP3D", "hybridization_SP3D2",
      "other", "aromaticity", "TotalNumHs_0", "TotalNumHs_1",
      "TotalNumHs_2", "TotalNumHs_3", "TotalNumHs_4", "chirality",
      "chirality_R", "chirality_S"]

V2 = ["WienerIndex", "RandicIndex", "MolWt", "NumAtoms",
      "NumBonds", "NumHeavyAtoms", "NumHeteroatoms", "NumHDonors",
      "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings",
      "NumAliphaticRings", "NumAmideBonds", "NumAtomStereoCenters",
      "NHOHCount", "NOCount", "FractionCSP3", "BalabanJ", "BertzCT",
      "Kappa1", "Kappa2", "Kappa3", "TPSA", "LabuteASA", "RadiusGyration",
      "GeneralCharge", "Charge_mean", "Charge_std", "Charge_min",
      "Charge_max", "Dipole_Debye", "DipoleVec_X", "DipoleVec_Y",
      "DipoleVec_Z"]


def plot_explanations(data, node_mask, pred, out, k=10):
    labels = np.array(V1)
    smiles = data.smiles
    fragments = data.frag
    mol = Chem.MolFromSmiles(smiles)

    # * Feature-level bar plot
    mask_feat = torch.sum(node_mask, dim=0).numpy()
    (top_lbl, top_val), (bot_lbl, bot_val) = _get_cut(mask_feat, labels, k)

    fig, ax = plt.subplots(figsize=(10, 6))
    all_lbl = np.concatenate([top_lbl, bot_lbl])
    all_val = np.concatenate([top_val, bot_val])
    colors = ["blue"] * len(top_lbl) + ["red"] * len(bot_lbl)

    ax.barh(all_lbl, all_val, color=colors)
    ax.set_title(f"Top {k} Features - {data.idx}")
    ax.set_xlabel("Contribution")
    ax.invert_yaxis()

    plt.xlim(mask_feat.min() * 1.15, mask_feat.max() * 1.15)
    for i, v in enumerate(all_val):
        x_off = 0.02 if v > 0 else -0.3
        ax.text(v + x_off, i, str(v), va="center", fontsize=8)

    plt.tight_layout()
    out_feat = out / f"{data.idx}_feat.svg"
    fig.savefig(out_feat, format="svg")
    plt.close(fig)

    # * Molecule-level visualization
    atom_map = dict(zip(data.atom_map[0], data.atom_map[1]))
    mask_atom = torch.sum(node_mask, dim=1).numpy()
    mask_atom = np.round(mask_atom, 3)

    min_val = mask_atom.min()
    max_val = mask_atom.max()
    if min_val > 0:
        cmap = mpl.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=max_val),
                                     cmap=mpl.cm.Blues)
    elif max_val < 0:
        cmap = mpl.cm.ScalarMappable(norm=Normalize(vmin=min_val, vmax=0),
                                     cmap=mpl.cm.Reds)
    else:
        cmap = mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=min_val, vcenter=0,
                                                       vmax=max_val),
                                     cmap=mpl.cm.RdBu)

    highlight_node = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        frag_val = mask_atom[atom_map[idx]]
        rgb = cmap.to_rgba(frag_val)[:-1]
        highlight_node[idx] = [rgb]
        atom.SetProp("atomNote", str(frag_val))

    legend = (f"Graph ID: {data.idx}\n{smiles}\n"
              f"Prediction: {pred:.3f}\tTrue: {float(data.y)}")

    drawer = rdMolDraw2D.MolDraw2DSVG(1200, 800)
    opts = drawer.drawOptions()
    opts.fillHighlights = True
    opts.annotationFontScale = 0.5
    opts.legendFontSize = 25
    drawer.DrawMoleculeWithHighlights(mol, legend, highlight_node, {}, {}, {})
    drawer.FinishDrawing()

    with open(out / f"{data.idx}_mol.svg", "w") as f:
        f.write(drawer.GetDrawingText())

    # * Fragment-level visualization
    mask_frag = node_mask.numpy().tolist()
    frag_texts, frag_imgs = [], []

    for i, frag in enumerate(fragments):
        top, bot = _get_cut(mask_frag[i], labels, k)
        (top_lbl, top_val), (bot_lbl, bot_val) = top, bot

        entries = [f"{lbl}: {val}" for lbl, val in
                   zip(np.append(top_lbl, bot_lbl),
                       np.append(top_val, bot_val))]
        text = [", ".join(entries[e:e + 3]) for e in range(0, len(entries), 3)]
        frag_texts.append("\n".join(text))
        frag_imgs.append(_subplot(frag, (350, 200)))

    n_axes = int(np.ceil(len(frag_imgs) / 3))
    fig, axes = plt.subplots(n_axes, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, img_data in enumerate(frag_imgs):
        if i < len(axes):
            axes[i].imshow(plt.imread(BytesIO(img_data)))
            axes[i].set_title(f"Frag {i} - {fragments[i]}", fontsize=10)
            axes[i].set_xlabel(frag_texts[i], fontsize=9)
        else:
            axes[i].axis("off")

    plt.tight_layout()
    out_feat = out / f"{data.idx}_frag.svg"
    fig.savefig(out_feat, format="svg")
    plt.close(fig)


def _get_cut(mask, labels, k=10):
    """Return top-k and bottom-k values + labels from a mask."""
    mask = np.round(mask, 3)
    labels = np.array(labels)
    cut = k // 2

    idx_top = np.argsort(mask)[-cut:][::-1]
    idx_bot = np.argsort(mask)[:cut][::-1]

    top = (labels[idx_top], mask[idx_top])
    bot = (labels[idx_bot], mask[idx_bot])

    return top, bot


def _subplot(frag, size=(300, 300)):
    mol = Chem.MolFromSmiles(frag)

    drawer = MolDraw2DCairo(size[0], size[1])
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().dummiesAreAttachments = False
    drawer.drawOptions().padding = 0.0
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    return drawer.GetDrawingText()


def retrieve_info(data, node_mask, pred, correct, wrong, k=10):
    cut = k // 2
    # labels = np.array(V1)
    pred_label = int(pred > 0.5)
    fragments = np.array(data.frag)

    if pred_label == int(data.y):
        top_counter, bot_counter = correct
    else:
        top_counter, bot_counter = wrong

    mask_frag = node_mask.sum(dim=1).cpu().numpy()

    # top-k positives
    pos_idx = []
    pos_sel = np.where(mask_frag > 0)[0]
    if len(pos_sel) > 0:
        pos_sel = pos_sel[np.argsort(-mask_frag[pos_sel])]
        pos_idx = pos_sel[:cut]
        pos_frags = fragments[pos_idx]
        top_frags, top_vals = np.unique(pos_frags, return_counts=True)

        for frag, val in zip(top_frags, top_vals):
            top_counter[frag] += val

    # bottom-k negatives
    neg_sel = np.where(mask_frag < 0)[0]
    if len(neg_sel) > 0:
        neg_sel = neg_sel[np.argsort(mask_frag[neg_sel])]
        neg_sel = np.array([i for i in neg_sel if i not in pos_idx])
        neg_idx = neg_sel[:cut]
        neg_frags = fragments[neg_idx]
        bot_frags, bot_vals = np.unique(neg_frags, return_counts=True)

        for frag, val in zip(bot_frags, bot_vals):
            bot_counter[frag] += val


def plot_counters(correct, wrong, out, top_n=35):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data
    top_correct, bot_correct = correct
    top_wrong, bot_wrong = wrong

    def prepare_data(counter, top_n=None):
        sorted_items = sorted(counter.items(), key=lambda x: x[1],
                              reverse=True)
        if top_n is not None:
            sorted_items = sorted_items[:top_n]
        frags = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        return frags, counts

    # Top-K → Correct
    frags, counts = prepare_data(top_correct, top_n)
    axes[0, 0].bar(frags, counts, color="RoyalBlue")
    axes[0, 0].set_title("Positive contributions")
    axes[0, 0].tick_params(axis="x", rotation=90)

    # Bottom-K → Correct
    frags, counts = prepare_data(bot_correct, top_n)
    axes[1, 0].bar(frags, counts, color="Crimson")
    axes[1, 0].set_title("Negative contributions")
    axes[1, 0].tick_params(axis="x", rotation=90)

    # Top-K → Wrong
    frags, counts = prepare_data(top_wrong, top_n)
    axes[0, 1].bar(frags, counts, color="RoyalBlue")
    axes[0, 1].set_title("Positive contributions")
    axes[0, 1].tick_params(axis="x", rotation=90)

    # Bottom-K → Wrong
    frags, counts = prepare_data(bot_wrong, top_n)
    axes[1, 1].bar(frags, counts, color="Crimson")
    axes[1, 1].set_title("Negative contributions")
    axes[1, 1].tick_params(axis="x", rotation=90)

    fig.text(0.25, 0.95, "Correct predictions", ha="center",
             fontsize=14, fontweight="bold")
    fig.text(0.75, 0.95, "Wrong predictions", ha="center",
             fontsize=14, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_feat = out / "0_fragments.svg"
    fig.savefig(out_feat, format="svg")
    plt.close(fig)
