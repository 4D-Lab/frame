import torch
import numpy as np
from rdkit import Chem
import matplotlib as mpl
from lxml import etree
import matplotlib.pyplot as plt
import svgutils.transform as sg
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.colors import Normalize, TwoSlopeNorm

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


def plot_explanations(data, node_mask, pred, out, loader, k=10):
    smiles = data.smiles
    fragments = data.frag
    mol = Chem.MolFromSmiles(smiles)

    if (loader == "default") or (loader == "decompose"):
        labels = np.array(V1)
    elif loader == "decompose_v2":
        labels = np.array(V2)

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
    frag_imgs = []

    for i, frag in enumerate(fragments):
        top, bot = _get_cut(mask_frag[i], labels, k)
        (top_lbl, top_val), (bot_lbl, bot_val) = top, bot

        contrib = np.sum(mask_frag[i]).round(3)
        entries = [f"{lbl}: {val}" for lbl, val in
                   zip(np.append(top_lbl, bot_lbl),
                       np.append(top_val, bot_val))]
        frag_imgs.append(_subplot(frag, entries, contrib))

    # Create image
    n_rows = int(np.ceil(len(frag_imgs) / 3))
    width = 1600
    height = 300 * n_rows
    bg = f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>'
    background = etree.fromstring(bg)

    x = 50
    y = 20
    count = 0
    for img in frag_imgs:
        img.moveto(x, y)
        x += 510
        count += 1

        if count == 3:
            count = 0
            x = 50
            y += 300

    fig = sg.SVGFigure(str(width), str(height))
    fig.append([background] + frag_imgs)
    fig.save(out / f"{data.idx}_frag.svg")


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


def _subplot(frag, entries, contrib, size=(500, 250)):
    mol = Chem.MolFromSmiles(frag)

    pos_1 = ", ".join(entries[:3])
    pos_2 = ", ".join(entries[3: 5])
    neg_1 = ", ".join(entries[5: 8])
    neg_2 = ", ".join(entries[8:])
    legend = (f"{frag}\nTotal: {contrib}"
              f"\n{pos_1}\n{pos_2}\n"
              f"\n{neg_1}\n{neg_2}")

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.addStereoAnnotation = True
    opts.dummiesAreAttachments = False
    opts.padding = 0.0
    opts.legendFraction = 0.5
    opts.legendFontSize = 16
    drawer.DrawMolecule(mol, legend=legend)
    drawer.FinishDrawing()

    mol_svg = drawer.GetDrawingText()
    mol_fig = sg.fromstring(mol_svg)
    mol_plot = mol_fig.getroot()

    return mol_plot


def retrieve_info(data, node_mask, pred, loader, count_frag, count_lbl, k=10):
    cut = k // 2
    pred_label = int(pred > 0.5)
    fragments = np.array(data.frag)

    if (loader == "default") or (loader == "decompose"):
        labels = np.array(V1)
    elif loader == "decompose_v2":
        labels = np.array(V2)

    top_frag_c, bot_frag_c, top_frag_w, bot_frag_w = count_frag
    top_lbl_c, bot_lbl_c, top_lbl_w, bot_lbl_w = count_lbl

    if pred_label == int(data.y):
        top_counter_frag = top_frag_c
        bot_counter_frag = bot_frag_c
        top_counter_lbl = top_lbl_c
        bot_counter_lbl = bot_lbl_c
    else:
        top_counter_frag = top_frag_w
        bot_counter_frag = bot_frag_w
        top_counter_lbl = top_lbl_w
        bot_counter_lbl = bot_lbl_w

    mask_frag = node_mask.sum(dim=1).numpy()

    # top-k positives - Fragments
    pos_idx = []
    pos_sel = np.where(mask_frag > 0)[0]
    if len(pos_sel) > 0:
        pos_sel = pos_sel[np.argsort(-mask_frag[pos_sel])]
        pos_idx = pos_sel[:cut]

        pos_frags = fragments[pos_idx]
        top_frags, top_vals = np.unique(pos_frags, return_counts=True)
        for frag, val in zip(top_frags, top_vals):
            top_counter_frag[frag] += val

    # bottom-k negatives - Fragments
    neg_sel = np.where(mask_frag < 0)[0]
    if len(neg_sel) > 0:
        neg_sel = neg_sel[np.argsort(mask_frag[neg_sel])]
        neg_sel = np.array([i for i in neg_sel if i not in pos_idx])
        neg_idx = neg_sel[:cut]

        neg_frags = fragments[neg_idx]
        bot_frags, bot_vals = np.unique(neg_frags, return_counts=True)
        for frag, val in zip(bot_frags, bot_vals):
            bot_counter_frag[frag] += val

    # Feature
    mask_label = node_mask.sum(dim=0).numpy().round(3)
    top, bot = _get_cut(mask_label, labels, k=10)

    for lab in top[0]:
        top_counter_lbl[lab] += 1
    for lab in bot[0]:
        bot_counter_lbl[lab] += 1


def plot_counters(data, out, prefix="", top_n=35):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data
    top_correct, bot_correct, top_wrong, bot_wrong = data

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
    out_feat = out / f"all_{prefix}.svg"
    fig.savefig(out_feat, format="svg")
    plt.close(fig)
