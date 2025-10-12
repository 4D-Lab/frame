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


class MolExplain:
    def __init__(self, explanation, pred, pred_lbl, loader, out_dir, k=10):
        self.mask = explanation.node_mask.detach().cpu()
        self.batch = explanation.batch.detach().cpu()
        self.pred = pred
        self.pred_lbl = pred_lbl
        self.out = out_dir
        self.k = k
        self.cut = k // 2

        if loader == "default":
            self.labels = np.array(V1)
        elif loader == "decompose":
            self.labels = np.array(V1)
        elif loader == "decompose_v2":
            self.labels = np.array(V2)

    def retrieve_info(self, graphs, count_frag, count_lbl):
        batch_num = self.batch.unique()
        masks = [self.mask[self.batch == b] for b in batch_num]

        for idx, node_mask in enumerate(masks):
            data = graphs[idx]
            pred_label = self.pred_lbl[idx].numpy()[0]
            fragments = np.array(data.frag)
            mask_frag = node_mask.sum(dim=1).numpy()
            mask_label = node_mask.sum(dim=0).numpy().round(3)

            # Unpack counters
            key = f"{pred_label}_{int(data.y)}"
            class_0_frag = count_frag[key][0]
            class_1_frag = count_frag[key][1]
            class_0_lbl = count_lbl[key][0]
            class_1_lbl = count_lbl[key][1]

            # * Count fragments
            top_fragments = self._get_top(mask_frag, fragments)

            if len(top_fragments[1]) > 0:
                for frag in top_fragments[1]["fragment"]:
                    class_1_frag[frag] += 1

            if len(top_fragments[0]) > 0:
                for frag in top_fragments[0]["fragment"]:
                    class_0_frag[frag] += 1

            # * Count features
            top_features = self._get_top(mask_label)

            for feat in top_features[1]["labels"]:
                class_1_lbl[feat] += 1
            for feat in top_features[0]["labels"]:
                class_0_lbl[feat] += 1

    def plot_explanations(self, graphs):
        batch_num = self.batch.unique()
        masks = [self.mask[self.batch == b] for b in batch_num]

        for idx, node_mask in enumerate(masks):
            data = graphs[idx]
            smiles = data.smiles
            fragments = data.frag
            pred = self.pred[idx]
            pred_label = self.pred_lbl[idx].numpy()[0]
            mol = Chem.MolFromSmiles(smiles)

            # * Feature-level bar plot
            mask_feat = torch.sum(node_mask, dim=0).numpy()
            feats = self._get_top(mask_feat)

            fig, ax = plt.subplots(figsize=(10, 6))
            all_lbl = np.append(feats[1]["labels"], feats[0]["labels"])
            all_val = np.append(feats[1]["contrib"], feats[0]["contrib"])
            colors = (["RoyalBlue"] * len(feats[1]["labels"]) +
                      ["Crimson"] * len(feats[0]["labels"]))

            ax.barh(all_lbl, all_val, color=colors)
            ax.set_title(f"Top {self.k} Features - {data.idx}")
            ax.set_xlabel("Contribution")
            ax.invert_yaxis()

            plt.xlim(mask_feat.min() * 1.15, mask_feat.max() * 1.15)
            for i, v in enumerate(all_val):
                x_off = 0.02 if v > 0 else -0.3
                ax.text(v + x_off, i, str(v), va="center", fontsize=8)

            plt.tight_layout()
            out_feat = self.out / f"{data.idx}_feat.svg"
            fig.savefig(out_feat, format="svg")
            plt.close(fig)

            # * Molecule-level visualization
            atom_map = dict(zip(data.atom_map[0], data.atom_map[1]))
            mask_atom = torch.sum(node_mask, dim=1).numpy()
            mask_atom = np.round(mask_atom, 3)

            min_val = mask_atom.min()
            max_val = mask_atom.max()
            if min_val > 0:
                cmap = mpl.cm.ScalarMappable(norm=Normalize(vmin=0,
                                                            vmax=max_val),
                                             cmap=mpl.cm.Blues)
            elif max_val < 0:
                cmap = mpl.cm.ScalarMappable(norm=Normalize(vmin=min_val,
                                                            vmax=0),
                                             cmap=mpl.cm.Reds_r)
            else:
                cmap = mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=min_val,
                                                               vcenter=0,
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
                      f"Prediction: {pred:.3f} ({pred_label})"
                      f"\tTrue: {float(data.y)}")

            drawer = rdMolDraw2D.MolDraw2DSVG(1200, 800)
            opts = drawer.drawOptions()
            opts.fillHighlights = True
            opts.annotationFontScale = 0.5
            opts.legendFontSize = 25
            drawer.DrawMoleculeWithHighlights(mol, legend, highlight_node,
                                              {}, {}, {})
            drawer.FinishDrawing()

            with open(self.out / f"{data.idx}_mol.svg", "w") as f:
                f.write(drawer.GetDrawingText())

            # * Fragment-level visualization
            mask_frag = node_mask.numpy().tolist()
            frag_imgs = []

            for i, frag in enumerate(fragments):
                top_val = self._get_top(mask_frag[i])
                label = np.append(top_val[1]["labels"], top_val[0]["labels"])
                cntrb = np.append(top_val[1]["contrib"], top_val[0]["contrib"])

                contrib = np.sum(mask_frag[i]).round(3)
                entries = [f"{lbl}: {val}" for lbl, val in zip(label, cntrb)]
                frag_imgs.append(self._subplot(frag, entries, contrib))

            # Create image
            fig = self._create_frag_image(frag_imgs, 1600, 300)
            fig.save(self.out / f"{data.idx}_frag.svg")

    def _get_top(self, mask, fragments=None):
        mask = np.round(mask, 3)
        labels = np.array(self.labels)

        pos = {"contrib": np.array([]),
               "labels": np.array([]),
               "fragment": np.array([])}
        pos_mask = mask > 0
        if np.any(pos_mask):
            idx_pos = np.argsort(mask[pos_mask])[-self.cut:][::-1]
            idx_pos = np.where(pos_mask)[0][idx_pos]
            pos = {"contrib": mask[idx_pos]}

            if fragments is None:
                pos["labels"] = labels[idx_pos]
            else:
                pos["fragment"] = fragments[idx_pos]

        neg = {"contrib": np.array([]),
               "labels": np.array([]),
               "fragment": np.array([])}
        neg_mask = mask < 0
        if np.any(neg_mask):
            idx_neg = np.argsort(mask[neg_mask])[:self.cut]
            idx_neg = np.where(neg_mask)[0][idx_neg]
            neg = {"contrib": mask[idx_neg]}

            if fragments is None:
                neg["labels"] = labels[idx_neg]
            else:
                neg["fragment"] = fragments[idx_neg]

        cuts = {0: neg, 1: pos}
        return cuts

    def _subplot(self, frag, entries, contrib, size=(500, 250)):
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

    def _create_frag_image(self, images, width=1600, height=300):
        n_rows = int(np.ceil(len(images) / 3))
        h = height * n_rows
        bg = f'<rect x="0" y="0" width="{width}" height="{h}" fill="white"/>'
        background = etree.fromstring(bg)

        x = 50
        y = 20
        count = 0
        for img in images:
            img.moveto(x, y)
            x += width / 3.14
            count += 1

            if count == 3:
                count = 0
                x = 50
                y += height

        fig = sg.SVGFigure(str(width), str(h))
        fig.append([background] + images)
        return fig


def plot_counters(data, out, prefix="", top_n=35):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 12))

    col = 0
    keys = ["0_0", "0_1", "1_0", "1_1"]
    title = {"0_0": "Predicted: 0, Real: 0",
             "0_1": "Predicted: 0, Real: 1",
             "1_1": "Predicted: 1, Real: 1",
             "1_0": "Predicted: 1, Real: 0"}

    for key in keys:
        inner_dict = data[key]

        # Get class 0 and class 1 dict
        class_0 = inner_dict[0]
        class_0 = dict(sorted(class_0.items(), key=lambda item: item[1],
                              reverse=True))
        class_0_lbl = list(class_0.keys())[: top_n]
        class_0_num = list(class_0.values())[: top_n]

        class_1 = inner_dict[1]
        class_1 = dict(sorted(class_1.items(), key=lambda item: item[1],
                              reverse=True))
        class_1_lbl = list(class_1.keys())[: top_n]
        class_1_num = list(class_1.values())[: top_n]

        # Plot
        axes[0, col].bar(class_1_lbl, class_1_num, color="RoyalBlue")
        axes[0, col].set_title(f"{title[key]} - Class 1")
        axes[0, col].tick_params(axis="x", rotation=90)

        axes[1, col].bar(class_0_lbl, class_0_num, color="Crimson")
        axes[1, col].set_title(f"{title[key]} - Class 0")
        axes[1, col].tick_params(axis="x", rotation=90)

        col += 1

    plt.tight_layout()
    out_feat = out / f"all_{prefix}.svg"
    fig.savefig(out_feat, format="svg")
    plt.close(fig)
