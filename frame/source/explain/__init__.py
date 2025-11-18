import torch
import numpy as np
from lxml import etree
from rdkit import Chem
import matplotlib as mpl
import matplotlib.pyplot as plt
import svgutils.transform as sg
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.colors import LinearSegmentedColormap
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


class MolExplain:
    def __init__(self, explanation, pred, pred_lbl, loader, out_dir, k=10):
        self.mask = explanation.node_mask.detach().cpu()
        self.batch = explanation.batch.detach().cpu()
        self.pred = pred
        self.pred_lbl = pred_lbl
        self.loader = loader
        self.out = out_dir
        self.k = k
        self.cut = k // 2

        self.labels = np.array(V1)

    def retrieve_info(self, graphs):
        if self.loader == "default":
            header = "id,smiles,real,pred_label,pred,fragment"
            with open(self.out / "predictions.csv", "w") as f:
                f.write(f"{header}\n")

            self._info_atom(graphs)

        else:
            labels = ",".join(V1)
            header = "id,smiles,real,pred_label,pred"
            with open(self.out / "predictions.csv", "w") as f:
                f.write(f"{header+labels}\n")

            self._info_fragment(graphs)

    def _info_atom(self, graphs):
        batch_num = self.batch.unique()
        masks = [self.mask[self.batch == b] for b in batch_num]

        for idx in range(len(masks)):
            data = graphs[idx]
            real_label = int(data.y.cpu().numpy()[0])
            pred = self.pred[idx]
            pred_label = self.pred_lbl[idx].numpy()[0]

            text = (f"{data.idx},{data.smiles},{real_label},"
                    f"{pred_label},{pred:.3f}\n")

            # * Export prediction
            with open(self.out / "predictions.csv", "a") as f:
                f.writelines(text)

    def _info_fragment(self, graphs):
        batch_num = self.batch.unique()
        masks = [self.mask[self.batch == b] for b in batch_num]

        for idx, node_mask in enumerate(masks):
            data = graphs[idx]
            real_label = int(data.y.cpu().numpy()[0])
            pred = self.pred[idx]
            pred_label = self.pred_lbl[idx].numpy()[0]
            fragments = np.array(data.frag)

            mask_list = node_mask.cpu().numpy().tolist()
            mask_list = [[f"{m:.3f}" for m in mask] for mask in mask_list]

            text = []
            for mask, frag in zip(mask_list, fragments):
                contribs = ",".join(mask)
                txt = (f"{data.idx},{data.smiles},{real_label},{pred_label},"
                       f"{pred:.3f},{frag},{contribs}\n")
                text.append(txt)

            # * Export prediction
            with open(self.out / "predictions.csv", "a") as f:
                f.writelines(text)

    def plot_explanations(self, graphs):
        batch_num = self.batch.unique()
        masks = [self.mask[self.batch == b] for b in batch_num]

        for idx, node_mask in enumerate(masks):
            data = graphs[idx]
            name = data.idx
            pred = self.pred[idx]
            pred_label = self.pred_lbl[idx].numpy()[0]

            if self.loader == "default":
                self._explain_atom(data, node_mask, pred, pred_label, name)

            else:
                # * Feature-level bar plot
                self._bar_plot(node_mask, name)

                # * Fragment-level visualization
                fragments = data.frag
                self._frag_visualization(node_mask, fragments, name)

                # * Molecule-level visualization
                self._explain_frag(data, node_mask, pred, pred_label, name)

    def _explain_atom(self, data, node_mask, pred, pred_label, name):
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles)

        mask_atom = torch.sum(node_mask, dim=1).numpy()
        mask_atom = np.round(mask_atom, 3)

        min_val = mask_atom.min()
        max_val = mask_atom.max()
        if min_val > 0:
            max_val *= 1.3
            cmap = mpl.cm.ScalarMappable(norm=Normalize(vmin=0,
                                                        vmax=max_val),
                                         cmap=mpl.cm.Blues)
        elif max_val < 0:
            min_val *= 1.3
            cmap = mpl.cm.ScalarMappable(norm=Normalize(vmin=min_val,
                                                        vmax=0),
                                         cmap=mpl.cm.Oranges_r)
        else:
            min_val *= 1.3
            max_val *= 1.3
            pos_colors = plt.cm.Blues(np.linspace(0, 1, 128))
            neg_colors = plt.cm.Oranges_r(np.linspace(0, 1, 128))
            combined = np.vstack((neg_colors, pos_colors))
            color = LinearSegmentedColormap.from_list("OrBu", combined)
            cmap = mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=min_val,
                                                           vcenter=0,
                                                           vmax=max_val),
                                         cmap=color)

        highlight_node = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            rgb = cmap.to_rgba(mask_atom[idx])[:-1]
            highlight_node[idx] = [rgb]
            atom.SetProp("atomNote", str(mask_atom[idx]))

        legend = (f"Graph ID: {name}\n{smiles}\n"
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

    def _explain_frag(self, data, node_mask, pred, pred_label, name):
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles)

        atom_map = dict(zip(data.atom_map[0], data.atom_map[1]))
        mask_atom = torch.sum(node_mask, dim=1).numpy()
        mask_atom = np.round(mask_atom, 3)

        min_val = mask_atom.min()
        max_val = mask_atom.max()
        if min_val > 0:
            max_val *= 1.3
            cmap = mpl.cm.ScalarMappable(norm=Normalize(vmin=0,
                                                        vmax=max_val),
                                         cmap=mpl.cm.Blues)
        elif max_val < 0:
            min_val *= 1.3
            cmap = mpl.cm.ScalarMappable(norm=Normalize(vmin=min_val,
                                                        vmax=0),
                                         cmap=mpl.cm.Oranges_r)
        else:
            min_val *= 1.3
            max_val *= 1.3
            pos_colors = plt.cm.Blues(np.linspace(0, 1, 128))
            neg_colors = plt.cm.Oranges_r(np.linspace(0, 1, 128))
            combined = np.vstack((neg_colors, pos_colors))
            color = LinearSegmentedColormap.from_list("OrBu", combined)
            cmap = mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=min_val,
                                                           vcenter=0,
                                                           vmax=max_val),
                                         cmap=color)

        highlight_node = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            frag_val = mask_atom[atom_map[idx]]
            rgb = cmap.to_rgba(frag_val)[:-1]
            highlight_node[idx] = [rgb]
            atom.SetProp("atomNote", str(frag_val))

        legend = (f"Graph ID: {name}\n{smiles}\n"
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

        with open(self.out / f"{name}_mol.svg", "w") as f:
            f.write(drawer.GetDrawingText())

    def _bar_plot(self, node_mask, name):
        # * Feature-level bar plot
        mask_feat = torch.sum(node_mask, dim=0).numpy()
        feats = self._get_top(mask_feat)

        fig, ax = plt.subplots(figsize=(10, 6))
        all_lbl = np.append(feats[1]["labels"], feats[0]["labels"])
        all_val = np.append(feats[1]["contrib"], feats[0]["contrib"])
        colors = (["SteelBlue"] * len(feats[1]["labels"]) +
                  ["DarkOrange"] * len(feats[0]["labels"]))

        ax.barh(all_lbl, all_val, color=colors)
        ax.set_title(f"Top {self.k} Features - {name}")
        ax.set_xlabel("Contribution")
        ax.invert_yaxis()

        plt.xlim(mask_feat.min() * 1.15, mask_feat.max() * 1.15)
        for i, v in enumerate(all_val):
            x_off = 0.02 if v > 0 else -0.3
            ax.text(v + x_off, i, str(v), va="center", fontsize=8)

        plt.tight_layout()
        out_feat = self.out / f"{name}_feat.svg"
        fig.savefig(out_feat, format="svg")
        plt.close(fig)

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

    def _frag_visualization(self, node_mask, fragments, name):
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
        fig.save(self.out / f"{name}_frag.svg")

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
