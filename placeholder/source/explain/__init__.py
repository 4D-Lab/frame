import torch
from rdkit import Chem
import matplotlib as mpl
from rdkit.Chem.Draw import rdMolDraw2D

mpl.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
