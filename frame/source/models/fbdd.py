import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, TransformerConv, Set2Set


class GNN_FBDD(torch.nn.Module):
    """Edge-aware GNN tailored for fragment-based drug discovery.

    Combines edge-conditioned message passing with a global attention
    block and a Set2Set readout. Targets BRICS-fragment graphs (few
    nodes, informative edges) but works on atom-level molecular graphs
    too.

    Pipeline:
        1. Project node and edge features to ``hidden_channels``.
        2. ``num_layers`` :class:`GINEConv` blocks with residual
           connections, :class:`LayerNorm`, GELU, and dropout. Each
           block uses a two-layer MLP and consumes edge embeddings.
        3. One :class:`TransformerConv` block providing global,
           attention-based message passing across the small fragment
           graph (gated residual via ``beta=True``).
        4. :class:`Set2Set` readout — order-invariant pooling well
           suited to variable-size molecular graphs.
        5. Two-layer MLP head producing a single scalar.

    Args:
        config: Hyperparameter dictionary. Recognised keys:
            ``feat_size`` (int): node feature dimension.
            ``edge_dim`` (int): edge feature dimension.
            ``hidden_channels`` (int): hidden embedding size. Rounded
                down to the nearest multiple of ``heads``.
            ``num_layers`` (int): number of GINEConv layers.
            ``dropout_rate`` (float): dropout probability.
            ``heads`` (int): heads for the TransformerConv.
            ``set2set_steps`` (int): processing steps in Set2Set.

    Raises:
        ValueError: If ``feat_size`` or ``edge_dim`` is missing, or if
            ``heads`` is less than 1.

    Example:
        >>> model = GNN_FBDD({"feat_size": 33, "edge_dim": 10})
        >>> out = model(x, edge_index, edge_attr, batch)
    """

    def __init__(self, config: dict):
        super().__init__()
        in_channels = config.get("feat_size", None)
        edge_dim = config.get("edge_dim", None)
        hidden = int(config.get("hidden_channels", 128))
        num_layers = int(config.get("num_layers", 3))
        dropout = float(config.get("dropout_rate", 0.2))
        heads = int(config.get("heads", 4))
        s2s_steps = int(config.get("set2set_steps", 3))

        if in_channels is None:
            raise ValueError("config['feat_size'] is required")
        if edge_dim is None:
            raise ValueError("config['edge_dim'] is required")
        if heads < 1:
            raise ValueError(f"heads must be >= 1; got {heads}.")
        hidden = max(heads, (hidden // heads) * heads)

        self.dropout_p = dropout
        self.node_embed = nn.Linear(in_channels, hidden)
        self.edge_embed = nn.Linear(edge_dim, hidden)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden),
                                nn.GELU(),
                                nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.norms.append(nn.LayerNorm(hidden))

        self.transformer = TransformerConv(hidden,
                                           hidden // heads,
                                           heads=heads,
                                           edge_dim=hidden,
                                           dropout=dropout,
                                           beta=True)
        self.transformer_norm = nn.LayerNorm(hidden)

        self.readout = Set2Set(hidden, processing_steps=s2s_steps)

        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden),
                                  nn.GELU(),
                                  nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        self.readout.lstm.flatten_parameters()
        h = self.node_embed(x)
        e = self.edge_embed(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h_in = h
            h = conv(h, edge_index, e)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout_p,
                          training=self.training)
            h = h + h_in

        h_t = self.transformer(h, edge_index, edge_attr=e)
        h = self.transformer_norm(h + h_t)

        h_pool = self.readout(h, batch)
        out = self.head(h_pool)

        return out
