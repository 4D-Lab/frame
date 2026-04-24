import torch
from torch_geometric.nn import (GCN, GraphSAGE, GIN, GAT,
                                AttentiveFP,
                                global_mean_pool,
                                global_add_pool,
                                global_max_pool)


_POOLS = {"mean": global_mean_pool,
          "add": global_add_pool,
          "sum": global_add_pool,
          "max": global_max_pool}


def _resolve_pool(name):
    key = (name or "mean").lower()
    if key not in _POOLS:
        raise ValueError(f"Unknown pool '{name}'. "
                         f"Choose from {sorted(set(_POOLS))}.")
    return _POOLS[key]


class GNN_GCN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("feat_size", None)
        hidden_channels = config.get("hidden_channels", 64)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout_rate", 0.4)
        improved = config.get("gcn_improved", True)

        self.pool = _resolve_pool(config.get("pool", "mean"))
        self.model = GCN(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         out_channels=1,
                         dropout=dropout,
                         improved=improved)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.model(x, edge_index, edge_attr=edge_attr)
        x_pool = self.pool(x, batch)

        return x_pool


class GNN_SAGE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("feat_size", None)
        hidden_channels = config.get("hidden_channels", 64)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout_rate", 0.4)

        self.pool = _resolve_pool(config.get("pool", "mean"))
        self.model = GraphSAGE(in_channels=in_channels,
                               hidden_channels=hidden_channels,
                               num_layers=num_layers,
                               out_channels=1,
                               dropout=dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.model(x, edge_index, edge_attr=edge_attr)
        x_pool = self.pool(x, batch)

        return x_pool


class GNN_GIN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("feat_size", None)
        hidden_channels = config.get("hidden_channels", 64)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout_rate", 0.4)

        self.pool = _resolve_pool(config.get("pool", "mean"))
        self.model = GIN(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         out_channels=1,
                         dropout=dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.model(x, edge_index, edge_attr=edge_attr)
        x_pool = self.pool(x, batch)

        return x_pool


class GNN_GAT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("feat_size", None)
        hidden_channels = config.get("hidden_channels", 64)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout_rate", 0.4)
        edge_dim = config.get("edge_dim", None)
        v2 = config.get("att_v2", True)
        heads = config.get("heads", 1)

        self.pool = _resolve_pool(config.get("pool", "mean"))
        self.model = GAT(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         out_channels=1,
                         dropout=dropout,
                         v2=v2,
                         heads=heads,
                         edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.model(x, edge_index, edge_attr=edge_attr)
        x_pool = self.pool(x, batch)

        return x_pool


class GNN_AttentiveFP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("feat_size", None)
        hidden_channels = config.get("hidden_channels", 64)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout_rate", 0.4)
        edge_dim = config.get("edge_dim", None)
        num_timesteps = config.get("num_timesteps", 2)

        self.model = AttentiveFP(in_channels=in_channels,
                                 hidden_channels=hidden_channels,
                                 out_channels=1,
                                 edge_dim=edge_dim,
                                 num_layers=num_layers,
                                 dropout=dropout,
                                 num_timesteps=num_timesteps)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.model(x, edge_index, edge_attr=edge_attr, batch=batch)

        return x
