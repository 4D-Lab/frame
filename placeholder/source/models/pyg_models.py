import torch
from torch_geometric.nn import (GCN, GraphSAGE, GIN, GAT,
                                AttentiveFP, global_mean_pool)

torch.manual_seed(8)


class GNN_GCN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("feat_size", None)
        hidden_channels = config.get("hidden_channels", 64)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout_rate", 0.4)
        improved = config.get("gcn_improved", True)

        self.model = GCN(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         out_channels=1,
                         dropout=dropout,
                         improved=improved)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.model(x, edge_index, edge_attr=edge_attr)
        x_pool = global_mean_pool(x, batch)

        return x_pool


class GNN_SAGE(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("feat_size", None)
        hidden_channels = config.get("hidden_channels", 64)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout_rate", 0.4)

        self.model = GraphSAGE(in_channels=in_channels,
                               hidden_channels=hidden_channels,
                               num_layers=num_layers,
                               out_channels=1,
                               dropout=dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.model(x, edge_index, edge_attr=edge_attr)
        x_pool = global_mean_pool(x, batch)

        return x_pool


class GNN_GIN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("feat_size", None)
        hidden_channels = config.get("hidden_channels", 64)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout_rate", 0.4)

        self.model = GIN(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         out_channels=1,
                         dropout=dropout)

    def forward(self, x, edge_index, batch):
        x = self.model(x, edge_index)
        x_pool = global_mean_pool(x, batch)

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
        heads = config.get("num_heads", 1)

        self.model = GAT(in_channels=in_channels,
                         hidden_channels=hidden_channels,
                         num_layers=num_layers,
                         out_channels=1,
                         dropout=dropout,
                         v2=v2,
                         heads=heads,
                         edge_dim=edge_dim)

    def forward(self, x, edge_index, batch):
        x = self.model(x, edge_index)
        x_pool = global_mean_pool(x, batch)

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

    def forward(self, x, edge_index, batch):
        x = self.model(x, edge_index, batch=batch)

        return x
