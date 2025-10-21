import random
from copy import deepcopy

import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch_geometric.nn import GATv2Conv, MFConv, global_add_pool

device = "cuda" if torch.cuda.is_available() else "cpu"


class TentativeModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = config.get("feat_size", None)
        out_channels = config.get("hidden_channels", 64)
        num_layers = config.get("num_layers", 2)
        dropout = config.get("dropout_rate", 0.4)

        self.in_channels = deepcopy(in_channels)
        self.out_channels = deepcopy(out_channels)
        self.num_layers = deepcopy(num_layers)
        self.dropout = deepcopy(dropout)

        heads = config.get("heads", 1)
        max_degree = config.get("max_degree", 10)

        self.heads = deepcopy(heads)
        self.max_degree = deepcopy(max_degree)

        # GATconv block
        self.gat = GATv2Conv(in_channels, out_channels, heads=heads)

        self.gat_block = torch.nn.ModuleList([])
        for _ in range(self.num_layers - 1):
            self.gat_block.append(GATv2Conv(out_channels * heads,
                                            out_channels,
                                            heads=heads, bias=True))

        # MFconv block
        self.mf = MFConv(in_channels, out_channels, max_degree, bias=True)

        self.mf_block = torch.nn.ModuleList([])
        for _ in range(self.num_layers - 1):
            self.mf_block.append(MFConv(out_channels, out_channels, max_degree,
                                        bias=False))

        self.projection = torch.nn.Linear(in_channels, out_channels)

        # Final MLP
        size = out_channels
        self.fc_lin1 = torch.nn.Linear(size, out_channels)
        self.fc_norm = torch.nn.LayerNorm(out_channels)
        self.fc_lin2 = torch.nn.Linear(out_channels, out_channels // 2)
        self.fc_lin3 = torch.nn.Linear(out_channels // 2, 1)

        self.reset_parameters()
        self.set_seed(8)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        # MFconv block
        h = x.clone()
        h = F.elu(self.mf(x, edge_index))

        for i in range(self.num_layers - 1):
            h = F.elu(self.mf_block[i](h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = global_add_pool(h, batch)

        # GAT Block
        x = F.elu(self.gat(x, edge_index))

        for i in range(self.num_layers - 1):
            x = F.elu(self.gat_block[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(x.shape[0], self.heads, self.out_channels).mean(dim=1)
        x = global_add_pool(x, batch)

        # Merge
        pool = torch.add(x, h)

        # Final MLP
        pool = F.relu(self.fc_norm(self.fc_lin1(pool)))
        pool = F.dropout(pool, p=self.dropout, training=self.training)
        pool = F.relu(self.fc_lin2(pool))
        pool = self.fc_lin3(pool)

        return pool

    def reset_parameters(self):
        self.gat.reset_parameters()
        for layer in self.gat_block:
            layer.reset_parameters()
        self.mf.reset_parameters()
        for layer in self.mf_block:
            layer.reset_parameters()
        self.fc_lin1.reset_parameters()
        self.fc_lin2.reset_parameters()
        self.fc_lin3.reset_parameters()
        self.fc_norm.reset_parameters()
        self.projection.reset_parameters()

    def set_seed(self, value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(value)
            torch.cuda.manual_seed_all(value)
        cudnn.deterministic = True
        cudnn.benchmark = False
