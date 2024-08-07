from typing import List, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

class HG_LGCN_Conv(MessagePassing):
    def __init__(self):
        super(HG_LGCN_Conv, self).__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight, edge_type):
        # Adding self-loops to the adjacency matrix
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=x.size(0))
        # weight norm 
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))  # common practice in GNN to incorporate edge_weight to calc deg
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Separate edges into intra-group (A) and inter-group (B)
        intra_mask = edge_type == 1
        inter_mask = edge_type == 0

        # Perform separate convolutions for intra-group and inter-group
        intra_features = self.propagate(edge_index[:, intra_mask], x=x, norm=norm[intra_mask])
        inter_features = self.propagate(edge_index[:, inter_mask], x=x, norm=norm[inter_mask])

        # Concatenate the features from both types of edges
        return torch.cat([intra_features, inter_features], dim=1)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class HG_LGCN_vanilla(torch.nn.Module):
    def __init__(self, num_layers):
        super(HG_LGCN_vanilla, self).__init__()
        self.convs = torch.nn.ModuleList([ModifiedLightGCNConv() for _ in range(num_layers)])
        # Adjust the input dimension of the linear layer
        # inp_dim = sum([2**i for i in range(num_layers+1)])*in_channels

    def forward(self, x, edge_index, edge_weight, edge_type):
        all_features = [x]

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight, edge_type)
            all_features.append(x)

        # Concatenate features from all layers
        total_features = torch.cat(all_features, dim=1)
        return total_features

class HG_LGCN(torch.nn.Module):
    def __init__(self, input_size, output_size, K=8, dec_l=1, hidden_size=512, dropout=0.2):
        super(HG_LGCN, self).__init__()
        self.conv1 = HG_LGCN_vanilla(num_layers=K)
        inp_dim = sum([2**i for i in range(K+1)])*input_size
        self.fc1 = torch.nn.Linear(inp_dim, hidden_size)
        self.bn = torch.nn.BatchNorm1d(hidden_size)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

        if dec_l == 1:
            self.decoder = torch.nn.Linear(output_size, input_size)
        else:
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(output_size, output_size),
                torch.nn.ReLU(),
                torch.nn.Linear(output_size, input_size)
            )
        
    def forward(self, feature, edge_index, edge_weight, edge_type):
        x = self.conv1(feature, edge_index, edge_weight, edge_type)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        r = self.decoder(x)
        x = F.normalize(x, p=2, dim=1)
        return x, r