"""Heterogeneous LightGCN layers and models for SpaMosaic.

Provides a LightGCN-style convolution for heterogeneous graphs and an encoder-decoder network.
"""

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
    """
    Heterogeneous Graph LightGCN convolution layer.

    Performs degree-normalized aggregation with self-loops and separates
    intra-group vs. inter-group message passing, then concatenates the two streams.
    """
    def __init__(self):
        super(HG_LGCN_Conv, self).__init__(aggr='add')

    def forward(self, x, edge_index, edge_weight, edge_type):
        """
        Forward pass for heterogeneous LightGCN convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input node features of shape ``(N, F)``.
        edge_index : torch.LongTensor
            Edge list in COO format, shape ``(2, E)``.
        edge_weight : torch.Tensor
            Edge weights, shape ``(E,)``.
        edge_type : torch.Tensor
            Edge type indicator (``1`` for intra-group, ``0`` for inter-group).

        Returns
        -------
        torch.Tensor
            Concatenated features from intra-group and inter-group aggregations.
        """
        # Add self-loops to the adjacency.
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value=1, num_nodes=x.size(0))

        # Symmetric degree normalization with edge weights.
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Split edges into intra-group (1) and inter-group (0).
        intra_mask = edge_type == 1
        inter_mask = edge_type == 0

        # Convolve separately and concatenate.
        intra_features = self.propagate(edge_index[:, intra_mask], x=x, norm=norm[intra_mask])
        inter_features = self.propagate(edge_index[:, inter_mask], x=x, norm=norm[inter_mask])
        return torch.cat([intra_features, inter_features], dim=1)

    def message(self, x_j, norm):
        """
        Compute per-edge messages.

        Parameters
        ----------
        x_j : torch.Tensor
            Source-node features for each edge.
        norm : torch.Tensor
            Normalized scalar weight per edge.

        Returns
        -------
        torch.Tensor
            Weighted messages ``norm * x_j``.
        """
        return norm.view(-1, 1) * x_j


class HG_LGCN_vanilla(torch.nn.Module):
    """
    Stacked HG_LGCN_Conv with layer-wise feature concatenation.

    Parameters
    ----------
    num_layers : int
        Number of HG_LGCN_Conv layers to stack.

    Notes
    -----
    This class builds a list of LightGCN-style layers and concatenates the
    input features with the output of each layer.
    """
    def __init__(self, num_layers):
        super(HG_LGCN_vanilla, self).__init__()
        self.convs = torch.nn.ModuleList([HG_LGCN_Conv() for _ in range(num_layers)])
        # Adjust the input dimension of the linear layer
        # inp_dim = sum([2**i for i in range(num_layers+1)])*in_channels

    def forward(self, x, edge_index, edge_weight, edge_type):
        """
        Forward pass of stacked HG_LGCN layers.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        edge_index : torch.LongTensor
            Edge list in COO format.
        edge_weight : torch.Tensor
            Weights for each edge.
        edge_type : torch.Tensor
            Indicator for intra- vs. inter-group edges.

        Returns
        -------
        torch.Tensor
            Concatenated node features from all layers (including the input).
        """
        all_features = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight, edge_type)
            all_features.append(x)
        return torch.cat(all_features, dim=1)


class HG_LGCN(torch.nn.Module):
    """
    Heterogeneous Graph LightGCN with encoder–decoder architecture.

    Uses ``HG_LGCN_vanilla`` as the feature encoder and an MLP/linear decoder
    for reconstruction.

    Parameters
    ----------
    input_size : int
        Input feature dimensionality per node.
    output_size : int
        Latent embedding dimensionality.
    K : int, default=8
        Number of LightGCN layers.
    dec_l : int, default=1
        Number of decoder layers (``1`` = linear decoder).
    hidden_size : int, default=512
        Hidden size of the MLP head.
    dropout : float, default=0.2
        Dropout rate in the MLP head.
    """
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
        """
        Forward pass for HG_LGCN.

        Parameters
        ----------
        feature : torch.Tensor
            Input node features.
        edge_index : torch.LongTensor
            COO edge list.
        edge_weight : torch.Tensor
            Edge weights.
        edge_type : torch.Tensor
            Edge type indicators.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - L2-normalized latent representation (``[N, output_size]``).
            - Reconstructed features (``[N, input_size]``).
        """
        x = self.conv1(feature, edge_index, edge_weight, edge_type)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        r = self.decoder(x)
        x = F.normalize(x, p=2, dim=1)
        return x, r
