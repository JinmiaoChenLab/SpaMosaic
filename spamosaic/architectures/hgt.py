from typing import List, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

class HGT(torch.nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) model for learning on heterogeneous graphs.

    This implementation builds on PyTorch Geometric's HGTConv and supports 
    node-type specific input projection, multi-layer HGT attention, and per-type decoders.

    Parameters
    ----------
    in_channels : int
        Input feature dimension for each node.
    hidden_channels : int
        Hidden feature dimension used in HGT layers.
    num_heads : int
        Number of attention heads in each HGTConv layer.
    num_layers : int
        Number of stacked HGTConv layers.
    n_dec_l : int
        Number of layers in decoder. If 1, uses a single linear layer.
    data_obj : HeteroData
        A PyG HeteroData object describing the node and edge types in the graph.
    out_channels : int, optional
        If specified, adds an intermediate projection layer to this dimension before decoding.

    Attributes
    ----------
    lin_dict : nn.ModuleDict
        Node-type specific input projections to hidden space.
    convs : nn.ModuleList
        List of stacked HGTConv layers.
    decoder : nn.ModuleDict
        Node-type specific decoders for reconstruction.
    lin : nn.Linear, optional
        Optional projection layer if out_channels is specified.
    node_type : str
        The target node type for downstream prediction.
    """
    def __init__(self, in_channels, hidden_channels, num_heads, num_layers, n_dec_l, data_obj, out_channels=None):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data_obj.node_types:
            self.lin_dict[node_type] = torch.nn.Linear(in_channels, hidden_channels)
        self.node_type = data_obj.node_types[0]

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data_obj.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin, dec_inp_channels = None, hidden_channels
        if out_channels is not None:
            self.lin = torch.nn.Linear(hidden_channels, out_channels)
            dec_inp_channels = out_channels

        self.decoder = torch.nn.ModuleDict()
        for node_type in data_obj.node_types:
            if n_dec_l == 1:
                self.decoder[node_type] = torch.nn.Linear(dec_inp_channels, in_channels)
            else:
                self.decoder[node_type] = torch.nn.Sequential(
                    torch.nn.Linear(dec_inp_channels, dec_inp_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(dec_inp_channels, in_channels)
                )

    def forward(self, x_dict, edge_index_dict):
        """
        Follow https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/hgt_dblp.py
        
        Forward pass of the HGT model.

        Parameters
        ----------
        x_dict : Dict[str, Tensor]
            Dictionary mapping node types to feature matrices.
        edge_index_dict : Dict[Tuple[str, str, str], Tensor]
            Dictionary mapping edge types to edge index tensors.

        Returns
        -------
        z : Tensor
            The final node embeddings for the primary node type, L2 normalized.
        recon : Tensor
            Reconstructed input for the primary node type (for autoencoder training).
        """
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()  
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        r_dict = {
            node_type: self.decoder[node_type](x)
            for node_type, x in x_dict.items()
        }

        return F.normalize(x_dict[self.node_type], p=2, dim=1), r_dict[self.node_type]


