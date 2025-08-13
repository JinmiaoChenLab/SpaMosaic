from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

# forked from https://github.com/QIFEIDKN/STAGATE_pyG
class GATConv(MessagePassing):
    """
    Graph Attention Network (GAT) layer adapted from STAGATE implementation.

    Parameters
    ----------
    in_channels : int or tuple of int
        Dimension(s) of input node features.
    out_channels : int
        Dimension of output node features.
    heads : int, default=1
        Number of attention heads.
    concat : bool, default=True
        Whether to concatenate multi-head outputs (True) or average them (False).
    negative_slope : float, default=0.2
        LeakyReLU angle of the negative slope.
    dropout : float, default=0.0
        Dropout probability for attention coefficients.
    add_self_loops : bool, default=True
        Whether to add self-loops to the input graph.
    bias : bool, default=True
        Whether to add bias (not used here).
    **kwargs : optional
        Additional arguments for `MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        self._alpha = None
        self.attentions = None

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, attention=True, tied_attention = None):
        """
        Forward pass for the GAT layer.

        Parameters
        ----------
        x : Tensor or tuple of Tensors
            Input node features, shape (N, F).
        edge_index : Tensor or SparseTensor
            Graph connectivity in COO format.
        size : tuple of int, optional
            Size of source and target node sets.
        return_attention_weights : bool, optional
            Whether to return attention weights along with output.
        attention : bool, default=True
            Whether to apply attention mechanism.
        tied_attention : tuple of Tensors, optional
            Precomputed attention weights to reuse.

        Returns
        -------
        out : Tensor
            Output node embeddings.
        attention_weights : optional
            Attention weights (if `return_attention_weights` is True).
        """
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            # x_src = x_dst = self.lin_src(x).view(-1, H, C)
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)

        if tied_attention == None:
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        """
        Message passing step: applies attention-weighted aggregation.

        Parameters
        ----------
        x_j : Tensor
            Features of source nodes.
        alpha_j : Tensor
            Attention logits from source nodes.
        alpha_i : Tensor or None
            Attention logits from target nodes.
        index : Tensor
            Target indices for aggregation.
        ptr : Tensor or None
            Optional pointer array for variable-sized batches.
        size_i : int, optional
            Number of target nodes.

        Returns
        -------
        Tensor
            Aggregated node features after attention weighting.
        """
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i

        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F

# FROM STAGATE_pyg
class GAT(torch.nn.Module):
    """
    A 4-layer Graph Attention Network used for spatial transcriptomics.

    This module implements a symmetric encoder-decoder GAT with tied weights.

    Parameters
    ----------
    hidden_dims : list of int
        A list [in_dim, hidden_dim, out_dim] specifying the feature dimensions.
    """
    def __init__(self, hidden_dims):
        super(GAT, self).__init__()

        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, features, edge_index):
        """
        Forward pass of the GAT model.

        Parameters
        ----------
        features : Tensor
            Input node features.
        edge_index : Tensor
            Edge indices in COO format.

        Returns
        -------
        Tuple[Tensor, Tensor]
            - Latent embeddings after encoder (normalized).
            - Reconstructed features after decoder.
        """
        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(self.conv3(h2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4 = self.conv4(h3, edge_index, attention=False)

        return F.normalize(h2, p=2, dim=1), h4  # F.log_softmax(x, dim=-1)


