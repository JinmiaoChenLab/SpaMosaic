"""Weighted Light graph convolution (WLGCN) layers for SpaMosaic.

Implements a parameter-free propagation layer on pre-normalized sparse adjacencies
and an MLP head for embedding + reconstruction.
"""

from __future__ import annotations
from typing import Optional, Literal, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

class WLGCN_vanilla(nn.Module):
    """
    Parameter-free LightGCN-style propagation.

    Accepts a precomputed normalized sparse adjacency
    :math:`\\hat{A} \\approx D^{-1/2}(A + I\\cdot\\text{fill})D^{-1/2}` as a
    ``torch_sparse.SparseTensor``.

    Parameters
    ----------
    K : int, default=1
        Number of propagation steps (returns ``K+1`` representations including the 0-th order input).
    agg : {'cat', 'sum', 'mean'}, default='cat'
        How to aggregate representations across steps.
    """
    def __init__(
        self,
        K: int = 1,
        agg: Literal["cat", "sum", "mean"] = "cat",
    ):
        super().__init__()
        self.K = int(K)
        self.agg = agg
    
    def forward(self, x: torch.Tensor, A_hat: SparseTensor) -> torch.Tensor:
        """
        Propagate features via ``A_hat`` for ``K`` steps and aggregate.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[N, F]``.
        A_hat : torch_sparse.SparseTensor
            Pre-normalized sparse adjacency on the correct device.

        Returns
        -------
        torch.Tensor
            Aggregated output. Shapes depend on ``agg``:
            - ``'cat'``  → ``[N, F * (K+1)]``
            - ``'sum'``  → ``[N, F]``
            - ``'mean'`` → ``[N, F]``
        """
        H = x

        if self.agg == "cat":
            outs = [H]
        else:
            acc = H
        for _ in range(self.K):
            H = A_hat.matmul(H)  
            if self.agg == "cat":
                outs.append(H)
            else:
                acc = acc + H

        if self.agg == "cat":
            return torch.cat(outs, dim=-1)
        elif self.agg == "sum":
            return acc
        else:  # mean
            denom = self.K + 1
            return acc / (denom + 1e-12)

class HEAD(nn.Module):
    """
    MLP head that outputs an L2-normalized embedding and reconstructs input features.

    Parameters
    ----------
    input_size : int
        Input feature dimension.
    output_size : int
        Embedding dimension.
    dec_l : int, default=1
        Number of decoder layers (``1`` = linear decoder).
    hidden_size : int, default=512
        Hidden size of the MLP trunk.
    dropout : float, default=0.2
        Dropout rate.
    slope : float, default=0.2
        Negative slope for LeakyReLU.
    """
    def __init__(self, input_size, output_size, dec_l=1, hidden_size=512, dropout=0.2, slope=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = nn.LeakyReLU(negative_slope=slope)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)

        if dec_l == 1:
            self.decoder = nn.Linear(output_size, input_size)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(output_size, output_size),
                nn.ReLU(),
                nn.Linear(output_size, input_size),
            )

    def forward(self, feature):
        """
        Forward pass.

        Parameters
        ----------
        feature : torch.Tensor
            Input features of shape ``[N, input_size]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - ``z``: L2-normalized embedding ``[N, output_size]``.
            - ``r``: Reconstructed features ``[N, input_size]`` from pre-norm representation.
        """
        x = self.fc1(feature)
        x = self.act(x)
        x = self.bn(x)
        x = self.dropout1(x)
        x = self.fc2(x)              # pre-norm embedding (for decoder)
        r = self.decoder(x)          # reconstruct from pre-norm
        z = F.normalize(x, p=2, dim=1)
        return z, r
