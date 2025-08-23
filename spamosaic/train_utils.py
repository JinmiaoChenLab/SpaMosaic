"""Training utilities for SpaMosaic.

Provides seed control, edge decoding & graph reconstruction loss, and the main training loop.
"""

import os, gc
from pathlib import Path, PurePath
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
import h5py
import math
from tqdm import tqdm
import scipy.sparse as sps
import scipy.io as sio
import seaborn as sns
import warnings
# import gzip
from scipy.io import mmread

from os.path import join
import torch

import logging
import torch
from matplotlib import rcParams
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from torch_geometric.utils import negative_sampling

import random
def set_seeds(seed, dt=True):
    """
    Set random seeds for reproducibility across multiple libraries.

    Parameters
    ----------
    seed : int
        Random seed to use for reproducibility.
    dt : bool, default=True
        Whether to enforce deterministic algorithms in PyTorch.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(dt)  # ensure reproducibility

def graph_decode(z, edge_index):
    """
    Compute edge-wise similarity scores using dot product of node embeddings.

    Parameters
    ----------
    z : torch.Tensor
        Node embeddings of shape [num_nodes, embedding_dim].
    edge_index : torch.Tensor
        Edge index tensor with shape [2, num_edges].

    Returns
    -------
    torch.Tensor
        Edge probabilities computed via sigmoid(dot(z_i, z_j)).
    """
    logit = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    return torch.sigmoid(logit)

def graph_recon_crit(
    z,
    pos_edge_index,
    num_neg 
):
    """
    BCE-style reconstruction loss on sampled positive/negative edges.

    Parameters
    ----------
    z : torch.Tensor
        Node embeddings of shape ``[N, D]``.
    pos_edge_index : torch.Tensor
        Positive edges as index tensor of shape ``[2, E_pos]``.
    num_neg : int or None
        Number of negative edges to sample. If ``None``, defaults to ``E_pos``.

    Returns
    -------
    torch.Tensor
        Scalar loss value averaged over positives and negatives.
    """
    if pos_edge_index is None or pos_edge_index.numel() == 0:
        return torch.zeros((), device=z.device, dtype=z.dtype)

    E_pos = pos_edge_index.size(1)
    num_neg = E_pos if num_neg is None else max(1, int(num_neg))

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=z.size(0),
        num_neg_samples=num_neg
    ).to(z.device)

    pos_prob = graph_decode(z, pos_edge_index)
    neg_prob = graph_decode(z, neg_edge_index)

    pos_loss = -torch.log(pos_prob + 1e-20)               
    neg_loss = -torch.log(1.0 - neg_prob + 1e-20)
    return pos_loss.mean() + neg_loss.mean()

def train_model(
    mod_model, mod_optims, crit1, crit2, loss_type, w_rec_g,
    mod_input_feat, modBatch_intra_subgraphs,
    batch_train_meta_numbers, mod_batch_split,
    T, n_epochs, device
):
    """
    End-to-end training loop with contrastive loss on bridge batches and
    feature/graph reconstruction on test batches.

    Parameters
    ----------
    mod_model : dict
        Mapping ``{modality_name -> torch.nn.Module}`` (encoder/decoder).
    mod_optims : dict
        Mapping ``{modality_name -> torch.optim.Optimizer}``.
    crit1 : dict or torch.nn.Module
        Contrastive loss. If a dict, it should be keyed by bridge batch id.
    crit2 : torch.nn.Module
        Feature reconstruction criterion (e.g., ``nn.MSELoss``).
    loss_type : {'adapted', 'ce'}
        Type of contrastive loss to apply.
    w_rec_g : float
        Weight in ``[0, 1]`` for graph vs. feature reconstruction on test batches.
    mod_input_feat : dict
        Modality → full-graph input features tensor.
    modBatch_intra_subgraphs : dict or None
        Cached intra subgraphs; expects entries like
        ``modBatch_intra_subgraphs[k][bi] = {'edge_index': LongTensor[2, E] or None}``.
    batch_train_meta_numbers : dict
        Training metadata per batch (sizes, modality sets, etc.).
    mod_batch_split : dict
        Per-modality list of batch sizes for splitting full-graph tensors.
    T : float
        Temperature used for contrastive logits.
    n_epochs : int
        Number of epochs.
    device : torch.device or str
        Training device.

    Returns
    -------
    tuple
        ``(mod_model, loss_cl, loss_rec)``, where ``loss_cl``/``loss_rec`` are lists of scalars.
    """
    bridge_batch_num_ids = [bi for bi, v in batch_train_meta_numbers.items() if v[0]]
    test_batch_num_ids   = [bi for bi, v in batch_train_meta_numbers.items() if not v[0]]

    loss_cl, loss_rec = [], []

    for _ in tqdm(range(1, n_epochs + 1)):
        for k in mod_model:  mod_model[k].train()
        for k in mod_optims: mod_optims[k].zero_grad()

        # full forward
        mod_embs, mod_recs = {}, {}
        for k in mod_model:
            z, r = mod_model[k](mod_input_feat[k].to(device))
            mod_embs[k] = list(torch.split(z, mod_batch_split[k]))
            mod_recs[k] = list(torch.split(r, mod_batch_split[k]))
        # split_input = {k: list(torch.split(v, mod_batch_split[k])) for k, v in mod_input_feat.items()}

        # reconstruction (test batches)
        l2, l2n = torch.zeros((), device=device), 0
        for bi in test_batch_num_ids:
            mods = batch_train_meta_numbers[bi][1]
            start = sum(mod_batch_split[mods[0]][:bi])
            for k in mods:
                n_bi = mod_batch_split[k][bi]
                y_rec = mod_input_feat[k][start:(start+n_bi)].to(device, non_blocking=True)
                f_rec_loss = 0.0 if w_rec_g == 1.0 else crit2(mod_recs[k][bi], y_rec)

                g_rec_loss = 0.0
                if w_rec_g > 0.0: # graph rec
                    sub = modBatch_intra_subgraphs[k][bi]
                    e_local = sub.get("edge_index", None)
                    if e_local is not None and e_local.numel() > 0:
                        g_rec_loss = graph_recon_crit(
                            z=mod_embs[k][bi],
                            pos_edge_index=e_local.to(device),
                            num_neg=None
                        )
                
                l2 = l2 + (1.0 - w_rec_g) * f_rec_loss + w_rec_g * g_rec_loss
                l2n += 1

        # contrastive (bridge batches)
        l1, l1n = torch.zeros((), device=device), 0
        for bi in bridge_batch_num_ids:
            meta = batch_train_meta_numbers[bi]
            n_cell, n_bs, n_batch, mods = meta[1], meta[2], meta[3], meta[5]
            if n_cell > n_bs:
                perm = torch.randperm(n_cell, device=device)
                for k in mods:
                    mod_embs[k][bi] = mod_embs[k][bi][perm]
            for i in range(n_batch):
                sl = slice(i*n_bs, (i+1)*n_bs)
                feats = [mod_embs[k][bi][sl] for k in mods]
                if isinstance(crit1, dict):
                    C = crit1[bi]
                    f = torch.cat(feats, dim=0)
                    l1 = l1 + C((f @ f.T) / T)
                else:
                    assert len(feats) == 2, "CE contrastive expects exactly 2 modalities"
                    C = crit1
                    logit = (feats[0] @ feats[1].T) / T
                    target = torch.arange(feats[0].size(0), device=device)
                    l1 = l1 + 0.5 * (C(logit, target) + C(logit.T, target))
                l1n += 1

        loss = (l1 / max(1, l1n)) + (l2 / max(1, l2n))
        loss.backward()
        for k in mod_optims: mod_optims[k].step()

        if l1n: loss_cl.append((l1 / max(1, l1n)).detach().item())
        if l2n: loss_rec.append((l2 / max(1, l2n)).detach().item())

    return mod_model, loss_cl, loss_rec
