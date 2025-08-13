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

def graph_recon_crit(z, pos_edge_index):
    """
    Compute binary cross-entropy loss for graph reconstruction using positive and negative edges.

    Parameters
    ----------
    z : torch.Tensor
        Node embeddings of shape [num_nodes, embedding_dim].
    pos_edge_index : torch.Tensor
        Positive edge indices of shape [2, num_pos_edges].

    Returns
    -------
    torch.Tensor
        Scalar reconstruction loss combining positive and negative samples.
    """

    pos_loss = -torch.log(graph_decode(z, pos_edge_index)+1e-20)
    pos_loss = pos_loss.mean()

    neg_edge_index = negative_sampling(
        pos_edge_index, z.size(0), num_neg_samples=pos_edge_index.size(1)
    )
    # neg_edge_index = self.subgraph_negative_sampling(pos_edge_index)
    neg_loss = -torch.log(1 - graph_decode(z, neg_edge_index) + 1e-20)
    neg_loss = neg_loss.mean()
    return pos_loss + neg_loss

def train_model(
        mod_model, mod_optims, crit1, crit2, loss_type, w_rec_g, 
        mod_input_feat, mod_graphs_edge, mod_graphs_edge_w, mod_graphs_edge_c, 
        batch_train_meta_numbers, mod_batch_split, 
        T, n_epochs, device
    ):
    """
    Train multi-modal GNN model with contrastive and reconstruction losses.

    Parameters
    ----------
    mod_model : dict[str, torch.nn.Module]
        Dictionary mapping modality to its encoder model.
    mod_optims : dict[str, torch.optim.Optimizer]
        Optimizer for each modality.
    crit1 : dict[int, Callable] or torch.nn.Module
        Contrastive loss function or dict of such functions per batch.
    crit2 : Callable
        Reconstruction loss function (e.g., MSELoss).
    loss_type : str
        Either 'adapted' (multi-modality) or 'standard' (2-modality) contrastive loss.
    w_rec_g : float
        Weight for graph reconstruction loss; (1 - w_rec_g) used for feature reconstruction.
    mod_input_feat : dict[str, torch.Tensor]
        Input features per modality.
    mod_graphs_edge : dict[str, torch.Tensor]
        Edge indices per modality.
    mod_graphs_edge_w : dict[str, torch.Tensor]
        Edge weights per modality.
    mod_graphs_edge_c : dict[str, torch.Tensor]
        Batch-specific edge masks (used to isolate subgraphs).
    batch_train_meta_numbers : dict[int, tuple]
        Metadata per batch describing whether it is bridge/test, size, and modalities.
    mod_batch_split : dict[str, list[int]]
        Mapping modality to list of cell counts per batch.
    T : float
        Temperature for contrastive loss.
    n_epochs : int
        Number of training epochs.
    device : torch.device
        Device used for training.

    Returns
    -------
    tuple
        Trained model dict, list of contrastive losses, list of reconstruction losses.
    """
    bridge_batch_num_ids = [bi for bi,v in batch_train_meta_numbers.items() if v[0]]
    test_batch_num_ids   = [bi for bi,v in batch_train_meta_numbers.items() if not v[0]]
    
    loss_cl, loss_rec = [], []
    for epoch in tqdm(range(1, n_epochs+1)):
        for k in mod_model.keys(): mod_model[k].train()
        for k in mod_optims.keys(): mod_optims[k].zero_grad() 

        mod_embs, mod_recs = {}, {}
        for k in mod_model.keys():
            z, r = mod_model[k](mod_input_feat[k], mod_graphs_edge[k], mod_graphs_edge_w[k])  # outputs for all spots within each modality 
            mod_embs[k] = list(torch.split(z, mod_batch_split[k]))  
            mod_recs[k] = list(torch.split(r, mod_batch_split[k]))   
        split_input_feat = {k: list(torch.split(v, mod_batch_split[k])) for k,v in mod_input_feat.items()}  # target of reconstruction 

        # feature reconstruction
        l2, l2_step = torch.FloatTensor([0]).to(device), 0  # in case empty test batches
        for bi in test_batch_num_ids:
            ms = batch_train_meta_numbers[bi][1]
            for k in ms:
                # feature reconst
                f_rec_loss = 0 if w_rec_g==1 else crit2(mod_recs[k][bi], split_input_feat[k][bi]) 
                
                if w_rec_g > 0:
                    # graph reconst
                    intra_graph_mask = mod_graphs_edge_c[k] == bi
                    sub_graph = mod_graphs_edge[k][:, intra_graph_mask]
                    start_idx = torch.unique(sub_graph).min()
                    sub_graph_edge_index = sub_graph - start_idx
                    g_rec_loss = graph_recon_crit(mod_embs[k][bi], sub_graph_edge_index)
                else:
                    g_rec_loss = 0.

                l2 += w_rec_g * g_rec_loss + (1-w_rec_g) * f_rec_loss
                l2_step += 1

        # contrastive learning
        l1, l1_step = 0, 0 
        for bi in bridge_batch_num_ids:
            meta = batch_train_meta_numbers[bi]
            # if setting multiple mini-batches, randomly shuffle the spots 
            if meta[3] > 1:                       
                shuffle_ind = torch.randperm(meta[1]).to(device)
                for k in meta[5]: 
                    mod_embs[k][bi] = mod_embs[k][bi][shuffle_ind]

            for i in range(meta[3]):
                feat = [mod_embs[k][bi][i*meta[2]:(i+1)*meta[2]] for k in meta[5]]
                if loss_type=='adapted':
                    feat = torch.cat(feat)
                    l1 += crit1[bi]((feat @ feat.t()) / T)
                else:
                    logit = (feat[0] @ feat[1].T)/T
                    target = torch.arange(meta[2]).to(device)
                    l1 += 0.5 * (crit1[bi](logit, target) + crit1[bi](logit.T, target))
                l1_step += 1
        
        loss = l1/max(1, l1_step) + l2/max(1, l2_step) 

        loss.backward()
        for k in mod_optims.keys(): mod_optims[k].step()

        if l1_step > 0:
            loss_cl.append(l1.item())
        if l2_step > 0:
            loss_rec.append(l2.item())

    return mod_model, loss_cl, loss_rec 
