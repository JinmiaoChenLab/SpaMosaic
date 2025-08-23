"""Graph construction utilities for spatial transcriptomics.

This module provides helpers to build spatial neighbor graphs and cross-batch
mutual nearest neighbor (MNN) matches on top of AnnData objects.
"""

import os, gc
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
# import squidpy as sq
import h5py
import math
import sklearn
from tqdm import tqdm
import scipy.sparse as sps
import scipy.io as sio
import seaborn as sns
import warnings
import networkx as nx

from os.path import join
from collections import Counter
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from annoy import AnnoyIndex
from sklearn.preprocessing import normalize
from sklearn.ensemble import IsolationForest

import spamosaic.MNN as MNN

RAD_CUTOFF = 2000


# adapted from STAGATE
def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None,
                    max_neigh=50, model='Radius', verbose=True):
    """Construct spatial graph from spatial coordinates using radius or kNN method.

    Parameters
    ----------
    adata : AnnData
        Input annotated data.
    rad_cutoff : float, optional
        Distance cutoff for radius-based graph.
    k_cutoff : int, optional
        Number of neighbors for kNN-based graph.
    max_neigh : int
        Maximum number of neighbors to consider.
    model : str
        Type of graph construction: 'Radius' or 'KNN'.
    verbose : bool
        Whether to print debug info.

    Returns
    -------
    None

    Notes
    -----
    On success, adds the adjacency matrix to ``adata.uns['adj']`` (SciPy sparse matrix).
    """
    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    cells = np.array(adata.obs.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sps.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sps.eye(G.shape[0])  # self-loop
    adata.uns['adj'] = G


def build_intra_graph(ads, rad_cutoff, knns):
    """Construct intra-batch spatial graphs for a list of AnnData objects.

    Parameters
    ----------
    ads : list of AnnData
        List of spatial AnnData objects. ``None`` entries are skipped.
    rad_cutoff : float
        Distance threshold for the radius graph.
    knns : list of int
        Maximum neighbors to query for each AnnData (same length as ``ads``).

    Returns
    -------
    None
    """
    for k, ad in zip(knns, ads):
        if ad is not None:
            Cal_Spatial_Net(ad, rad_cutoff=rad_cutoff, max_neigh=k)


def make_Ahat_sparse(A, improved=False, symm=False):
    """Build a normalized adjacency matrix ``A_hat`` suitable for GNN layers.

    Given a sparse adjacency ``A`` (N×N), this function optionally symmetrizes it,
    adds self-loops, and applies the symmetric normalization
    :math:`D^{-1/2} A D^{-1/2}`.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse adjacency matrix of shape (N, N).
    improved : bool, optional
        If ``True``, use a self-loop weight of 2.0 (as in "improved" GCN).
    symm : bool, optional
        If ``True``, force symmetry by replacing ``A`` with ``(A + A.T)`` before normalization.

    Returns
    -------
    scipy.sparse.csr_matrix
        Normalized sparse matrix ``A_hat`` of shape (N, N).
    """
    N = A.shape[0]
    A = A.tocoo(copy=True)
    A.setdiag(0)

    A.eliminate_zeros()
    A_sym = A.tocoo(copy=True)
    if symm:
        A_sym = (A + A.T).tocoo()

    fill = 2.0 if improved else 1.0
    A_sym = (A_sym + sps.eye(N, dtype=np.float32, format="coo") * fill).tocoo()

    deg = np.asarray(A_sym.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    r, c, v = A_sym.row, A_sym.col, A_sym.data.astype(np.float32)
    v = v * d_inv_sqrt[r] * d_inv_sqrt[c]

    A_hat = sps.csr_matrix((v, (r, c)), shape=(N, N), dtype=np.float32)
    return A_hat


def determine_kSize(adi, adj, knn_base, auto_thr):
    """Determine asymmetric k values for MNN search based on dataset sizes.

    If two datasets have similar numbers of observations (the smaller-to-larger ratio
    is at least ``auto_thr``), both sides use ``knn_base``. Otherwise, the smaller side
    uses ``floor(knn_base * size_ratio)`` (at least 1).

    Parameters
    ----------
    adi : AnnData
        First dataset.
    adj : AnnData
        Second dataset.
    knn_base : int
        Base number of neighbors.
    auto_thr : float
        Size similarity threshold, e.g. 0.8.

    Returns
    -------
    tuple of int
        ``(knn_adi, knn_adj)`` to use for the pair.
    """
    size_ratio = min(adi.n_obs, adj.n_obs) / max(adi.n_obs, adj.n_obs)
    if size_ratio >= auto_thr:
        return knn_base, knn_base

    if adi.n_obs > adj.n_obs:
        return max(1, int(knn_base * size_ratio)), knn_base
    else:
        return knn_base, max(1, int(knn_base * size_ratio))


def remove_outlier(mnn_set, ad1, ad2, contamination='auto'):
    """Filter spatial outliers from an MNN pair set using Isolation Forest.

    A feature matrix is built from the concatenated spatial coordinates of both
    cells and their differences. Pairs predicted as outliers (label ``-1``) are
    removed.

    Parameters
    ----------
    mnn_set : set of tuple of str
        Set of MNN barcode pairs ``(cell_id_1, cell_id_2)``.
    ad1 : AnnData
        Dataset of the first cell; must contain ``.obsm['spatial']``.
    ad2 : AnnData
        Dataset of the second cell; must contain ``.obsm['spatial']``.
    contamination : {'auto', float}, optional
        Expected outlier fraction; passed to ``sklearn.ensemble.IsolationForest``.

    Returns
    -------
    set of tuple of str
        Filtered set with spatial outliers removed.
    """
    X_spatial_1 = ad1[[p[0] for p in mnn_set]].obsm['spatial']
    X_spatial_2 = ad2[[p[1] for p in mnn_set]].obsm['spatial']
    data = np.c_[X_spatial_1, X_spatial_2, X_spatial_1 - X_spatial_2]

    clf = IsolationForest(max_samples='auto', contamination=contamination, random_state=0)
    y_outlier = clf.fit_predict(data)

    new_mnn_set = set()
    for p, y in zip(mnn_set, y_outlier):
        if y == 1:
            new_mnn_set.add(p)

    return new_mnn_set


def build_mnn_graph(
    bridge_ads, test_ads, use_rep, batch_key,
    knn_base=10, auto_knn=False, auto_thr=0.8,
    rmv_outlier=False, contamination='auto', seed=1234
):
    """Build a mutual nearest neighbor (MNN) graph across batches in one modality.

    This procedure matches cells across batches using approximate kNN (Annoy) on a
    given embedding, returning a set of matched barcode pairs. It supports (1)
    bridge–bridge matches (between multi-modal batches) and (2) bridge–test matches
    (between a bridge batch and a single-modality batch). Optionally, it can remove
    spatial outliers.

    Parameters
    ----------
    bridge_ads : list of AnnData
        AnnData objects from bridge batches.
    test_ads : list of AnnData
        AnnData objects from test batches.
    use_rep : str
        Key in ``.obsm`` containing the embedding to search.
    batch_key : str
        Column in ``.obs`` used only for logging/batch names.
    knn_base : int, optional
        Base number of neighbors per side. Default is 10.
    auto_knn : bool, optional
        If ``True``, use ``determine_kSize`` to adapt k based on batch sizes.
    auto_thr : float, optional
        Size ratio threshold for ``auto_knn``. Default is 0.8.
    rmv_outlier : bool, optional
        Whether to run ``remove_outlier``. Default is ``False``.
    contamination : {'auto', float}, optional
        Outlier rate used by Isolation Forest.
    seed : int, optional
        Random seed for stochastic components.

    Returns
    -------
    set of tuple of str
        Set of matched barcode pairs across batches.
    """
    n_bridge = len(bridge_ads)
    n_test = len(test_ads)

    # If no bridge available, treat test as reference
    if n_bridge == 0:
        bridge_ads = test_ads
        n_bridge = len(bridge_ads)
        n_test = 0  # No test mode

    mnn_bridge = set()
    mnn_cross = set()

    def compute_mnn(adi, adj):
        # Determine KNN size
        if auto_knn:
            knn1, knn2 = determine_kSize(adi, adj, knn_base, auto_thr)
        else:
            knn1 = knn2 = knn_base

        # Log pair info
        src_name = adi.obs[batch_key][0]
        tgt_name = adj.obs[batch_key][0]
        print(f"Finding MNN between ({src_name}, {tgt_name}) using KNN ({knn1}, {knn2})")

        # Run approximate MNN search
        mnn_pairs = MNN.mnn(
            adi.obsm[use_rep], adj.obsm[use_rep],
            adi.obs_names, adj.obs_names,
            knn1=knn1, knn2=knn2,
            approx=True, way='annoy', metric='manhattan', norm=True
        )

        # Optionally filter outliers
        if rmv_outlier:
            n_before = len(mnn_pairs)
            mnn_pairs = remove_outlier(mnn_pairs, adi, adj, contamination)
            n_after = len(mnn_pairs)
            if n_before > 0:
                print(f"==> Filtered {n_before - n_after} from {n_before} ({(n_before - n_after) / n_before * 100:.2f}%)")

        return mnn_pairs

    # Bridge-bridge connections
    for i in range(n_bridge):
        for j in range(i + 1, n_bridge):
            mnn_bridge |= compute_mnn(bridge_ads[i], bridge_ads[j])

    # Bridge-test connections
    for i in range(n_bridge):
        for j in range(n_test):
            mnn_cross |= compute_mnn(bridge_ads[i], test_ads[j])

    # Return combined MNN graph
    return mnn_bridge | mnn_cross
