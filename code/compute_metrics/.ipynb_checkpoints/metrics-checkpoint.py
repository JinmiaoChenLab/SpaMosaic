import os
import gc
import sys
import copy
import yaml
import h5py
import math
import torch
import scipy
import pickle
import warnings
# import episcanpy as epi
import matplotlib as mpl
import matplotlib.pyplot as plt

import anndata as ad
import numpy as np
import pandas as pd
import logging
import scanpy as sc
from os.path import join
import scipy.io as sio
import scipy.sparse as sps
from sklearn.cluster import KMeans
# import gzip
from scipy.io import mmread
from pathlib import Path, PurePath
from annoy import AnnoyIndex
import itertools
from scib.metrics import lisi
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

from scipy.spatial import *
from sklearn.preprocessing import *
from sklearn.metrics import *
from scipy.spatial.distance import *

def binarize(Xs, bin_thr=0):
    rs = []
    for X in Xs:
        X = copy.deepcopy(X.A) if sps.issparse(X) else copy.deepcopy(X)
        # X[X>bin_thr] = 1
        X = np.where(X>bin_thr, 1, 0)
        rs.append(X)
    return rs

def eval_AUC_all(gt_X, pr_X, bin_thr=1):
    gt_X = binarize([gt_X], bin_thr)[0].flatten()
    pr_X = pr_X.flatten()
    auroc = roc_auc_score(gt_X, pr_X)
    return auroc

def PCCs(gt_X, pr_X):
    pcc_cell = [np.corrcoef(gt_X[i,:], pr_X[i,:])[0,1] for i in range(gt_X.shape[0])] 
    pcc_feat = [np.corrcoef(gt_X[:,i], pr_X[:,i])[0,1] for i in range(gt_X.shape[1])] 
    return pcc_cell, pcc_feat

def CMD(pr_X, gt_X):
    zero_rows_indices1 = list(np.where(~pr_X.any(axis=1))[0]) # all-zero rows
    zero_rows_indices2 = list(np.where(~gt_X.any(axis=1))[0])
    zero_rows_indices = zero_rows_indices1 + zero_rows_indices2
    rm_p = len(zero_rows_indices) / pr_X.shape[0]
    if rm_p >= .05:
        print(f'Warning: two many rows {rm_p}% with all zeros')
    pr_array = pr_X[~np.isin(np.arange(pr_X.shape[0]), zero_rows_indices)].copy()
    gt_array = gt_X[~np.isin(np.arange(gt_X.shape[0]), zero_rows_indices)].copy()
    corr_pr = np.corrcoef(pr_array,dtype=np.float32)   # correlation matrix
    corr_gt = np.corrcoef(gt_array,dtype=np.float32)
    
    x = np.trace(corr_pr.dot(corr_gt))
    y = np.linalg.norm(corr_pr,'fro')*np.linalg.norm(corr_gt,'fro')
    cmd = 1- x/(y+1e-8)
    return cmd

def nn_annoy(ds1, ds2, norm=True, knn = 20, metric='euclidean', n_trees = 10):
    if norm:
        ds1 = normalize(ds1) 
        ds2 = normalize(ds2) 

    """ Assumes that Y is zero-indexed. """
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    return ind

def knn_smoothing(ad, hvf_name=None, dim_red_key='X_lsi', knn=50):
    knn_ind = nn_annoy(ad.obsm[dim_red_key], ad.obsm[dim_red_key], 
            norm=True, knn = knn+1, metric='manhattan', n_trees = 10)[:, 1:]
    hvf_name = hvf_name if hvf_name is not None else ad.var_names.to_list()
    X = ad[:, hvf_name].X.A if sps.issparse(ad.X) else ad[:, hvf_name].X
    smthed_X = np.mean(X[knn_ind.ravel()].reshape(X.shape[0], knn, X.shape[1]), axis=1)
    return smthed_X

def iLISI(adata, batch_key, use_rep):
    adata.obs[batch_key] = adata.obs[batch_key].astype('category')
    _lisi = lisi.ilisi_graph(
            adata,
            batch_key,
            'embed',
            use_rep=use_rep,
            k0=90,
            subsample=None,
            scale=True,
            n_cores=1,
            verbose=False,
        )
    return _lisi

def snn_scores(
        x, y, k=1
):
    '''
        return: matching score matrix
    '''

    # print(f'{k} neighbors to consider during matching')

    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)

    ky = k or min(round(0.01 * y.shape[0]), 1000)   
    nny = NearestNeighbors(n_neighbors=ky).fit(y)
    x2y = nny.kneighbors_graph(x)
    y2y = nny.kneighbors_graph(y)

    kx = k or min(round(0.01 * x.shape[0]), 1000)
    nnx = NearestNeighbors(n_neighbors=kx).fit(x)
    y2x = nnx.kneighbors_graph(y)
    x2x = nnx.kneighbors_graph(x)

    x2y_intersection = x2y @ y2y.T
    y2x_intersection = y2x @ x2x.T
    jaccard = x2y_intersection + y2x_intersection.T
    jaccard.data = jaccard.data / (2 * kx + 2 * ky - jaccard.data)
    matching_matrix = jaccard.multiply(1 / jaccard.sum(axis=1)).tocsr()
    return matching_matrix

def MS(
        mod1, mod2, split_by='batch', k=1, use_rep='X'
):  
    '''
        return: scipy.sparse.csr_matrix
    '''

    mod1_splits = set(mod1.obs[split_by])
    mod2_splits = set(mod2.obs[split_by])
    splits = mod1_splits | mod2_splits
    
    matching_matrices, mod1_obs_names, mod2_obs_names = [], [], []
    for split in splits:
        mod1_split = mod1[mod1.obs[split_by] == split]
        mod2_split = mod2[mod2.obs[split_by] == split]
        mod1_obs_names.append(mod1_split.obs_names)
        mod2_obs_names.append(mod2_split.obs_names)
        
        matching_matrices.append(
            snn_scores(mod1_split.X, mod2_split.X, k)
            if use_rep=='X' else
            snn_scores(mod1_split.obsm[use_rep], mod2_split.obsm[use_rep], k)
        )
        
    mod1_obs_names = pd.Index(np.concatenate(mod1_obs_names))
    mod2_obs_names = pd.Index(np.concatenate(mod2_obs_names))
    combined_matrix = scipy.sparse.block_diag(matching_matrices, format="csr")
    score_matrix = combined_matrix[
        mod1_obs_names.get_indexer(mod1.obs_names), :
    ][
        :, mod2_obs_names.get_indexer(mod2.obs_names)
    ]

    score = (score_matrix.diagonal() / score_matrix.sum(axis=1).A1).mean()
    return score

def batch_gpu_pairdist(emb1, emb2, batch_size=1024):
    n_batch = math.ceil(emb2.shape[0] / batch_size)
    emb2_gpu = torch.FloatTensor(emb2).cuda()
    emb2_gpu = emb2_gpu / torch.linalg.norm(emb2_gpu, ord=2, dim=1, keepdim=True)
    
    st = 0
    dist = []
    for i in range(n_batch):
        bsz = min(batch_size, emb1.shape[0] - i*batch_size)
        emb1_batch_gpu = torch.FloatTensor(emb1[st:st+bsz]).cuda()
        emb1_batch_gpu /= torch.linalg.norm(emb1_batch_gpu, ord=2, dim=1, keepdim=True)
        
        _ = -emb1_batch_gpu @ emb2_gpu.T  # 0-similarity => dist
        dist.append(_.cpu().numpy())
        st = st+bsz
        
        del emb1_batch_gpu
        torch.cuda.empty_cache()
        gc.collect()
    
    del emb2_gpu
    torch.cuda.empty_cache()
    gc.collect()
    
    dist = np.vstack(dist)
    return dist

def FOSCTTM(adata1, adata2, use_rep='X_emb'):
    dist = batch_gpu_pairdist(adata1.obsm[use_rep], adata2.obsm[use_rep], batch_size=2048)
    foscttm_x = (dist < dist.diagonal().reshape(-1, 1)).mean(axis=1)
    foscttm_y = (dist < dist.diagonal()).mean(axis=0)
    foscttm = (foscttm_x+foscttm_y).mean()/2

    return foscttm

def LabTransfer(ad1, ad2, use_rep, lab_key, knn=10):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        neigh1 = KNeighborsClassifier(n_neighbors=knn)
        neigh1.fit(ad1.obsm[use_rep], ad1.obs[lab_key].to_list())
        pr_lab2 = neigh1.predict(ad2.obsm[use_rep])
        f1_1 = f1_score(ad2.obs[lab_key].values, pr_lab2, 
                        average='macro')

        neigh2 = KNeighborsClassifier(n_neighbors=knn)
        neigh2.fit(ad2.obsm[use_rep], ad2.obs[lab_key].to_list())
        pr_lab1 = neigh2.predict(ad1.obsm[use_rep])
        f1_2 = f1_score(ad1.obs[lab_key].values, pr_lab1,
                        average='macro')
        return (f1_1+f1_2)/2

def compute_CHAOS2(adata,pred_key,spatial_key='spatial'):
    return _compute_CHAOS2(adata.obs[pred_key],adata.obsm[spatial_key])

def _compute_CHAOS2(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)

    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 2:
            continue
        nbrs = NearestNeighbors(n_neighbors=2).fit(location_cluster)
        distances, _ = nbrs.kneighbors(location_cluster)
        # distances[:, 0] is self-distance (0), distances[:, 1] is 1-NN
        dist_val[count] = np.sum(distances[:, 1])
        count += 1

    return np.sum(dist_val) / len(clusterlabel)

def compute_PAS2(adata, pred_key, spatial_key='spatial'):
    return _compute_PAS2(adata.obs[pred_key].values, adata.obsm[spatial_key])

def _compute_PAS2(clusterlabel, location, k=10):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)

    # Use NearestNeighbors to find k+1 neighbors (including self)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(location)
    distances, indices = nbrs.kneighbors(location)

    # Remove the first neighbor (it is the point itself)
    neighbor_indices = indices[:, 1:]

    # Check for label disagreement
    mismatches = np.sum([        np.sum(clusterlabel[neighbors] != clusterlabel[i]) > k / 2
        for i, neighbors in enumerate(neighbor_indices)
    ])

    return mismatches / len(clusterlabel)