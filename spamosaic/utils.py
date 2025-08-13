import os, gc
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
# import squidpy as sq
import pandas as pd
import h5py
import yaml
import math
import sklearn
from tqdm import tqdm
import scipy.sparse as sps
import scipy.io as sio
import seaborn as sns
import warnings
import networkx as nx

from os.path import join
import torch
from collections import Counter
import logging
from sklearn.decomposition import PCA
from annoy import AnnoyIndex
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

class Config:
    """
    A wrapper class that recursively converts a dictionary to an object with attribute-style access.

    Attributes:
        __dict__ (dict): Stores the nested configuration structure.
    """
    def __init__(self, dictionary):
        for k,v in dictionary.items():
            if isinstance(v, dict):
                v = Config(v)
            self.__dict__[k] = v
    
    def __getitem__(self, item):
        return self.__dict__[item]

    def __getattr__(self, item):
        return self.__dict__[item]
    
    def __repr__(self):
        return repr(self.__dict__)

def load_config(filepath):
    """
    Loads a YAML configuration file into a Config object.

    Args:
        filepath (str): Path to the YAML configuration file.

    Returns:
        Config: Parsed configuration object.
    """
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)

def check_batch_empty(modBatch_dict, verbose=True):
    """
    Checks if any batch in the modality-batch dictionary is empty and returns indices of modalities per batch.

    Args:
        modBatch_dict (dict): Dictionary of modality -> list of AnnData objects (or None).
        verbose (bool): Whether to print out batch composition.

    Returns:
        list[list[int]]: List of lists indicating indices of modalities present in each batch.

    Raises:
        ValueError: If any batch is completely empty.
    """
    mod_names = list(modBatch_dict.keys())
    n_batches = len(modBatch_dict[mod_names[0]])
    batch_contained_mod_ids = []
    for bi in range(n_batches):
        modIds_in_bi = []
        for mi, mod in enumerate(mod_names):
            if modBatch_dict[mod][bi] is not None:
                modIds_in_bi.append(mi)
        if len(modIds_in_bi) == 0:
            raise ValueError(f'batch {bi} empty')

        batch_contained_mod_ids.append(modIds_in_bi)
        if verbose:
            print(f'batch{bi}: {[mod_names[_] for _ in modIds_in_bi]}')
    return batch_contained_mod_ids

def get_barc2batch(modBatch_dict):
    """
    Creates a mapping from cell barcodes to their batch indices.

    Args:
        modBatch_dict (dict): Dictionary of modality -> list of AnnData objects.

    Returns:
        dict: Mapping from barcode to batch index.
    """
    mods = list(modBatch_dict.keys())
    n_batches = len(modBatch_dict[mods[0]])

    batch_list, barc_list = [], []
    for i in range(n_batches):
        for m in mods:
            if modBatch_dict[m][i] is not None:
                barc_list.extend(modBatch_dict[m][i].obs_names.to_list())
                batch_list.extend([i] * modBatch_dict[m][i].n_obs)
                break
    return dict(zip(barc_list, batch_list))

def nn_approx(ds1, ds2, norm=True, knn=10, metric='manhattan', n_trees=10, include_distances=False):
    """
    Performs approximate nearest neighbor search using Annoy.

    Args:
        ds1 (np.ndarray): Query data (n_samples, n_features).
        ds2 (np.ndarray): Reference data (n_samples, n_features).
        norm (bool): Whether to normalize data before search.
        knn (int): Number of nearest neighbors to retrieve.
        metric (str): Distance metric for Annoy (e.g., 'manhattan').
        n_trees (int): Number of trees used by Annoy.
        include_distances (bool): Whether to return distances with indices.

    Returns:
        np.ndarray or tuple: Indices of nearest neighbors, optionally with distances.
    """
    if norm:
        ds1 = normalize(ds1)
        ds2 = normalize(ds2) 

    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind, dist = [], []
    for i in range(ds1.shape[0]):
        i_ind, i_dist = a.get_nns_by_vector(ds1[i, :], knn, search_k=-1, include_distances=True)
        ind.append(i_ind)
        dist.append(i_dist)
    ind = np.array(ind)
    
    if include_distances:
        return ind, np.array(dist)
    else:
        return ind

# followed STAGATE
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2020):
    """
    Performs R's Mclust via rpy2.

    Args:
        adata (AnnData): AnnData object with embedding stored in `.obsm`.
        num_cluster (int): Desired number of clusters.
        modelNames (str): Covariance structure model in Mclust.
        used_obsm (str): Key in `.obsm` to use for clustering.
        random_seed (int): Random seed for reproducibility.

    Returns:
        AnnData: Annotated object with added `mclust` cluster label.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int').astype('category')
    return adata

def split_adata_ob(ads, ad_ref, ob='obs', key='emb'):
    """
    Splits a reference AnnData object's observations or embeddings into a list of AnnData objects.

    Args:
        ads (list[AnnData]): List of target AnnData objects.
        ad_ref (AnnData): Source AnnData object containing merged info.
        ob (str): Whether to split 'obs' or 'obsm'.
        key (str): Key to split in `.obs` or `.obsm`.
    """
    len_ads = [_.n_obs for _ in ads]
    if ob=='obsm':
        split_obsms = np.split(ad_ref.obsm[key], np.cumsum(len_ads[:-1]))
        for ad, v in zip(ads, split_obsms):
            ad.obsm[key] = v
    else:
        split_obs = np.split(ad_ref.obs[key].to_list(), np.cumsum(len_ads[:-1]))
        for ad, v in zip(ads, split_obs):
            ad.obs[key] = v

def clustering(adata, n_cluster, used_obsm, algo='kmeans', key='tmp_clust'):
    """
    Performs clustering on an AnnData object using k-means or Mclust.

    Args:
        adata (AnnData): Input data.
        n_cluster (int): Number of clusters.
        used_obsm (str): Key in `.obsm` to use for clustering.
        algo (str): Clustering algorithm, 'kmeans' or 'mclust'.
        key (str): Key name to store clustering results in `.obs`.

    Returns:
        AnnData: Annotated object with cluster assignments.
    """
    if algo == 'kmeans':
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(adata.obsm[used_obsm])
        adata.obs[key] = kmeans.labels_.astype('str')
    else:
        try:
            adata = mclust_R(adata, n_cluster, used_obsm=used_obsm)
            adata.obs[key] = adata.obs['mclust'].astype('str')
        except:
            print('mclust failed')
            kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(adata.obsm[used_obsm])
            adata.obs[key] = kmeans.labels_.astype('str')
    return adata

def get_umap(ad, use_reps=[]):
    """
    Computes UMAP embeddings for given representation keys and stores them in `.obsm`.

    Args:
        ad (AnnData): Input data.
        use_reps (list[str]): Keys in `.obsm` to compute UMAP for.

    Returns:
        AnnData: Annotated object with new UMAPs.
    """
    for use_rep in use_reps:
        umap_add_key = f'{use_rep}_umap'
        sc.pp.neighbors(ad, use_rep=use_rep, n_neighbors=15)
        sc.tl.umap(ad)
        ad.obsm[umap_add_key] = ad.obsm['X_umap']
    return ad

def plot_basis(ad, basis, color, **kwargs):
    """
    Wrapper for Scanpy's `sc.pl.embedding` with warning suppression.

    Args:
        ad (AnnData): Annotated data object.
        basis (str): Key of embedding (e.g., 'umap').
        color (str): Variable in `.obs` to color by.
        **kwargs: Additional plotting arguments.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        sc.pl.embedding(ad, basis=basis, color=color, **kwargs)

def flip_axis(ads, axis=0):
    """
    Flips the spatial coordinates of AnnData objects along a specified axis.

    Args:
        ads (list[AnnData]): List of data objects to modify.
        axis (int): Axis to flip (0 for x, 1 for y).
    """
    for ad in ads:
        ad.obsm['spatial'][:, axis] = -1 * ad.obsm['spatial'][:, axis]

def reorder(ad1, ad2):
    """
    Aligns and reorders two AnnData objects to only shared barcodes.

    Args:
        ad1 (AnnData): First data object.
        ad2 (AnnData): Second data object.

    Returns:
        Tuple[AnnData, AnnData]: Reordered and aligned data objects.
    """
    shared_barcodes = ad1.obs_names.intersection(ad2.obs_names)
    ad1 = ad1[shared_barcodes].copy()
    ad2 = ad2[shared_barcodes].copy()
    return ad1, ad2

def dict_map(_dict, _list):
    """
    Applies dictionary mapping over a list.

    Args:
        _dict (dict): Dictionary to map from.
        _list (list): List of keys.

    Returns:
        list: Mapped values.
    """
    return [_dict[x] for x in _list]
