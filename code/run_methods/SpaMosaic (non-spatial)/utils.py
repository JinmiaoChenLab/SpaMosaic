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
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    return Config(config_dict)

def check_batch_empty(modBatch_dict, verbose=True):
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
    for use_rep in use_reps:
        umap_add_key = f'{use_rep}_umap'
        sc.pp.neighbors(ad, use_rep=use_rep, n_neighbors=15)
        sc.tl.umap(ad)
        ad.obsm[umap_add_key] = ad.obsm['X_umap']
    return ad

def plot_basis(ad, basis, color, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        sc.pl.embedding(ad, basis=basis, color=color, **kwargs)

def flip_axis(ads, axis=0):
    for ad in ads:
        ad.obsm['spatial'][:, axis] = -1 * ad.obsm['spatial'][:, axis]

def reorder(ad1, ad2):
    shared_barcodes = ad1.obs_names.intersection(ad2.obs_names)
    ad1 = ad1[shared_barcodes].copy()
    ad2 = ad2[shared_barcodes].copy()
    return ad1, ad2

def dict_map(_dict, _list):
    return [_dict[x] for x in _list]
