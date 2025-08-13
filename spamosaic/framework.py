import os, gc
from collections.abc import Iterable
from pathlib import Path, PurePath
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
import math
from tqdm import tqdm
import scipy.sparse as sps
import warnings
import itertools
from os.path import join
import torch

import logging
import torch
from matplotlib import rcParams
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import GAE
from torch_geometric.utils import train_test_split_edges, negative_sampling

from spamosaic.train_utils import set_seeds, train_model
import spamosaic.utils as utls
import spamosaic.build_graph as build_graph
import spamosaic.architectures as archs
from spamosaic.loss import CL_loss

cur_dir = os.path.dirname(os.path.abspath(__file__))

class SpaMosaic(object):
    """
    SpaMosaic: A modular framework for multi-modal spatial omics integration.

    This class manages data pre-processing, intra- and inter-batch graph construction, 
    model initialization and training, feature alignment, and imputation across multiple
    omics modalities (e.g., RNA, ADT, ATAC) in spatial transcriptomics.

    Parameters
    ----------
    modBatch_dict : dict
        Dictionary mapping modality name (e.g., 'rna', 'adt') to a list of AnnData objects (batches).
    input_key : str
        Key in `obsm` where input features are stored (e.g., 'dimred_bc').
    mnn_rep_key : str, optional
        Key for representation used in MNN search. If None, defaults to `input_key`.
    batch_key : str
        Column name in `obs` denoting batch identity.
    radius_cutoff : int
        Radius threshold to construct spatial neighbor graph.
    intra_knns : int or list of int
        Number of neighbors in intra-batch graph (can be int or per-batch list).
    inter_knn_base : int
        Base KNN size for inter-batch MNN search.
    smooth_input : bool
        Whether to apply GCN-based input feature smoothing.
    smooth_L : int
        Number of GCN layers used in smoothing.
    inter_auto_knn : bool
        Whether to adapt KNN size based on batch size ratio.
    inter_auto_thr : float
        Size ratio threshold to apply adaptive KNN.
    rmv_outlier : bool
        Whether to remove outlier MNN pairs.
    contamination : str or float
        Contamination level for outlier detection (used in IsolationForest).
    w_g : float
        Weight for inter-batch expression edges in the merged graph.
    log_dir : str, optional
        Directory for saving logs or results.
    seed : int
        Random seed.
    num_workers : int
        Number of workers used for computation.
    device : str
        Device to use, e.g., 'cuda:0' or 'cpu'.
    """
    def __init__(
        self, 
        modBatch_dict={},  # dict={'rna':[batch1, None, ...], 'adt':[batch1, batch2, ...], ...}
        input_key='dimred_bc',
        mnn_rep_key=None,
        batch_key='batch', 
        radius_cutoff=2000, intra_knns=10, inter_knn_base=10, 
        smooth_input=False, smooth_L=1,
        inter_auto_knn=False, inter_auto_thr=0.8,
        rmv_outlier=False, contamination='auto',
        w_g=0.8, log_dir=None,
        seed=1234, num_workers=6,
        device='cuda:0'
    ):  
        if 'cuda' in device and not torch.cuda.is_available():
            warnings.warn("CUDA was requested but no CUDA device is available. Falling back to CPU.")
        self.device = torch.device(device) if ('cuda' in device) and torch.cuda.is_available() else torch.device('cpu')
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        set_seeds(seed)
        self.radius_cutoff = radius_cutoff
        self.w_g = w_g
        self.seed = seed
        self.num_workers = num_workers
        self.rmv_outlier = rmv_outlier
        self.contamination = contamination

        self.smooth_input = smooth_input
        self.smooth_L = smooth_L

        self.input_key = input_key
        self.mnn_rep_key = input_key if mnn_rep_key is None else mnn_rep_key
        self.mod_list = np.array(list(modBatch_dict.keys()))
        self.n_mods = len(self.mod_list)
        self.n_batches = len(modBatch_dict[self.mod_list[0]])
        self.batch_key = batch_key
        self.barc2batch = utls.get_barc2batch(modBatch_dict)

        self.intra_knns = intra_knns if isinstance(intra_knns, Iterable) else [intra_knns]*self.n_batches
        self.inter_knn_base = inter_knn_base
        self.inter_auto_knn = inter_auto_knn
        self.inter_auto_thr = inter_auto_thr
        self.use_expr_adj = (w_g!=0)

        # check if there is empty batch
        self.batch_contained_mod_ids = utls.check_batch_empty(modBatch_dict)

        # check if this dataset can be integrated
        self.check_integrity()

        # prepare spot-spot for each modality
        self.prepare_intra_graphs(modBatch_dict)

        if self.smooth_input:
            self.apply_smoothing(modBatch_dict)

        self.prepare_inter_graphs(modBatch_dict)
        self.prepare_inputs(modBatch_dict)

    def check_integrity(self):
        """
        Check if all modalities across batches form a connected integration graph.

        Raises
        ------
        RuntimeError
            If the graph of shared modalities across batches is not fully connected.
        """
        mod_graph = np.zeros((self.n_mods, self.n_mods))
        for bi in range(self.n_batches):
            modIds_in_bi = self.batch_contained_mod_ids[bi]
            mod_pairs = np.array(list(itertools.product(modIds_in_bi, modIds_in_bi)))
            mod_graph[mod_pairs[:, 0], mod_pairs[:, 1]] = 1
        n_cs, labels = connected_components(mod_graph, directed=False, return_labels=True)
        if n_cs > 1:
            for ci in np.unique(labels):
                ni_msk = labels == ci
                print(f'conn {ci}:', self.mod_list[ni_msk])
            raise RuntimeError('Dataset not connected, cannot be integrated')

    def prepare_intra_graphs(self, modBatch_dict):
        """
        Build spatial neighbor graphs for each modality across batches.

        Parameters
        ----------
        modBatch_dict : dict
            Dictionary mapping modality to list of AnnData objects.
        """
        # spatial-neighbor graph
        for key in self.mod_list:
            # ads = [ad for ad in modBatch_dict[key] if ad is not None]
            build_graph.build_intra_graph(modBatch_dict[key], rad_cutoff=self.radius_cutoff, knns=self.intra_knns)

    def apply_smoothing(self, modBatch_dict):
        """
        Apply feature smoothing using LGCN on intra-batch graphs.

        Parameters
        ----------
        modBatch_dict : dict
            Dictionary mapping modality to list of AnnData objects.
        """
        for mod in self.mod_list:
            model = archs.wlgcn.WLGCN_vanilla(K=self.smooth_L).to(self.device)

            for ad in modBatch_dict[mod]:
                if ad is None:
                    continue

                attr = torch.tensor(ad.obsm[self.input_key], dtype=torch.float32).to(self.device)
                adj = ad.uns['adj'].tocoo()
                edge_index = torch.tensor(np.vstack((adj.row, adj.col)), dtype=torch.long).to(self.device)

                with torch.no_grad():
                    smth_attr = model(attr, edge_index).cpu().numpy()
                    ad.obsm[self.input_key+'_smth'] = smth_attr

        self.mnn_rep_key = self.input_key + '_smth'

    
    def prepare_inter_graphs(self, modBatch_dict):
        """
        Build mutual nearest neighbor (MNN) graphs between batches for each modality.
        
        This includes:
        - Identifying bridge and non-bridge batches.
        - Computing MNN pairs within each modality.
        - Optionally filtering outliers.
        """

        # determine which batches as bridges, which not. 
        mod_mask = np.zeros((self.n_mods, self.n_batches))
        for bi in range(self.n_batches):
            for ki,k in enumerate(self.mod_list):
                if modBatch_dict[k][bi] is not None:
                    mod_mask[ki][bi] = 1

        self.bridge_batch_num_ids = np.where(mod_mask.sum(axis=0) >= 2)[0]  # e.g., [idx, idx2, ]
        self.non_bridge_batch_num_ids = np.where(mod_mask.sum(axis=0) < 2)[0]

        ### prepare meta parameter for training
        self.mod_batch_split = {   # {'rna':[n_obs, 0, ...]}
            key: [
                modBatch_dict[key][bi].n_obs 
                if modBatch_dict[key][bi] is not None 
                else 0 
                for bi in range(self.n_batches)
            ]
            for key in self.mod_list
        }

        # mapping batch id to their modality set
        # e.g., 1 -> ['rna', 'adt'], 2 -> ['atac']
        self.bridge_batch_num_ids2mod = {
            bi: [key for key in self.mod_list if modBatch_dict[key][bi] is not None]
            for bi in self.bridge_batch_num_ids
        }
        self.non_bridge_batch_num_ids2mod = {
            bi: [key for key in self.mod_list if modBatch_dict[key][bi] is not None]
            for bi in self.non_bridge_batch_num_ids
        }

        bridge_ads = {
            key: [modBatch_dict[key][aid] for aid in self.bridge_batch_num_ids if modBatch_dict[key][aid] is not None]
            for key in self.mod_list
        }
        test_ads = {
            key: [modBatch_dict[key][aid] for aid in self.non_bridge_batch_num_ids if modBatch_dict[key][aid] is not None]
            for key in self.mod_list
        }
        
        if self.use_expr_adj:
            mod_mnn_set = {}
            for key in self.mod_list:
                print(f'Searching MNN within {key}')
                mod_mnn_set[key] = build_graph.build_mnn_graph(
                    bridge_ads[key], test_ads[key], self.mnn_rep_key, self.batch_key,
                    self.inter_knn_base, self.inter_auto_knn, self.inter_auto_thr, 
                    self.rmv_outlier, self.contamination,
                    self.seed
                )
                print('Number of mnn pairs for {}:{}'.format(key, len(mod_mnn_set[key])))
        else:
            mod_mnn_set = {
                key: []
                for key in self.mod_list
            }
        self.mod_mnn_set = mod_mnn_set
        
    def prepare_inputs(self, modBatch_dict):
        """
        Merge AnnData objects and construct final PyTorch graph inputs.

        Includes:
        - Concatenating features and adjacency matrices.
        - Adding intra- and inter-batch edges.
        - Computing edge types (same-batch or cross-batch).
        """
        mod_graphs = {}
        mod_graphs['edge'], mod_graphs['edgeW'], mod_graphs['edgeC'] = {}, {}, {}
        mod_graphs['attribute'], mod_graphs['attribute_dim'] = {}, {}
        inp_key = self.input_key+'_smth' if self.smooth_input else self.input_key 

        self.mod_input_ads = {}
        self.mod_graph_idx2barc = {}

        for k in self.mod_list:
            ads = [ad for ad in modBatch_dict[k] if ad is not None]
            mnn_set = self.mod_mnn_set[k]

            # === mergeGraph logic inlined ===
            concat_obs = pd.concat([ad.obs.copy() for ad in ads])
            concat_obsm = np.vstack([ad.obsm[inp_key] for ad in ads])
            ad_merge = sc.AnnData(np.zeros((len(concat_obs), 2)), obs=concat_obs, obsm={inp_key: concat_obsm})

            # intra graph: block diagonal
            intra_graph = sps.block_diag([ad.uns['adj'] for ad in ads]).tocoo()

            # build inter-graph from mnn_set
            name2idx = dict(zip(ad_merge.obs_names, np.arange(ad_merge.n_obs)))
            idx2name = {v: k for k, v in name2idx.items()}
            # self.mod_graph_idx2barc[k] = idx2name

            if len(mnn_set) > 0:
                mnn_i = np.array([[name2idx[e[0]], name2idx[e[1]]] for e in mnn_set])
                mnn_symm = np.vstack([mnn_i, mnn_i[:, ::-1]])
                inter_graph = sps.coo_matrix(
                    (np.ones(len(mnn_symm)), (mnn_symm[:, 0], mnn_symm[:, 1])),
                    shape=(ad_merge.n_obs, ad_merge.n_obs)
                ) * self.w_g
            else:
                inter_graph = sps.coo_matrix((ad_merge.n_obs, ad_merge.n_obs))

            # ad_merge.uns['mnn_graph'] = inter_graph

            if self.use_expr_adj:
                merged_graph = (intra_graph + inter_graph).tocoo()
            else:
                merged_graph = intra_graph

            # === store merged adata ===
            ad_merge.uns['adj'] = merged_graph
            ad_merge.uns['edgeList'] = (merged_graph.row, merged_graph.col)
            ad_merge.uns['edgeW'] = merged_graph.data
            # self.mod_input_ads[k] = ad_merge

            # === convert to PyTorch inputs ===
            edge_idx = np.vstack([merged_graph.row, merged_graph.col])
            mod_graphs['edge'][k] = torch.tensor(edge_idx, dtype=torch.long).to(self.device)
            mod_graphs['edgeW'][k] = torch.tensor(merged_graph.data, dtype=torch.float32).to(self.device)
            mod_graphs['attribute'][k] = torch.tensor(ad_merge.obsm[inp_key], dtype=torch.float32).to(self.device)
            mod_graphs['attribute_dim'][k] = ad_merge.obsm[inp_key].shape[1]

            # === edgeC: intra vs inter (used for smoothing etc.)
            barc1 = utls.dict_map(idx2name, merged_graph.row)
            barc2 = utls.dict_map(idx2name, merged_graph.col)
            b1 = np.array(utls.dict_map(self.barc2batch, barc1))
            b2 = np.array(utls.dict_map(self.barc2batch, barc2))
            edgeC = np.where(b1 == b2, b1, -1)
            mod_graphs['edgeC'][k] = torch.tensor(edgeC, dtype=torch.long).to(self.device)
        
        self.mod_graphs = mod_graphs

    def prepare_net(self, net):
        """
        Instantiate the architecture for each modality based on config.

        Parameters
        ----------
        net : str
            Name of model architecture (must match a config YAML file).

        Returns
        -------
        dict
            Mapping from modality name to initialized PyTorch model.
        """
        config = utls.load_config(f'{cur_dir}/configs/{net}.yaml')
        
        mod_model = {}
        for k in self.mod_list:
            encoder = archs.wlgcn.WLGCN(self.mod_graphs['attribute_dim'][k], config.model.out_dim, K=config.model.n_layer, dec_l=config.model.n_dec_l, 
                                        hidden_size=config.model.hid_dim, dropout=config.model.dropout, slope=config.model.slope)
            mod_model[k] = encoder.to(self.device)
            
        return mod_model

    def train(self, net, lr, use_mini_thr=8000, mini_batch_size=1024, loss_type='adapted', T=0.01, bias=0, n_epochs=100, w_rec_g=0.):
        """
        Train the SpaMosaic model using contrastive and reconstruction losses.

        Parameters
        ----------
        net : str
            Architecture name.
        lr : float
            Learning rate.
        use_mini_thr : int
            Threshold above which mini-batch training is used.
        mini_batch_size : int
            Size of mini-batches if needed.
        loss_type : str
            Type of loss ('adapted' or 'ce').
        T : float
            Temperature for contrastive loss.
        bias : float
            Bias term in adapted contrastive loss.
        n_epochs : int
            Number of training epochs.
        w_rec_g : float
            Weight for reconstruction loss.
        """
        # set model architectures
        mod_model = self.prepare_net(net)

        # set optimizer
        mod_optims = {k:torch.optim.Adam(mod_model[k].parameters(), lr=lr, weight_decay=5e-4) for k in self.mod_list}

        # for each batch, the meaning of parameter:
        # bridge_batch_meta_numbers = [{if as bridge}, {number of spots}, {size of mini-batches}, {number of measured modalites}, {measured modalities}]
        # test_batch_meta_numbers   = [{if as bridge}, {measured modalities}]
        batch_train_meta_numbers = {}
        for bi, ms in self.bridge_batch_num_ids2mod.items():
            n_cell = self.mod_batch_split[ms[0]][bi]
            n_loss_batch = n_cell if n_cell <= use_mini_thr else mini_batch_size  # determine mini-batch size for each batch
            n_batch = n_cell // n_loss_batch                                      # size of mini-batches
            batch_train_meta_numbers[bi] = (True, n_cell, n_loss_batch, n_batch, len(ms), ms)
        for bi, ms in self.non_bridge_batch_num_ids2mod.items():
            batch_train_meta_numbers[bi] = (False, ms)
        # print(batch_train_meta_numbers)

        # detetermine contrastive learning loss 
        if loss_type=='adapted':  # use our proposed contrastive learning loss, which can handle alignment of three or more modalities
            crit1 = {
                bi:CL_loss(batch_train_meta_numbers[bi][2], rep=batch_train_meta_numbers[bi][4], bias=bias).to(self.device)
                for bi in self.bridge_batch_num_ids
            }
        else:  
            crit1 = nn.CrossEntropyLoss().to(self.device)  # canonical implementation for CL loss, can only handle 2-modal alignment

        # feature reconstruction loss
        crit2 = nn.MSELoss().to(self.device)  

        # mod_input_feat, mod_graphs_edge, mod_graphs_edge_w
        mod_model, loss_cl, loss_rec = train_model(
            mod_model, mod_optims, crit1, crit2, loss_type, w_rec_g,
            self.mod_graphs['attribute'], self.mod_graphs['edge'], self.mod_graphs['edgeW'], self.mod_graphs['edgeC'],
            batch_train_meta_numbers, self.mod_batch_split,
            T, n_epochs, self.device
        )

        self.mod_model = mod_model
        self.loss_cl  = loss_cl
        self.loss_rec = loss_rec

    def infer_emb(self, modBatch_dict, emb_key='emb', final_latent_key='merged_emb', cat=False): 
        """
        Infer latent embeddings for each cell and return merged AnnData list.

        Parameters
        ----------
        modBatch_dict : dict
            Original input dictionary of modalities and batches.
        emb_key : str
            Key to store intermediate embeddings.
        final_latent_key : str
            Key to store final merged embedding in returned AnnData.
        cat : bool
            Whether to concatenate (True) or average (False) modalities.

        Returns
        -------
        list of AnnData
            Reconstructed AnnData objects with merged embeddings.
    """
        for k in self.mod_list:
            self.mod_model[k].eval()
            z, _ = self.mod_model[k](self.mod_graphs['attribute'][k], self.mod_graphs['edge'][k], self.mod_graphs['edgeW'][k])
            z_split = torch.split(z, self.mod_batch_split[k])
            for bi in range(self.n_batches):
                if modBatch_dict[k][bi] is not None:
                    modBatch_dict[k][bi].obsm[emb_key] = z_split[bi].detach().cpu().numpy()

        # merge embs from measured modality
        ad_finals = []
        for bi in range(self.n_batches):
            embs = []
            for m in self.mod_list:
                if modBatch_dict[m][bi] is not None:
                    embs.append(modBatch_dict[m][bi].obsm[emb_key])
                    ad_tmp = modBatch_dict[m][bi]
            if cat:
                emb = np.hstack(embs)
            else:
                emb = np.mean(embs, axis=0)
            ad = sc.AnnData(np.zeros((emb.shape[0], 2)), obs=ad_tmp.obs.copy(), obsm={'spatial':ad_tmp.obsm['spatial']})
            ad.obsm[final_latent_key] = emb
            ad_finals.append(ad)
        return ad_finals

    def impute(self, modBatch_dict, emb_key='emb', layer_key='counts', imp_knn=10):
        """
        Impute missing modalities using aligned embedding space and KNN.

        Parameters
        ----------
        modBatch_dict : dict
            Input dictionary of modalities and batches.
        emb_key : str
            Key where latent embeddings are stored.
        layer_key : str
            Which layer to impute (e.g., 'counts').
        imp_knn : int
            Number of neighbors to use in KNN-based imputation.

        Returns
        -------
        dict
        Imputed data dictionary: {modality -> list of arrays (or None)}.
    """
        aligned_pool = {
            k: np.vstack([ad.obsm[emb_key] for ad in modBatch_dict[k] if ad is not None])
            for k in modBatch_dict.keys()
        }

        target_pool = {
            k: np.vstack([ad.layers[layer_key].A if sps.issparse(ad.layers[layer_key]) else ad.layers[layer_key]
                        for ad in modBatch_dict[k] if ad is not None])
            for k in modBatch_dict.keys()
        }
        
        imputed_batchDict = {
            k: [None]*self.n_batches
            for k in modBatch_dict.keys()
        }
        for bi in range(self.n_batches):
            bi_measued_mod_names = [_ for _ in modBatch_dict.keys() if modBatch_dict[_][bi] is not None]
            bi_missing_mod_names = list(set(modBatch_dict.keys()) - set(bi_measued_mod_names))
            for k_q in bi_missing_mod_names:
                print(f'impute {k_q}-{layer_key} for batch-{bi+1}')
                # visit all measured mods to impute missing k 
                imps = []
                for k_v in bi_measued_mod_names:
                    knn_ind = utls.nn_approx(modBatch_dict[k_v][bi].obsm[emb_key], aligned_pool[k_q], knn=imp_knn)
                    p_q = target_pool[k_q][knn_ind.ravel()].reshape(*(knn_ind.shape), target_pool[k_q].shape[1])
                    imps.append(np.mean(p_q, axis=1))
                imp = np.mean(imps, axis=0)
                imputed_batchDict[k_q][bi] = imp

        return imputed_batchDict

