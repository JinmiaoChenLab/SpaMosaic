import os, gc
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
import pkg_resources

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

# cur_dir = os.path.dirname(os.path.abspath(__file__))

class SpaMosaic(object):
    def __init__(
        self, 
        modBatch_dict={},  # dict={'rna':[batch1, None, ...], 'adt':[batch1, batch2, ...], ...}
        input_key='dimred_bc',
        batch_key='batch', 
        radius_cutoff=2000, intra_knn=10, inter_knn=10, w_g=0.8, 
        log_dir=None,
        seed=1234, num_workers=6,
        device='cuda:0'
    ):  
        self.device = torch.device(device) if ('cuda' in device) and torch.cuda.is_available() else torch.device('cpu')
        self.log_dir = log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        set_seeds(seed)
        self.radius_cutoff = radius_cutoff
        self.w_g = w_g
        self.intra_knn = intra_knn
        self.inter_knn = inter_knn
        self.seed = seed
        self.num_workers = num_workers

        self.input_key = input_key
        self.mod_list = np.array(list(modBatch_dict.keys()))
        self.n_mods = len(self.mod_list)
        self.n_batches = len(modBatch_dict[self.mod_list[0]])
        self.batch_key = batch_key
        self.barc2batch = utls.get_barc2batch(modBatch_dict)

        # check if there is empty batch
        self.batch_contained_mod_ids = utls.check_batch_empty(modBatch_dict)

        # check if this dataset can be integrated
        self.check_integrity()

        # prepare spot-spot for each modality
        self.prepare_graphs(modBatch_dict)

    def check_integrity(self):
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

    def prepare_graphs(self, modBatch_dict):
        # determine which batches as bridges, which not. 
        mod_mask = np.zeros((self.n_mods, self.n_batches))
        for bi in range(self.n_batches):
            for ki,k in enumerate(self.mod_list):
                if modBatch_dict[k][bi] is not None:
                    mod_mask[ki][bi] = 1

        self.bridge_batch_num_ids = np.where(mod_mask.sum(axis=0) >= 2)[0]
        self.non_bridge_batch_num_ids = np.where(mod_mask.sum(axis=0) < 2)[0]

        ### prepare meta parameter for training
        self.mod_batch_split = {
            key: [modBatch_dict[key][bi].n_obs if modBatch_dict[key][bi] is not None else 0 for bi in range(self.n_batches)]
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

        # spatial-neighbor graph
        for key in self.mod_list:
            ads = [ad for ad in modBatch_dict[key] if ad is not None]
            build_graph.build_intra_graph(ads, rad_cutoff=self.radius_cutoff, knn=self.intra_knn)
        
        # expression-neighbor graph
        mod_mnn_set = {}
        for key in self.mod_list:
            mod_mnn_set[key] = build_graph.build_mnn_graph(bridge_ads[key], test_ads[key], self.input_key, self.inter_knn, self.seed)
            print('Number of mnn pairs for {}:{}'.format(key, len(mod_mnn_set[key])))

        ### prepare inputs
        mod_graphs, mod_input_ads, mod_graph_idx2barc = {}, {}, {}
        for k in self.mod_list:
            mod_input_ads[k], mod_graph_idx2barc[k] = build_graph.mergeGraph([ad for ad in modBatch_dict[k] if ad is not None], 
                                                                            mod_mnn_set[k], inter_weight=self.w_g, use_rep=self.input_key)
        
        mod_graphs_edge_arr = {k:mod_input_ads[k].uns['edgeList'] for k in self.mod_list}
        mod_graphs['edge'] = {
            k:torch.LongTensor(np.array([mod_graphs_edge_arr[k][0], mod_graphs_edge_arr[k][1]])).to(self.device) 
            for k in self.mod_list
        }
        mod_graphs['edgeW'] = {k:torch.FloatTensor(mod_input_ads[k].uns['edgeW']).to(self.device) for k in self.mod_list}  # edge weights
        mod_graphs['edgeC'] = {}
        for k in self.mod_list:
            barc1 = utls.dict_map(mod_graph_idx2barc[k], mod_graphs_edge_arr[k][0])
            barc2 = utls.dict_map(mod_graph_idx2barc[k], mod_graphs_edge_arr[k][1])
            b1, b2 = np.array(utls.dict_map(self.barc2batch, barc1)), np.array(utls.dict_map(self.barc2batch, barc2))
            mod_graphs['edgeC'][k] = torch.LongTensor(np.where(b1 == b2, b1, -1)).to(self.device)

        mod_graphs['attribute'] = {
            k:torch.FloatTensor(mod_input_ads[k].obsm[self.input_key]).to(self.device)
            for k in self.mod_list
        }

        mod_graphs['attribute_dim'] = {k:mod_graphs['attribute'][k].shape[1] for k in self.mod_list}

        self.mod_graphs = mod_graphs

    def prepare_net(self, net):
        # config = utls.load_config(f'{cur_dir}/configs/{net}.yaml')
        config_path = pkg_resources.resource_filename('spamosaic', f'configs/{net}.yaml')
        config = utls.load_config(config_path)

        mod_model = {}
        for k in self.mod_list:
            encoder = archs.wlgcn.WLGCN(self.mod_graphs['attribute_dim'][k], config.model.out_dim, K=config.model.n_layer, dec_l=config.model.n_dec_l, 
                                        hidden_size=config.model.hid_dim, dropout=config.model.dropout, slope=config.model.slope)
            mod_model[k] = encoder.to(self.device)
            
        return mod_model

    def train(self, net, lr, use_mini_thr=8000, mini_batch_size=1024, loss_type='adapted', T=0.01, bias=0, n_epochs=100, w_rec_g=0.):
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
                emb = np.hstack(emb)
            else:
                emb = np.mean(embs, axis=0)
            ad = sc.AnnData(np.zeros((emb.shape[0], 2)), obs=ad_tmp.obs.copy(), obsm={'spatial':ad_tmp.obsm['spatial']})
            ad.obsm[final_latent_key] = emb
            ad_finals.append(ad)
        return ad_finals

    def impute(self, modBatch_dict, emb_key='emb', layer_key='counts', imp_knn=10):
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

