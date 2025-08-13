import os, gc
# for cuda >= 10.2
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8' based on your preference

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scanpy as sc
import h5py
import math
import sklearn
from tqdm import tqdm
import scipy.sparse as sps
import scipy.io as sio
import seaborn as sns
import warnings

from os.path import join
import torch
from collections import Counter
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from annoy import AnnoyIndex
from sklearn.preprocessing import normalize
from harmony import harmonize

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def harmony(df_list, use_gpu=True):
    """
    Harmony batch effect correction applied jointly to multiple dataframes
    """
    all = pd.concat(df_list)
    all_batches = all.pop("batch")
    all_batches.columns = ["batch"]
    all_batches = all_batches.to_frame()
    all_harmony = harmonize(
        all.to_numpy(), all_batches, batch_key="batch", use_gpu=use_gpu, verbose=True
    )
    out_df_list = []
    curr_idx = 0
    for df in df_list:
        out_df_list.append(all_harmony[curr_idx : curr_idx + len(df)])
        curr_idx += len(df)
    return out_df_list

def HARMONY(all_df, batch_labels, use_gpu=True):
    all_df['batch'] = batch_labels
    all_arr = harmony([all_df])[0]
    return all_arr