import os
import scanpy as sc
from os.path import join
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../..')

from spamosaic.framework import SpaMosaic

os.environ['R_HOME'] = '/disco_500t/xuhua/miniforge3/envs/Seurat5/lib/R'
os.environ['R_USER'] = '/disco_500t/xuhua/miniforge3/envs/Seurat5/lib/python3.8/site-packages/rpy2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # for CuBLAS operation and you have CUDA >= 10.2
import spamosaic.utils as utls
from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess, Epigenome_preprocess

data_dir = '/home/xuhua/xuhua_disco/gitrepo/BridgeNorm/SpaMosaic-release/data/integration/Human_lymph_node'

ad1_rna = sc.read_h5ad(join(data_dir, 'slice1/s1_adata_rna.h5ad'))
ad1_adt = sc.read_h5ad(join(data_dir, 'slice1/s1_adata_adt.h5ad'))
ad2_rna = sc.read_h5ad(join(data_dir, 'slice2/s2_adata_rna.h5ad'))
ad3_adt = sc.read_h5ad(join(data_dir, 'slice3/s3_adata_adt.h5ad'))

input_dict = {
    'rna': [ad1_rna, ad2_rna, None],
    'adt': [ad1_adt, None,    ad3_adt]
}

input_key = 'dimred_bc'

RNA_preprocess(input_dict['rna'], batch_corr=True, favor='scanpy', n_hvg=5000, batch_key='src', key=input_key)
ADT_preprocess(input_dict['adt'], batch_corr=True, batch_key='src', key=input_key)

model = SpaMosaic(
    modBatch_dict=input_dict, input_key=input_key,
    batch_key='src', intra_knn=2, inter_knn=2, w_g=0.8, 
    seed=1234, 
    device='cuda:0'
)

model.train(net='wlgcn', lr=0.01, T=0.01, n_epochs=100)

ad_embs = model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')
ad_mosaic = sc.concat(ad_embs)
ad_mosaic = utls.get_umap(ad_mosaic, use_reps=['merged_emb'])

utls.plot_basis(ad_mosaic, basis='merged_emb_umap', color=['src'])

utls.clustering(ad_mosaic, n_cluster=7, used_obsm='merged_emb', algo='mclust', key='mclust')
utls.split_adata_ob(ad_embs, ad_mosaic, 'obs', 'mclust')

for i, ad in enumerate(ad_embs):
    utls.plot_basis(ad, 'spatial', 'mclust', s=70)
    plt.savefig(f'./output{i}.png', dpi=300)
    plt.show()