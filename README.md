# SpaMosaic: mosaic integration of spatial multi-omics
## Overview
With the advent of spatial multi-omics, we can mosaic integrate such datasets with partially overlapping modalities to construct higher dimensional views of the source tissue. SpaMosaic is a spatial multi-omics mosaic integration tool that employs contrastive learning and graph neural networks to construct a modality-agnostic and batch-corrected latent space suited for analyses like spatial domain identification and imputing missing omes. 

## Installation
We tested our code on a server running Ubuntu 18.04.5 LTS, equipped with 4 NVIDIA A6000 GPUs. The installation process typically takes 10–15 minutes.
```
git clone https://github.com/JinmiaoChenLab/SpaMosaic.git
cd SpaMosaic
conda create -n SpaMosaic python=3.8.8
conda activate SpaMosaic
pip install -r requirements.txt

# install torch
pip install torch==2.1.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
# install torch_geometrics
pip install torch_geometric==2.4.0 pyg_lib==0.3.1+pt21cu121 torch_scatter==2.1.2+pt21cu121 torch_sparse==0.6.18+pt21cu121 torch_cluster==1.6.3+pt21cu121 torch_spline_conv==1.2.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.1+cu121.html

python setup.py install
```
R package `mclust` is needed to perform clustering and make sure it installed in a R environment.  

## Tutorial
<!-- * [`horizontal integration`](./integration_examples/horizontal) 
* [`vertical integration`](./integration_examples/vertical) 
* [`mosaic integration`](./integration_examples/mosaic) 
* [`imputation `](./imputation_examples/)  -->

We provided detailed tutorials on applying SpaMosaic to various integration or imputation tasks. Please refer to [https://spamosaic.readthedocs.io/en/latest/](https://spamosaic.readthedocs.io/en/latest/).

## Data
Source of public datasets:
1. Mouse embryonic brain dataset: [`three slices`](http://www.biosino.org/node/project/detail/OEP003285) 
2. Mouse postnatal brain dataset (rna+atac): {[`slice 1, 2`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)}, {[`slice 3`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171943)}
3. Mouse postnatal brain dataset (rna+h3k4me3): {[`slice 1, 2`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)}, {[`slice 3`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165217)}
4. Mouse postnatal brain dataset (rna+h3k27me3): {[`slice 1,2`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)}, {[`slice 3`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165217)}
5. Mouse postnatal brain dataset (rna+h3k27ac): [`three slices`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)
6. Mouse embryo: {[`slice 1`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)}, {[`slice 2,3,4`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171943)}
7. Five-modal mouse brain dataset (rna+atac+histone): [`four slices`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)

We have compiled the simulation, in-house, and public datasets into h5ad files. Please refer to [zenodo](https://zenodo.org/uploads/12654113). 

## Reproduce results presented in manuscript
To reproduce SpaMosaic's results, please visit [`reproduce`](./reproduce/) folder.

To reproduce compared methods' results, including [`CLUE`](https://github.com/openproblems-bio/neurips2021_multimodal_topmethods/tree/main/src/match_modality/methods/clue), [`Cobolt`](https://github.com/epurdom/cobolt), [`scMoMaT`](https://github.com/PeterZZQ/scMoMaT), [`StabMap`](https://github.com/MarioniLab/StabMap), [`MIDAS`](https://sc-midas-docs.readthedocs.io/en/latest/mosaic.html), [`TotalVI`](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/totalVI.html), [`MultiVI`](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/MultiVI_tutorial.html), [`Babel`](https://github.com/OmicsML/dance/tree/main/examples/multi_modality/predict_modality/babel.py), please visit [`https://github.com/XiHuYan/Spamosaic-notebooks`](https://github.com/XiHuYan/Spamosaic-notebooks).


