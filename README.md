# SpaMosaic: a computational framework for spatial mosaic integration
## Overview
Mosaic integration, in which different datasets share part of the measured modalities, poses great challenges in single-cell data analysis. Spatial mosaic integration considers mosaic integration within spatial context, demanding addtional spatial awareness. Here, we develop SpaMosaic, a method for spatial mosaic integration based on contrastive learning and light graph neural network. SpaMosaic can handle various tasks: including spatial horizontal integration, spatial vertical integration, spatial rectangular integration, spatial mosaic integration, and imputation for missing modalities. SpaMosaic is good at identifying smoothing spatial domains and is also computationally efficient. 

## Installation
```
git clone https://github.com/XiHuYan/SpaMosaic.git
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
* [`horizontal integration`](./integration_examples/horizontal) 
* [`vertical integration`](./integration_examples/vertical) 
* [`mosaic integration`](./integration_examples/mosaic) 
* [`imputation `](./imputation_examples/) 

## Data
Public datasets
1. Mouse embryonic brain dataset: [`three slices`](http://www.biosino.org/node/project/detail/OEP003285) 
2. Mouse postnatal brain dataset (rna+atac): {[`slice 1, 2`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)}, {[`slice 3`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171943)}
3. Mouse postnatal brain dataset (rna+h3k4me3): {[`slice 1, 2`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)}, {[`slice 3`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165217)}
4. Mouse postnatal brain dataset (rna+h3k27me3): {[`slice 1,2`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)}, {[`slice 3`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE165217)}
5. Mouse postnatal brain dataset (rna+h3k27ac): [`three slices`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)
6. Mouse embryo: {[`slice 1`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)}, {[`slice 2,3,4`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE171943)}
7. Five-modal mouse brain dataset (rna+atac+histone): [`four slices`](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)

We have organized the simulation datasets, in-house datasets and public datasets into [`h5ad`](https://zenodo.org/uploads/12654113) files. 

## Reproduce other methods
We compared SpaMosaic with other methods, including [`CLUE`](https://github.com/openproblems-bio/neurips2021_multimodal_topmethods/tree/main/src/match_modality/methods/clue), [`Cobolt`](https://github.com/epurdom/cobolt), [`scMoMaT`](https://github.com/PeterZZQ/scMoMaT), [`StabMap`](https://github.com/MarioniLab/StabMap), [`MIDAS`](https://sc-midas-docs.readthedocs.io/en/latest/mosaic.html), [`TotalVI`](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/totalVI.html), [`MultiVI`](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/MultiVI_tutorial.html), [`Babel`](https://github.com/OmicsML/dance/tree/main/examples/multi_modality/predict_modality/babel.py). The reproducing notebooks can be found at [`notebooks`](https://github.com/XiHuYan/Spamosaic-notebooks)


