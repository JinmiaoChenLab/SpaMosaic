# SpaMosaic: Mosaic Integration of Spatial Multi-Omics

## Overview
With the advent of spatial multi-omics technologies, it is now possible to integrate datasets with partially overlapping modalities to construct higher-dimensional views of the source tissue. **SpaMosaic** is a spatial multi-omics integration tool that uses contrastive learning and graph neural networks to build a **modality-agnostic**, **batch-corrected** latent space, enabling downstream analyses such as spatial domain identification and imputation of missing omics.

## Installation

### 1. Create and activate a conda environment
```bash
git clone https://github.com/JinmiaoChenLab/SpaMosaic.git
cd SpaMosaic
conda env create -f environment.yml
conda activate spamosaic-env
```

### 2. Install core dependencies
Choose the PyTorch and PyTorch Geometric build that matches your local CUDA toolkit or driver.  
For other versions (including CPU-only builds), please refer to the [documentation](https://spamosaic.readthedocs.io/en/latest/).

**Example: PyTorch 2.0.0 with CUDA 11.7**
```bash
# PyTorch
pip install torch==2.0.0+cu117 --index-url https://download.pytorch.org/whl/cu117

# PyTorch Geometric (matching Torch 2.0.0 and CUDA version)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
  -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

pip install harmony-pytorch --no-deps

# Install SpaMosaic
pip install spamosaic
```

## Tutorials
We provide detailed tutorials on applying SpaMosaic to various integration and imputation tasks.  
Please refer to the documentation: [https://spamosaic.readthedocs.io/en/latest/](https://spamosaic.readthedocs.io/en/latest/)

<!--
### Example tutorial categories
* [Horizontal integration](./integration_examples/horizontal)  
* [Vertical integration](./integration_examples/vertical)  
* [Mosaic integration](./integration_examples/mosaic)  
* [Imputation](./imputation_examples/)  
-->


## Reproducibility
To reproduce the results of SpaMosaic and the compared methods, please use the code on the [`SpaMosaic-reproduce`](https://github.com/JinmiaoChenLab/SpaMosaic/tree/SpaMosaic-reproduce) branch.
