# SpaMosaic: Mosaic Integration of Spatial Multi-Omics

> **Development build**  
> It now relies on **sparse adjacency matrices (`torch_sparse.SparseTensor`)** for message passing, instead of PyTorch Geometric’s edge-list representation, which makes training on large graphs much more memory-efficient.  

## Installation

### 1. Create and activate a conda environment
```bash
git clone -b dev --single-branch https://github.com/JinmiaoChenLab/SpaMosaic.git
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
pip install -e .
```

## Tutorials
We provide detailed tutorials on applying SpaMosaic to various integration and imputation tasks.  
Please refer to the documentation: [https://spamosaic.readthedocs.io/en/dev/](https://spamosaic.readthedocs.io/en/dev/)

## Reproducibility
To reproduce the results of SpaMosaic and the compared methods, please use the code on the [`SpaMosaic-reproduce`](https://github.com/JinmiaoChenLab/SpaMosaic/tree/SpaMosaic-reproduce) branch.
