Installation
============

.. note::
   SpaMosaic supports running on both GPU and CPU environments.
   You can choose either depending on your hardware, although a GPU
   will generally provide faster training and inference.

Clone the GitHub repository and navigate into the project directory:

.. code-block:: bash

   git clone https://github.com/JinmiaoChenLab/SpaMosaic.git
   cd SpaMosaic

Create a new conda environment and activate it:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate spamosaic-env

-------------------------
Install core dependencies
-------------------------

**CPU-only example**

.. code-block:: bash

   # PyTorch 2.0.0 (CPU build)
   pip install torch==2.0.0+cpu --index-url https://download.pytorch.org/whl/cpu

   # PyTorch Geometric (CPU wheels; must match Torch 2.0.0)
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric \
       -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

   pip install harmony-pytorch --no-deps

**GPU examples**

.. code-block:: bash

   # Example: PyTorch 2.0.0 with CUDA 11.7
   pip install torch==2.0.0+cu117 --index-url https://download.pytorch.org/whl/cu117

   # PyTorch Geometric (match Torch 2.0.0 and CUDA version)
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric \
       -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

   pip install harmony-pytorch --no-deps

.. note::
   - The GPU commands above are **examples** using CUDA 11.7.
   - You should install the PyTorch build that matches your local CUDA toolkit or driver.
   - See the official PyTorch installation guide for other versions:
     https://pytorch.org/get-started/locally/
   - Matching PyTorch Geometric wheels can be found here:
     https://data.pyg.org/whl/

------------
Install SpaMosaic
------------

.. code-block:: bash

   pip install spamosaic
