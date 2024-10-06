Installation
============

.. note::
   Currently, SpaMosaic only supports GPU-enabled version. For our experiments, we used an NVIDIA A6000 GPU with CUDA 12.1.

Create a new environment and activate it:

.. code-block:: bash

   conda create -n SpaMosaic python=3.8.8
   conda activate SpaMosaic

Clone the repository and install the requirements:

.. code-block:: bash

   git clone https://github.com/JinmiaoChenLab/SpaMosaic.git
   cd SpaMosaic
   pip install -r requirements.txt

Install torch and torch geometric:

.. code-block:: bash

   pip install torch==2.1.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
   # install torch_geometrics
   pip install torch_geometric==2.4.0 pyg_lib==0.3.1+pt21cu121 torch_scatter==2.1.2+pt21cu121 torch_sparse==0.6.18+pt21cu121 torch_cluster==1.6.3+pt21cu121 torch_spline_conv==1.2.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.1+cu121.html

Set up:

.. code-block:: bash

   python setup.py install

.. note::
   SpaMosaic works better with the mclust algorithm. Make sure it is installed in your R environment.
