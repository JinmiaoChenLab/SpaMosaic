.. SpaMosaic documentation master file, created by
   sphinx-quickstart on Fri Oct  4 10:20:07 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SpaMosaic documentation
===============================

``SpaMosaic`` aims to integrate mosaic datasets with partially overlapping modalities to construct higher dimensional views of tissues. ``SpaMosaic`` can handle multiple serial (or non-serial) sections from the same biological tissue or sections that form a time series such as embryo development. ``SpaMosaic`` works well on various spatial-omics technologies (spatial-atac-seq, spatial-cut&tag-seq, spatial-rna-epigenomics-seq, misar-seq, visium, stereo-seq and etc.). 

.. figure:: _static/Fig_1.jpg
   :width: 80%
   :align: center

   Overview of SpaMosaic workflow

Features and Capabilities
-------------------------

From the perspective of computational task, ``SpaMosaic`` can handle:

1. **Integration**:

   - **Horizontal Integration**: Integrate multiple sections sharing the same modality.

   - **Vertical Integration**: Integrate multiple modalities from a single section.

   - **Mosaic Integration**: Integrate multiple sections with different multi-modal compositions.

2. **Imputation**:

   - For mosaic datasets, **SpaMosaic** imputes missing assays for each section, facilitating a more comprehensive analysis.


Getting started with SpaMosaic
------------------------------

To begin using **SpaMosaic**, please refer to the following sections of the documentation:

- The `Installation <./install.rst>`_ provides instructions for setting up **SpaMosaic** in your environment.

- The `Tutorials <./tutorials/index.rst>`_ contains examples on how to use **SpaMosaic** for various integration and imputation tasks.

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   install
   tutorials/index
   dataset
   credits