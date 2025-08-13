# SpaMosaic-reproduce

Code and data to reproduce the **mosaic integration** benchmarking results in our manuscript.

---

## Contents

- `code/`
  - `run_methods/`: method-specific notebooks and scripts (Cobolt, BANKSY, etc.).  
    Each subfolder contains end-to-end examples per dataset/task (e.g., `Simulations`, `Lymph-S1-S2-S3`, `Mux-VisiumHD-HE`, etc.).
  - `compute_metrics/`: ARI, iLISI, PAS, CHAOS, etc. (`metrics.py` and ready-to-run notebooks).
  - `plot_examples/`: quick UMAP and spatial cluster plotting examples.
- `data/`
  - `processed/`: curated AnnData/H5AD objects for each dataset.
  - `raw/`: raw outputs or original structure snapshots for selected datasets (Visium-like outputs, images, etc.).
- `results/`
  - `clusters/`: cluster labels.
  - `embeddings/`: low-dimensional embeddings from all methods.
  - `imputations/`: imputed expression profiles.
  - `scores/`: metric scores.

---

## Quick start

1. **Environment**  
   Use method-specific environments where needed. We follow the official tutorials of each method when setting up environments (see [Compared Methods](#compared-methods-official-repostutorials)).

2. **Run a method**  
   Open a notebook under `code/run_methods/<Method>/` that matches your dataset (e.g., `Lymph-S1-S2-S3.ipynb`, `Simulations.ipynb`).  
   Follow the cells to (i) preprocess / (ii) train / (iii) infer / (iv) save embeddings.

3. **Evaluate**  
   Use `code/compute_metrics` to compute:  
   - **Clustering**: ARI, PAS, CHAOS  
   - **Integration**: iLISI, FOSCTTM, MS
   - **Imputation**: PCC, AUROC, CMD

4. **Visualize**  
   See `code/plot_examples/umap_example.ipynb` and `spatial_cluster_example.ipynb` for UMAPs and spatial domain maps.

---

## Resources

### Data Availability  
All data used in the experiments are publicly available on [Zenodo](https://zenodo.org/records/16813383).  

### Compared Methods (official repos/tutorials)  
> We list all **compared** baselines here (excluding SpaMosaic and Leiden).

- **BANKSY** — [GitHub](https://github.com/prabhakarlab/Banksy)  
- **CellCharter** — [GitHub](https://github.com/CSOgroup/cellcharter)  
- **CLUE** — [GitHub (GLUE)](https://github.com/gao-lab/GLUE) · [Tutorial](https://github.com/openproblems-bio/neurips2021_multimodal_topmethods/tree/main/src/match_modality/methods/clue)  
- **Cobolt** — [GitHub](https://github.com/epurdom/cobolt)  
- **MIDAS** — [GitHub](https://github.com/labomics/midas)  
- **MultiVI** — [Tutorial](https://docs.scvi-tools.org/en/stable/tutorials/notebooks/multimodal/MultiVI_tutorial.html#setup-and-training-multivi)  
- **scMoMaT** — [GitHub](https://github.com/PeterZZQ/scMoMaT)  
- **StabMap** — [Tutorial](https://www.bioconductor.org/packages/devel/bioc/vignettes/StabMap/inst/doc/stabMap_PBMC_Multiome.html)  
- **UINMF (LIGER)** — [GitHub](https://github.com/welch-lab/liger) · [Benchmarking tutorial](https://github.com/QuKunLab/MultiomeBenchmarking/tree/main/code/Integration/pipeline/Mosaic)  
- **BABEL** — [GitHub](https://github.com/wukevin/babel) · [Tutorial](https://pydance.readthedocs.io/en/latest/index.html)  
- **TotalVI** — [Tutorial](https://docs.scvi-tools.org/en/1.2.2/tutorials/notebooks/multimodal/cite_scrna_integration_w_totalVI.html)  

