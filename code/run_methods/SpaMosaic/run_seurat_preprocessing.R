# ------------------------------------------------------------------------------
# Function: run_seurat_cca_integration
#
# Description:
#   This function implements a Seurat v5 preprocessing and CCA-based integration
#   workflow. It constructs a Seurat object from an expression matrix and cell
#   metadata, splits the RNA assay by sample (or another metadata field), runs
#   normalization, variable feature selection, scaling, PCA, and integrates the
#   layers using Canonical Correlation Analysis (CCA). After integration, it
#   rejoins the RNA layers and returns both the processed Seurat object and the
#   batch-corrected low-dimensional representations ("integrated.cca" embeddings).
#
#   ⚠️ IMPORTANT: This code relies on Seurat v5 functions
#   (SplitLayers, JoinLayers, IntegrateLayers). It will not run under Seurat v4.
#
# Input:
#   - expr_mat : numeric matrix or data.frame
#                If transpose = TRUE (default), the matrix is assumed to be
#                cells × genes and will be transposed inside the function.
#   - expr_meta: data.frame containing metadata for cells.
#                Row names must correspond to cell names in expr_mat.
#   - project  : character string, project name for the Seurat object.
#   - split_by : character string, metadata column used to split RNA assay
#                into layers (e.g., "Sample").
#   - transpose: logical, whether to transpose expr_mat (default = TRUE).
#
# Output:
#   A list with two elements:
#     - $seurat        : Seurat object after integration
#     - $cca_embeddings: matrix of integrated.cca embeddings
#
# Example:
#   res <- run_seurat_cca_integration(expr_mat, expr_meta)
#   obj <- res$seurat
#   cca <- res$cca_embeddings
# ------------------------------------------------------------------------------

# Seurat CCA integration, returning object and integrated.cca embeddings
run_seurat_cca_integration <- function(
  expr_mat,
  expr_meta,
  project   = "RNA",
  split_by  = "Sample",
  transpose = TRUE
){
  x <- if (transpose) t(expr_mat) else expr_mat

  # Build object and layer by sample
  obj <- CreateSeuratObject(counts = x, meta.data = expr_meta, project = project)
  obj[["RNA"]] <- split(obj[["RNA"]], f = obj[[split_by]])

  # Standard preprocessing
  obj <- NormalizeData(obj)
  obj <- FindVariableFeatures(obj)
  obj <- ScaleData(obj)
  obj <- RunPCA(obj)

  # Cross-layer integration with CCA
  obj <- IntegrateLayers(
    object = obj,
    method = CCAIntegration,
    orig.reduction = "pca",
    new.reduction  = "integrated.cca",
    verbose = FALSE
  )

  # Re-join RNA layers
  obj[["RNA"]] <- JoinLayers(obj[["RNA"]])

  # Return object and integrated embeddings
  cca_embed <- Embeddings(obj, reduction = "integrated.cca")
  return(list(seurat = obj, cca_embeddings = cca_embed))
}

### Applied to adult mouse brain and embryonic mouse brain (misar+stereo) datasets.