
library(GenomicRanges)
library(rliger)
library(Seurat)
library(stringr)
library(Matrix)

# ## Scenario 2: Intergration of RNA+ATAC and RNA data
load_data <- function(path) {
  # 读取细胞和基因名称
  cell_names <- read.csv(file.path(path, "barcodes.csv"), header = T)
  gene_names <- read.csv(file.path(path, "features.csv"), header = T)
  expr_matrix <- readMM(file.path(path, "matrix.mtx"))

  rownames(expr_matrix) <- cell_names[[2]]
  colnames(expr_matrix) <- gene_names[[2]]
  return(expr_matrix)
}

# #### Step1: read data
data_dir = '../../../data/processed/Mux-VisiumHD-HE/R/'
out_dir = '../../../results/embeddings/UINMF/Mux-VisiumHD-HE/'

mux_rna_x = load_data(paste0(data_dir, 'Mux/RNA'))
mux_atac_x = load_data(paste0(data_dir, 'Mux/ATAC'))

hd_rna_x = load_data(paste0(data_dir, 'HD/RNA'))
he_rna_x = load_data(paste0(data_dir, 'HE/RNA'))

# ATAC processing
# se = CreateSeuratObject(t(mux_atac_x))
# vars_2000 <- FindVariableFeatures(se, selection.method = "vst", nfeatures = 2000)
# top2000 <- head(VariableFeatures(vars_2000),2000)
# DE_peak = VariableFeatures(vars_2000)

liger <- createLiger(list(peaks = as.matrix(t(mux_atac_x))))
liger <- normalize(liger)
liger <- selectGenes(liger, var.thresh = 0.00001, datasets.use =1 , unshared = FALSE)
# length(liger@var.genes)
# liger@var.genes <- DE_peak
liger <- scaleNotCenter(liger)
mux_atac_scale = liger@scale.data$peaks
rm(liger)

dim(mux_atac_scale)  # 0.00001 -> ndim=1446

# RNA
liger <- createLiger(list(db=as.matrix(t(mux_rna_x)), sra50=as.matrix(t(hd_rna_x)), sra100=as.matrix(t(he_rna_x))))
liger <- normalize(liger)
liger <- selectGenes(liger, var.thresh = 0.1, datasets.use =1 , unshared = FALSE,  unshared.datasets = list(2, 3), unshared.thresh= 0.2)
liger <- scaleNotCenter(liger)

dim(liger@scale.data$db)  # ndim=1285

liger@var.unshared.features[[1]] = colnames(mux_atac_scale)
liger@scale.unshared.data[[1]] = t(mux_atac_scale)

#To factorize the datasets and include the unshared datasets, set the use.unshared parameter to TRUE. 
liger <- optimizeALS(liger, k=30, use.unshared = TRUE, max_iters=30, thresh=1e-10)
liger <- quantile_norm(liger)

x = liger@H.norm
dim(x)

write.csv(x, file = file.path(out_dir, "df_emb.csv"), quote = F, row.names = T)