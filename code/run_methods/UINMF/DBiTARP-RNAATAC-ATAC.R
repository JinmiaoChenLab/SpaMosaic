library(GenomicRanges)
library(rliger)
library(Seurat)
library(Signac)
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
data_dir = '../../../data/processed/DBiT-RnaAtac-Atac/R/'
out_dir = '../../../results/embeddings/UINMF/DBiT-RnaAtac-Atac/'

db_rna_x = load_data(paste0(data_dir, 'db/RNA'))
db_atac_x = load_data(paste0(data_dir, 'db/ATAC'))

sra50_rna_x = load_data(paste0(data_dir, 'sra50/RNA'))
sra50_atac_x = load_data(paste0(data_dir, 'sra50/ATAC'))

sra100_rna_x = load_data(paste0(data_dir, 'sra100/RNA'))
sra100_atac_x = load_data(paste0(data_dir, 'sra100/ATAC'))

sat_x = load_data(paste0(data_dir, 'mono/ATAC'))

head(rownames(db_rna_x))
head(rownames(db_atac_x))
head(colnames(db_rna_x))
head(colnames(db_atac_x))

# RNA processing
liger <- createLiger(list(db=as.matrix(t(db_rna_x)), sra50=as.matrix(t(sra50_rna_x)), sra100=as.matrix(t(sra100_rna_x))))
liger <- normalize(liger)
liger <- selectGenes(liger, var.thresh = 0.1, datasets.use =1 , unshared = FALSE,  unshared.datasets = list(2, 3), unshared.thresh= 0.2)
liger <- scaleNotCenter(liger)

db_rna_scale = liger@scale.data$db
sra50_rna_scale = liger@scale.data$sra50
sra100_rna_scale = liger@scale.data$sra100
# unshared_feats = liger@scale.data$geneExpression
rm(liger)

dim(db_rna_scale) # 0.2 => n_features=3906
dim(sra50_rna_scale)
dim(sra100_rna_scale)
# 0.1 => 
# [1] 9787 8004
# [1] 2497 8004
# [1] 9215 8004
# length(intersect(rownames(db_rna_scale), rownames(sra50_rna_scale)))
# all(rownames(db_rna_scale)==rownames(db_atac_x))

# ATAC processing
liger <- createLiger(list(db=as.matrix(t(db_atac_x)), sra50=as.matrix(t(sra50_atac_x)), sra100=as.matrix(t(sra100_atac_x)), sat=as.matrix(t(sat_x))))
liger <- normalize(liger)

# liger <- selectGenes(liger, var.thresh = 0.00001, datasets.use =1, unshared = FALSE, unshared.datasets = list(2,3,4), unshared.thresh= 0.2)
atac_list <- list(db = t(db_atac_x), sra50 = t(sra50_atac_x), sra100 = t(sra100_atac_x), sat = t(sat_x))
seurat_list <- lapply(atac_list, function(mat) {
  gr <- StringToGRanges(rownames(mat), sep = c(":", "-"))
  chrom_assay <- CreateChromatinAssay(counts = mat, ranges = gr)
  seu <- CreateSeuratObject(counts = chrom_assay, assay = "peaks")
  
  # Normalize and select top features
  seu <- RunTFIDF(seu)
  seu <- FindTopFeatures(seu, min.cutoff = "q75")
  return(seu)
})

peak_sets <- lapply(seurat_list, function(x) VariableFeatures(x))
shared_peaks <- Reduce(intersect, peak_sets)

liger@var.genes <- shared_peaks

liger <- scaleNotCenter(liger)
dim(liger@scale.data$db)  # thr=0.00001 => 9787 6341

#Add the unshared features that have been properly selected, such that they are added as a genes by cells matrix. 
liger@var.unshared.features[[1]] = colnames(db_rna_scale)
liger@scale.unshared.data[[1]] = t(db_rna_scale)

liger@var.unshared.features[[2]] = colnames(sra50_rna_scale)
liger@scale.unshared.data[[2]] = t(sra50_rna_scale)

liger@var.unshared.features[[3]] = colnames(sra100_rna_scale)
liger@scale.unshared.data[[3]] = t(sra100_rna_scale)
# t(rbind(db_rna_scale, sra50_rna_scale, sra100_rna_scale))
# rm(unshared_feats)

dim(liger)

# #### Step 4: Joint Matrix Factorization

#To factorize the datasets and include the unshared datasets, set the use.unshared parameter to TRUE. 
liger <- optimizeALS(liger, k=30, use.unshared = TRUE, max_iters=30, thresh=1e-10)
liger <- quantile_norm(liger)

x = liger@H.norm
dim(x)

write.csv(x, file = file.path(out_dir, "df_emb.csv"), quote = F, row.names = T)