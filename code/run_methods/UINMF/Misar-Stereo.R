
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
data_dir = '../../../data/processed/Misar-Stereo/R/'
out_dir = '../../../results/embeddings/UINMF/Misar-Stereo/'

mis13_rna_x = load_data(paste0(data_dir, 'Mis13/RNA'))
mis13_atac_x = load_data(paste0(data_dir, 'Mis13/ATAC'))

mis15_rna_x = load_data(paste0(data_dir, 'Mis15/RNA'))
mis15_atac_x = load_data(paste0(data_dir, 'Mis15/ATAC'))

mis18_rna_x = load_data(paste0(data_dir, 'Mis18/RNA'))
mis18_atac_x = load_data(paste0(data_dir, 'Mis18/ATAC'))

ste12_rna_x = load_data(paste0(data_dir, 'Ste12/RNA'))
ste14_rna_x = load_data(paste0(data_dir, 'Ste14/RNA'))
ste16_rna_x = load_data(paste0(data_dir, 'Ste16/RNA'))

# ATAC processing
liger <- createLiger(list(b1=as.matrix(t(mis13_atac_x)), b2=as.matrix(t(mis15_atac_x)), b3=as.matrix(t(mis18_atac_x))))
liger <- normalize(liger)
liger <- selectGenes(liger, var.thresh = 0.00001, datasets.use =1 , unshared = FALSE,  unshared.datasets = list(2,3), unshared.thresh= 0.2)
liger <- scaleNotCenter(liger)

mis13_atac_scale = liger@scale.data$b1
mis15_atac_scale = liger@scale.data$b2
mis18_atac_scale = liger@scale.data$b3
rm(liger)

dim(mis13_atac_scale)  # ndim=68106

# RNA
liger <- createLiger(list(mis13=as.matrix(t(mis13_rna_x)), mis15=as.matrix(t(mis15_rna_x)), mis18=as.matrix(t(mis18_rna_x)), ste12=as.matrix(t(ste12_rna_x)), ste14=as.matrix(t(ste14_rna_x)), ste16=as.matrix(t(ste16_rna_x))))
liger <- normalize(liger)
liger <- selectGenes(liger, var.thresh = 0.0001, datasets.use =1 , unshared = FALSE,  unshared.datasets = list(2, 3, 4, 5, 6), unshared.thresh= 0.2)
liger <- scaleNotCenter(liger)

dim(liger@scale.data$mis13)  # 0.0001 => 1130， 0.1=>761

liger@var.unshared.features[[1]] = colnames(mis13_atac_scale)
liger@scale.unshared.data[[1]] = t(mis13_atac_scale)

liger@var.unshared.features[[2]] = colnames(mis15_atac_scale)
liger@scale.unshared.data[[2]] = t(mis15_atac_scale)

liger@var.unshared.features[[3]] = colnames(mis18_atac_scale)
liger@scale.unshared.data[[3]] = t(mis18_atac_scale)

#To factorize the datasets and include the unshared datasets, set the use.unshared parameter to TRUE. 
liger <- optimizeALS(liger, k=30, use.unshared = TRUE, max_iters=30, thresh=1e-10)
liger <- quantile_norm(liger)

x = liger@H.norm
dim(x)
max(x)

write.csv(x, file = file.path(out_dir, "df_emb.csv"), quote = F, row.names = T)