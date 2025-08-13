library(GenomicRanges)
library(rliger)
library(Seurat)
library(stringr)
library(Matrix)

data_dir = '../../../data/processed/MB-5M/'
out_dir = '../../../results/embeddings/UINMF/MB-5M/'

pj <- file.path

read_mat <- function(dir2){
    mat = readMM(pj(dir2, 'matrix.mtx'))
    feat = read.table(pj(dir2, 'features.csv'), sep=',', header=T, comment.char = "")
    barc = read.table(pj(dir2, 'barcodes.csv'), sep=',', header=T, comment.char = "")
    rownames(mat) = barc$X0
    colnames(mat) = feat$X0

    return (t(mat))
}

s1_rna_count = read_mat(pj(data_dir, 'R/S1', 'rna'))
s1_h3k27me3_count = read_mat(pj(data_dir, 'R/S1', 'h3k27me3'))
s2_rna_count = read_mat(pj(data_dir, 'R/S2', 'rna'))
s2_h3k4me3_count = read_mat(pj(data_dir, 'R/S2', 'h3k4me3'))
s3_rna_count = read_mat(pj(data_dir, 'R/S3', 'rna'))
s3_h3k27ac_count = read_mat(pj(data_dir, 'R/S3', 'h3k27ac'))
s4_rna_count = read_mat(pj(data_dir, 'R/S4', 'rna'))
s4_atac_count = read_mat(pj(data_dir, 'R/S4', 'atac'))

# H3K27me3 processing
liger <- createLiger(list(data=s1_h3k27me3_count))
liger <- normalize(liger)
liger <- selectGenes(liger, var.thresh = 0.00001, datasets.use =1 , unshared = FALSE)
liger <- scaleNotCenter(liger)
s1_h3k27me3_scale = liger@scale.data$data
dim(s1_h3k27me3_scale) # 9752 6657
rm(liger)

# H3K27me3 processing
liger <- createLiger(list(data=s2_h3k4me3_count))
liger <- normalize(liger)
liger <- selectGenes(liger, var.thresh = 0.00001, datasets.use =1 , unshared = FALSE)
liger <- scaleNotCenter(liger)
s2_h3k4me3_scale = liger@scale.data$data
dim(s2_h3k4me3_scale)  # 9548 2250
rm(liger)

# H3K27ac processing
liger <- createLiger(list(data=s3_h3k27ac_count))
liger <- normalize(liger)
liger <- selectGenes(liger, var.thresh = 0.00001, datasets.use =1 , unshared = FALSE)
liger <- scaleNotCenter(liger)
s3_h3k27ac_scale = liger@scale.data$data
dim(s3_h3k27ac_scale)  # 9370 8991
rm(liger)

# ATAC processing
liger <- createLiger(list(data=s4_atac_count))
liger <- normalize(liger)
liger <- selectGenes(liger, var.thresh = 0.00001, datasets.use =1 , unshared = FALSE)
liger <- scaleNotCenter(liger)
s4_atac_scale = liger@scale.data$data
dim(s4_atac_scale)  # 9215 4912
rm(liger)

# RNA processing
liger <- createLiger(list(s1=as.matrix(s1_rna_count), s2=as.matrix(s2_rna_count), s3=as.matrix(s3_rna_count), s4=as.matrix(s4_rna_count)))
liger <- normalize(liger)
liger <- selectGenes(liger, var.thresh = 0.1, datasets.use =1 , unshared = FALSE,  unshared.datasets = list(2, 3), unshared.thresh= 0.2)
liger <- scaleNotCenter(liger)
dim(liger@scale.data$s1)  # 9752 8987

#Add the unshared features that have been properly selected, such that they are added as a genes by cells matrix. 
liger@var.unshared.features[[1]] = paste0('H3K27me3-', colnames(s1_h3k27me3_scale))
liger@scale.unshared.data[[1]] = t(s1_h3k27me3_scale)

liger@var.unshared.features[[2]] = paste0('H3K4me3-', colnames(s2_h3k4me3_scale))
liger@scale.unshared.data[[2]] = t(s2_h3k4me3_scale)

liger@var.unshared.features[[3]] = paste0('H3K27ac-', colnames(s3_h3k27ac_scale))
liger@scale.unshared.data[[3]] = t(s3_h3k27ac_scale)

liger@var.unshared.features[[4]] = paste0('ATAC-', colnames(s4_atac_scale))
liger@scale.unshared.data[[4]] = t(s4_atac_scale)


# #### Step 4: Joint Matrix Factorization

#To factorize the datasets and include the unshared datasets, set the use.unshared parameter to TRUE. 
liger <- optimizeALS(liger, k=30, use.unshared = TRUE, max_iters=30, thresh=1e-10)
liger <- quantile_norm(liger)

x = liger@H.norm
dim(x)

write.csv(x, file = file.path(out_dir, "df_emb.csv"), quote = F, row.names = T)