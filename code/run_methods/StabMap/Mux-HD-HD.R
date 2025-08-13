library(StabMap)
library(magrittr)
library(scater)
library(scran)
library(SingleCellMultiModal)
library(gridExtra)
library(glue)
library(Matrix)

set.seed(2025)

pj <- file.path

read_mat <- function(dir2){
    mat = readMM(pj(dir2, 'matrix.mtx'))
    feat = read.table(pj(dir2, 'features.csv'), sep=',', header=T, comment.char = "")
    barc = read.table(pj(dir2, 'barcodes.csv'), sep=',', header=T, comment.char = "")
    rownames(mat) = barc[, ncol(barc)]
    colnames(mat) = feat[, ncol(feat)]

    return (t(mat))
}

data_dir = '../../../data/processed/Mux-VisiumHD-HE/R/'
out_dir = '../../../results/embeddings/StabMap/Mux-VisiumHD-HE/'

mux_rna_x = read_mat(paste0(data_dir, 'Mux/RNA'))
mux_atac_x = read_mat(paste0(data_dir, 'Mux/ATAC'))

hd_rna_x = read_mat(paste0(data_dir, 'HD/RNA'))
he_rna_x = read_mat(paste0(data_dir, 'HE/RNA'))

head(rownames(mux_rna_x))
head(rownames(mux_atac_x))

# 创建 SingleCellExperiment 对象
mux_rna = SingleCellExperiment(assays = list(counts = mux_rna_x))
mux_atac = SingleCellExperiment(assays = list(counts = mux_atac_x))
hd_rna = SingleCellExperiment(assays = list(counts = hd_rna_x))
he_rna = SingleCellExperiment(assays = list(counts = he_rna_x))

# logNormCounts
mux_rna = logNormCounts(mux_rna)
mux_atac = logNormCounts(mux_atac)
hd_rna = logNormCounts(hd_rna)
he_rna = logNormCounts(he_rna)

# 高变基因筛选
decomp_mux_rna = modelGeneVar(mux_rna)
hvgs = rownames(decomp_mux_rna)[decomp_mux_rna$mean > 0.01 & decomp_mux_rna$p.value <= 0.05]
mux_rna = mux_rna[hvgs, ]

decomp_mux_atac = modelGeneVar(mux_atac)
hvgs = rownames(decomp_mux_atac)[decomp_mux_atac$mean > 0.025 & decomp_mux_atac$p.value <= 0.1]
# length(hvgs)
mux_atac = mux_atac[hvgs, ]

decomp_hd_rna = modelGeneVar(hd_rna)
hvgs = rownames(decomp_hd_rna)[decomp_hd_rna$mean > 0.01 & decomp_hd_rna$p.value <= 0.05]
hd_rna = hd_rna[hvgs, ]

decomp_he_rna = modelGeneVar(he_rna)
hvgs = rownames(decomp_he_rna)[decomp_he_rna$mean > 0.01 & decomp_he_rna$p.value <= 0.05]
he_rna = he_rna[hvgs, ]

dim_list <- list(
  mux_rna = dim(mux_rna),
  mux_atac = dim(mux_atac),
  hd_rna = dim(hd_rna),
  he_rna = dim(he_rna)
)
print(dim_list)

# 提取 logcounts 并加前缀
mux_rna_norm = as.matrix(logcounts(mux_rna))
mux_atac_norm = as.matrix(logcounts(mux_atac)); rownames(mux_atac_norm) = paste0('atac-', rownames(mux_atac_norm))
hd_rna_norm = as.matrix(logcounts(hd_rna))
he_rna_norm = as.matrix(logcounts(he_rna))

assay_list <- list(
  mux = rbind(mux_rna_norm, mux_atac_norm),
  hd = hd_rna_norm,
  he = he_rna_norm
)

stab <- stabMap(
    assay_list,
    reference_list = c("mux"),
    plot = FALSE
)

write.csv(as.data.frame(stab), 
    file=paste0(out_dir, 'df_emb.csv'),
    quote=F, row.names=T)




