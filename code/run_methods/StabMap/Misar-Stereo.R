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

data_dir = '../../../data/processed/Misar-Stereo/R/'
out_dir = '../../../results/embeddings/StabMap/Misar-Stereo/'

mis13_rna_x = read_mat(paste0(data_dir, 'Mis13/RNA'))
mis13_atac_x = read_mat(paste0(data_dir, 'Mis13/ATAC'))

mis15_rna_x = read_mat(paste0(data_dir, 'Mis15/RNA'))
mis15_atac_x = read_mat(paste0(data_dir, 'Mis15/ATAC'))

mis18_rna_x = read_mat(paste0(data_dir, 'Mis18/RNA'))
mis18_atac_x = read_mat(paste0(data_dir, 'Mis18/ATAC'))

ste12_rna_x = read_mat(paste0(data_dir, 'Ste12/RNA'))
ste14_rna_x = read_mat(paste0(data_dir, 'Ste14/RNA'))
ste16_rna_x = read_mat(paste0(data_dir, 'Ste16/RNA'))

dim(mis13_rna_x)
dim(mis13_atac_x)
head(rownames(mis13_rna_x))
head(rownames(mis13_atac_x))

# 创建 SingleCellExperiment 对象
mis13_rna = SingleCellExperiment(assays = list(counts = mis13_rna_x))
mis13_atac = SingleCellExperiment(assays = list(counts = mis13_atac_x))
mis15_rna = SingleCellExperiment(assays = list(counts = mis15_rna_x))
mis15_atac = SingleCellExperiment(assays = list(counts = mis15_atac_x))
mis18_rna = SingleCellExperiment(assays = list(counts = mis18_rna_x))
mis18_atac = SingleCellExperiment(assays = list(counts = mis18_atac_x))
ste12_rna = SingleCellExperiment(assays = list(counts = ste12_rna_x))
ste14_rna = SingleCellExperiment(assays = list(counts = ste14_rna_x))
ste16_rna = SingleCellExperiment(assays = list(counts = ste16_rna_x))

# logNormCounts
mis13_rna = logNormCounts(mis13_rna)
mis13_atac = logNormCounts(mis13_atac)
mis15_rna = logNormCounts(mis15_rna)
mis15_atac = logNormCounts(mis15_atac)
mis18_rna = logNormCounts(mis18_rna)
mis18_atac = logNormCounts(mis18_atac)
ste12_rna = logNormCounts(ste12_rna)
ste14_rna = logNormCounts(ste14_rna)
ste16_rna = logNormCounts(ste16_rna)

# 高变基因筛选
decomp_mis13_rna = modelGeneVar(mis13_rna)
hvgs = rownames(decomp_mis13_rna)[decomp_mis13_rna$mean > 0.01 & decomp_mis13_rna$p.value <= 0.05]
mis13_rna = mis13_rna[hvgs, ]
decomp_mis13_atac = modelGeneVar(mis13_atac)
hvgs = rownames(decomp_mis13_atac)[decomp_mis13_atac$mean > 0.05 & decomp_mis13_atac$p.value <= 0.05]
mis13_atac = mis13_atac[hvgs, ]

decomp_mis15_rna = modelGeneVar(mis15_rna)
hvgs = rownames(decomp_mis15_rna)[decomp_mis15_rna$mean > 0.01 & decomp_mis15_rna$p.value <= 0.05]
mis15_rna = mis15_rna[hvgs, ]
decomp_mis15_atac = modelGeneVar(mis15_atac)
hvgs = rownames(decomp_mis15_atac)[decomp_mis15_atac$mean > 0.05 & decomp_mis15_atac$p.value <= 0.05]
mis15_atac = mis15_atac[hvgs, ]

decomp_mis18_rna = modelGeneVar(mis18_rna)
hvgs = rownames(decomp_mis18_rna)[decomp_mis18_rna$mean > 0.01 & decomp_mis18_rna$p.value <= 0.05]
mis18_rna = mis18_rna[hvgs, ]
decomp_mis18_atac = modelGeneVar(mis18_atac)
hvgs = rownames(decomp_mis18_atac)[decomp_mis18_atac$mean > 0.05 & decomp_mis18_atac$p.value <= 0.05]
mis18_atac = mis18_atac[hvgs, ]

decomp_ste12_rna = modelGeneVar(ste12_rna)
hvgs = rownames(decomp_ste12_rna)[decomp_ste12_rna$mean > 0.01 & decomp_ste12_rna$p.value <= 0.05]
ste12_rna = ste12_rna[hvgs, ]
decomp_ste14_rna = modelGeneVar(ste14_rna)
hvgs = rownames(decomp_ste14_rna)[decomp_ste14_rna$mean > 0.01 & decomp_ste14_rna$p.value <= 0.05]
ste14_rna = ste14_rna[hvgs, ]
decomp_ste16_rna = modelGeneVar(ste16_rna)
hvgs = rownames(decomp_ste16_rna)[decomp_ste16_rna$mean > 0.01 & decomp_ste16_rna$p.value <= 0.05]
ste16_rna = ste16_rna[hvgs, ]

# 查看各自筛选后的维度
dim_list <- list(
  mis13_rna = dim(mis13_rna),
  mis13_atac = dim(mis13_atac),
  mis15_rna = dim(mis15_rna),
  mis15_atac = dim(mis15_atac),
  mis18_rna = dim(mis18_rna),
  mis18_atac = dim(mis18_atac),
  ste12_rna = dim(ste12_rna),
  ste14_rna = dim(ste14_rna),
  ste16_rna = dim(ste16_rna)
)
print(dim_list)


# 提取 logcounts 并加前缀
mis13_rna_norm = as.matrix(logcounts(mis13_rna))
mis13_atac_norm = as.matrix(logcounts(mis13_atac)); rownames(mis13_atac_norm) = paste0('atac-', rownames(mis13_atac_norm))
mis15_rna_norm = as.matrix(logcounts(mis15_rna))
mis15_atac_norm = as.matrix(logcounts(mis15_atac)); rownames(mis15_atac_norm) = paste0('atac-', rownames(mis15_atac_norm))
mis18_rna_norm = as.matrix(logcounts(mis18_rna))
mis18_atac_norm = as.matrix(logcounts(mis18_atac)); rownames(mis18_atac_norm) = paste0('atac-', rownames(mis18_atac_norm))
ste12_rna_norm = as.matrix(logcounts(ste12_rna))
ste14_rna_norm = as.matrix(logcounts(ste14_rna))
ste16_rna_norm = as.matrix(logcounts(ste16_rna))

assay_list <- list(
  mis13 = rbind(mis13_rna_norm, mis13_atac_norm),
  mis15 = rbind(mis15_rna_norm, mis15_atac_norm),
  mis18 = rbind(mis18_rna_norm, mis18_atac_norm),
  ste12 = ste12_rna_norm,
  ste14 = ste14_rna_norm,
  ste16 = ste16_rna_norm
)

stab <- stabMap(
    assay_list,
    reference_list = c("mis13"),
    plot = FALSE
)

write.csv(as.data.frame(stab), 
    file=paste0(out_dir, 'df_emb.csv'),
    quote=F, row.names=T)
