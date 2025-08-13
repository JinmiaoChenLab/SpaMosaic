library(StabMap)
library(magrittr)
library(scater)
library(scran)
library(SingleCellMultiModal)
library(gridExtra)
library(glue)
library(Matrix)

set.seed(2025)

data_dir = '../../../data/processed/DBiT-RnaAtac-Atac/R/'
out_dir = '../../../results/embeddings/StabMap/DBiT-RnaAtac-Atac/'

pj <- file.path

read_mat <- function(dir2){
    mat = readMM(pj(dir2, 'matrix.mtx'))
    feat = read.table(pj(dir2, 'features.csv'), sep=',', header=T, comment.char = "")
    barc = read.table(pj(dir2, 'barcodes.csv'), sep=',', header=T, comment.char = "")
    rownames(mat) = barc[, ncol(barc)]
    colnames(mat) = feat[, ncol(feat)]

    return (t(mat))
}

db_rna_x = read_mat(paste0(data_dir, 'db/RNA'))
db_atac_x = read_mat(paste0(data_dir, 'db/ATAC'))

sra50_rna_x = read_mat(paste0(data_dir, 'sra50/RNA'))
sra50_atac_x = read_mat(paste0(data_dir, 'sra50/ATAC'))

sra100_rna_x = read_mat(paste0(data_dir, 'sra100/RNA'))
sra100_atac_x = read_mat(paste0(data_dir, 'sra100/ATAC'))

sat_x = read_mat(paste0(data_dir, 'mono/ATAC'))

head(rownames(db_rna_x))
head(rownames(db_atac_x))

dim(db_rna_x)
dim(db_atac_x)

s1_rna = SingleCellExperiment(assays = list(counts = db_rna_x))
s2_rna = SingleCellExperiment(assays = list(counts = sra50_rna_x))
s3_rna = SingleCellExperiment(assays = list(counts = sra100_rna_x))
s1_atac = SingleCellExperiment(assays = list(counts = db_atac_x))
s2_atac = SingleCellExperiment(assays = list(counts = sra50_atac_x))
s3_atac = SingleCellExperiment(assays = list(counts = sra100_atac_x))
s4_atac = SingleCellExperiment(assays = list(counts = sat_x))

s1_rna = logNormCounts(s1_rna)
s2_rna = logNormCounts(s2_rna)
s3_rna = logNormCounts(s3_rna)
s1_atac = logNormCounts(s1_atac)
s2_atac =  logNormCounts(s2_atac)
s3_atac =  logNormCounts(s3_atac)
s4_atac =  logNormCounts(s4_atac)

# Feature selection
decomp_rna <- modelGeneVar(s1_rna)
hvgs <- rownames(decomp_rna)[decomp_rna$mean > 0.01 & decomp_rna$p.value <= 0.05]
s1_rna <- s1_rna[hvgs, ]

decomp_rna <- modelGeneVar(s2_rna)
hvgs <- rownames(decomp_rna)[decomp_rna$mean > 0.01 & decomp_rna$p.value <= 0.05]
s2_rna <- s2_rna[hvgs, ]

decomp_rna <- modelGeneVar(s3_rna)
hvgs <- rownames(decomp_rna)[decomp_rna$mean > 0.01 & decomp_rna$p.value <= 0.05]
s3_rna <- s3_rna[hvgs, ]

decomp_atac <- modelGeneVar(s1_atac)
hvgs <- rownames(decomp_atac)[decomp_atac$mean > 0.05 & decomp_atac$p.value <= 0.1]
s1_atac <- s1_atac[hvgs, ]

decomp_atac <- modelGeneVar(s2_atac)
hvgs <- rownames(decomp_atac)[decomp_atac$mean > 0.05 & decomp_atac$p.value <= 0.1]
s2_atac <- s2_atac[hvgs, ]

decomp_atac <- modelGeneVar(s3_atac)
hvgs <- rownames(decomp_atac)[decomp_atac$mean > 0.05 & decomp_atac$p.value <= 0.1]
s3_atac <- s3_atac[hvgs, ]

decomp_atac <- modelGeneVar(s4_atac)
hvgs <- rownames(decomp_atac)[decomp_atac$mean > 0.05 & decomp_atac$p.value <= 0.1]
s4_atac <- s4_atac[hvgs, ]

dim(s1_rna)
dim(s2_rna)
dim(s3_rna)
dim(s1_atac)
dim(s2_atac)
dim(s3_atac)
dim(s4_atac)

head(rownames(as.matrix(logcounts(s1_rna))))
head(rownames(as.matrix(logcounts(s1_atac))))

s1_atac_norm = as.matrix(logcounts(s1_atac))
s2_atac_norm = as.matrix(logcounts(s2_atac))
s3_atac_norm = as.matrix(logcounts(s3_atac))
s4_atac_norm = as.matrix(logcounts(s4_atac))

assay_list <- list(
    section1 = rbind(as.matrix(logcounts(s1_rna)), s1_atac_norm),
    section2 = rbind(as.matrix(logcounts(s2_rna)), s2_atac_norm),
    section3 = rbind(as.matrix(logcounts(s3_rna)), s3_atac_norm),
    section4 = s4_atac_norm
)

# mdt <- mosaicDataTopology(assay_list)
stab <- stabMap(
    assay_list,
    reference_list = c("section1"),
    plot = FALSE
)

write.csv(as.data.frame(stab), 
    file=paste0(out_dir, 'df_emb.csv'),
    quote=F, row.names=T)