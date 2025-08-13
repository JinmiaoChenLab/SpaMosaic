library(StabMap)
library(magrittr)
library(scater)
library(scran)
library(SingleCellMultiModal)
library(gridExtra)
library(glue)
library(Matrix)

set.seed(2025)

data_dir = '../../../data/processed/MB-5M/'
out_dir = '../../../results/embeddings/StabMap/MB-5M/'

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

dim(s1_rna_count)
dim(s1_h3k27me3_count)
dim(s2_rna_count)
dim(s2_h3k4me3_count)
dim(s3_rna_count)
dim(s3_h3k27ac_count)
dim(s4_rna_count)
dim(s4_atac_count)

s1_rna = SingleCellExperiment(assays = list(counts = s1_rna_count))
s2_rna = SingleCellExperiment(assays = list(counts = s2_rna_count))
s3_rna = SingleCellExperiment(assays = list(counts = s3_rna_count))
s4_rna = SingleCellExperiment(assays = list(counts = s4_rna_count))
s1_h3k27me3 = SingleCellExperiment(assays = list(counts = s1_h3k27me3_count))
s2_h3k4me3 = SingleCellExperiment(assays = list(counts = s2_h3k4me3_count))
s3_h3k27ac = SingleCellExperiment(assays = list(counts = s3_h3k27ac_count))
s4_atac = SingleCellExperiment(assays = list(counts = s4_atac_count))

s1_rna = logNormCounts(s1_rna)
s2_rna = logNormCounts(s2_rna)
s3_rna = logNormCounts(s3_rna)
s4_rna = logNormCounts(s4_rna)
s1_h3k27me3 = logNormCounts(s1_h3k27me3)
s2_h3k4me3 =  logNormCounts(s2_h3k4me3)
s3_h3k27ac =  logNormCounts(s3_h3k27ac)
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

decomp_rna <- modelGeneVar(s4_rna)
hvgs <- rownames(decomp_rna)[decomp_rna$mean > 0.01 & decomp_rna$p.value <= 0.05]
s4_rna <- s4_rna[hvgs, ]

decomp_atac <- modelGeneVar(s1_h3k27me3)
hvgs <- rownames(decomp_atac)[decomp_atac$mean > 0.05 & decomp_atac$p.value <= 0.1]
s1_h3k27me3 <- s1_h3k27me3[hvgs, ]

decomp_atac <- modelGeneVar(s2_h3k4me3)
hvgs <- rownames(decomp_atac)[decomp_atac$mean > 0.05 & decomp_atac$p.value <= 0.1]
s2_h3k4me3 <- s2_h3k4me3[hvgs, ]

decomp_atac <- modelGeneVar(s3_h3k27ac)
hvgs <- rownames(decomp_atac)[decomp_atac$mean > 0.05 & decomp_atac$p.value <= 0.1]
s3_h3k27ac <- s3_h3k27ac[hvgs, ]

decomp_atac <- modelGeneVar(s4_atac)
hvgs <- rownames(decomp_atac)[decomp_atac$mean > 0.05 & decomp_atac$p.value <= 0.1]
s4_atac <- s4_atac[hvgs, ]

dim(s1_rna)
dim(s2_rna)
dim(s3_rna)
dim(s4_rna)
dim(s1_h3k27me3)
dim(s2_h3k4me3)
dim(s3_h3k27ac)
dim(s4_atac)

head(rownames(as.matrix(logcounts(s1_rna))))
head(rownames(as.matrix(logcounts(s1_h3k27me3))))

s1_h3k27me3_norm = as.matrix(logcounts(s1_h3k27me3))
s2_h3k4me3_norm = as.matrix(logcounts(s2_h3k4me3))
s3_h3k27ac_norm = as.matrix(logcounts(s3_h3k27ac))
s4_atac_norm = as.matrix(logcounts(s4_atac))

rownames(s1_h3k27me3_norm) = paste0('h3k27me3-', rownames(s1_h3k27me3_norm))
rownames(s2_h3k4me3_norm) = paste0('h3k4me3-', rownames(s2_h3k4me3_norm))
rownames(s3_h3k27ac_norm) = paste0('h3k27ac-', rownames(s3_h3k27ac_norm))
rownames(s4_atac_norm) = paste0('atac-', rownames(s4_atac_norm))

assay_list <- list(
    section1 = rbind(as.matrix(logcounts(s1_rna)), s1_h3k27me3_norm),
    section2 = rbind(as.matrix(logcounts(s2_rna)), s2_h3k4me3_norm),
    section3 = rbind(as.matrix(logcounts(s3_rna)), s3_h3k27ac_norm),
    section4 = rbind(as.matrix(logcounts(s4_rna)), s4_atac_norm)
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