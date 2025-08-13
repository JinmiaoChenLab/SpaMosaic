library(StabMap)
library(magrittr)
library(scater)
library(scran)
library(SingleCellMultiModal)
library(gridExtra)
library(glue)
library(Matrix)

set.seed(2025)

data_dir = '../../../data/processed/Tonsil-S1-S2-S2/'
out_dir = '../../../results/embeddings/StabMap/Tonsil-S1-S2-S2/'

pj <- file.path

read_mat <- function(dir2){
    mat = readMM(pj(dir2, 'matrix.mtx'))
    feat = read.table(pj(dir2, 'features.csv'), sep=',', header=T, comment.char = "")
    barc = read.table(pj(dir2, 'barcodes.csv'), sep=',', header=T, comment.char = "")
    rownames(mat) = barc$X0
    colnames(mat) = feat$X0

    return (t(mat))
}

mult_rna_count = read_mat(pj(data_dir, 'R/S1', 'rna'))
mult_other_count = read_mat(pj(data_dir, 'R/S1', 'adt'))
test_rna_count = read_mat(pj(data_dir, 'R/S2-rna', 'rna'))
test_other_count = read_mat(pj(data_dir, 'R/S2-adt', 'adt'))

dim(mult_rna_count)
dim(mult_other_count)

mult_rna = SingleCellExperiment(assays = list(counts = mult_rna_count))
test_rna = SingleCellExperiment(assays = list(counts = test_rna_count))
mult_other = SingleCellExperiment(assays = list(counts = mult_other_count))
test_other = SingleCellExperiment(assays = list(counts = test_other_count))

mult_rna = logNormCounts(mult_rna)
mult_other =  logNormCounts(mult_other)
test_rna = logNormCounts(test_rna)
test_other =  logNormCounts(test_other)

# Feature selection
decomp_rna <- modelGeneVar(mult_rna)
hvgs <- rownames(decomp_rna)[decomp_rna$mean > 0.01 & decomp_rna$p.value <= 0.05]
mult_rna <- mult_rna[hvgs, ]

decomp_rna <- modelGeneVar(test_rna)
hvgs <- rownames(decomp_rna)[decomp_rna$mean > 0.01 & decomp_rna$p.value <= 0.05]
test_rna <- test_rna[hvgs, ]

# decomp_atac <- modelGeneVar(mult_other)
# hvgs <- rownames(decomp_atac)[decomp_atac$mean > 0.25 & decomp_atac$p.value <= 0.05]
# mult_other <- mult_other[hvgs, ]

# decomp_atac <- modelGeneVar(test_other)
# hvgs <- rownames(decomp_atac)[decomp_atac$mean > 0.25 & decomp_atac$p.value <= 0.05]
# test_other <- test_other[hvgs, ]

dim(mult_rna)
dim(test_rna)
dim(mult_other)
dim(test_other)

mult_rna_norm = as.matrix(logcounts(mult_rna))
mult_other_norm = as.matrix(logcounts(mult_other))
test_rna_norm = as.matrix(logcounts(test_rna))
test_other_norm = as.matrix(logcounts(test_other))

rownames(mult_rna_norm) = paste0('rna-', rownames(mult_rna_norm))
rownames(mult_other_norm) = paste0('adt-', rownames(mult_other_norm))
rownames(test_rna_norm) = paste0('rna-', rownames(test_rna_norm))
rownames(test_other_norm) = paste0('adt-', rownames(test_other_norm))

assay_list <- list(
    RNA = test_rna_norm,
    OTHER = test_other_norm,
    Multiome = rbind(mult_rna_norm, mult_other_norm)
)

# mdt <- mosaicDataTopology(assay_list)
stab <- stabMap(
    assay_list,
    reference_list = c("Multiome"),
    plot = FALSE
)

write.csv(as.data.frame(stab), 
    file=paste0(out_dir, 'df_emb.csv'),
    quote=F, row.names=T)