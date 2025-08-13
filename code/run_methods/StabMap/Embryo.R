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

# #### Step1: read data
data_dir = '../../../data/processed/Embryo/R/'
out_dir = '../../../results/embeddings/StabMap/Embryo/'

sra_rna_x = read_mat(paste0(data_dir, 'SRA/RNA'))
sra_atac_x = read_mat(paste0(data_dir, 'SRA/ATAC'))

mux2_27ac_x = read_mat(paste0(data_dir, 'Mux2/H3K27ac'))
mux2_27me_x = read_mat(paste0(data_dir, 'Mux2/H3K27me3'))

mux4_rna_x = read_mat(paste0(data_dir, 'Mux4/RNA'))
mux4_atac_x = read_mat(paste0(data_dir, 'Mux4/ATAC'))
mux4_4me3_x = read_mat(paste0(data_dir, 'Mux4/H3K4me3'))
mux4_27me_x = read_mat(paste0(data_dir, 'Mux4/H3K27me3'))

sat_x = read_mat(paste0(data_dir, 'SAT/ATAC'))
cut_27ac_x = read_mat(paste0(data_dir, 'CUT/H3K27ac'))
cut_27me_x = read_mat(paste0(data_dir, 'CUT/H3K27me3'))
cut_4me3_x = read_mat(paste0(data_dir, 'CUT/H3K4me3'))

head(rownames(sra_rna_x))
head(rownames(sra_atac_x))
head(rownames(cut_27ac_x))

sra_rna = SingleCellExperiment(assays = list(counts = sra_rna_x))
sra_atac = SingleCellExperiment(assays = list(counts = sra_atac_x))

mux2_27ac = SingleCellExperiment(assays = list(counts = mux2_27ac_x))
mux2_27me = SingleCellExperiment(assays = list(counts = mux2_27me_x))

mux4_rna = SingleCellExperiment(assays = list(counts = mux4_rna_x))
mux4_atac = SingleCellExperiment(assays = list(counts = mux4_atac_x))
mux4_4me3 = SingleCellExperiment(assays = list(counts = mux4_4me3_x))
mux4_27me = SingleCellExperiment(assays = list(counts = mux4_27me_x))

sat = SingleCellExperiment(assays = list(counts = sat_x))
cut_27ac = SingleCellExperiment(assays = list(counts = cut_27ac_x))
cut_27me = SingleCellExperiment(assays = list(counts = cut_27me_x))
cut_4me3 = SingleCellExperiment(assays = list(counts = cut_4me3_x))

sra_rna = logNormCounts(sra_rna)
sra_atac = logNormCounts(sra_atac)

mux2_27ac = logNormCounts(mux2_27ac)
mux2_27me = logNormCounts(mux2_27me)

mux4_rna = logNormCounts(mux4_rna)
mux4_atac = logNormCounts(mux4_atac)
mux4_4me3 = logNormCounts(mux4_4me3)
mux4_27me = logNormCounts(mux4_27me)

sat = logNormCounts(sat)
cut_27ac = logNormCounts(cut_27ac)
cut_27me = logNormCounts(cut_27me)
cut_4me3 = logNormCounts(cut_4me3)

decomp_rna <- modelGeneVar(sra_rna)
hvgs <- rownames(decomp_rna)[decomp_rna$mean > 0.01 & decomp_rna$p.value <= 0.05]
sra_rna <- sra_rna[hvgs, ]

decomp_mux4_rna <- modelGeneVar(mux4_rna)
hvgs <- rownames(decomp_mux4_rna)[decomp_mux4_rna$mean > 0.01 & decomp_mux4_rna$p.value <= 0.05]
mux4_rna <- mux4_rna[hvgs, ]

decomp_atac <- modelGeneVar(sra_atac)
hvgs <- rownames(decomp_atac)[decomp_atac$mean > 0.05 & decomp_atac$p.value <= 0.05]
sra_atac <- sra_atac[hvgs, ]

decomp_mux2_27ac <- modelGeneVar(mux2_27ac)
hvgs <- rownames(decomp_mux2_27ac)[decomp_mux2_27ac$mean > 0.05 & decomp_mux2_27ac$p.value <= 0.05]
mux2_27ac <- mux2_27ac[hvgs, ]

decomp_mux2_27me <- modelGeneVar(mux2_27me)
hvgs <- rownames(decomp_mux2_27me)[decomp_mux2_27me$mean > 0.05 & decomp_mux2_27me$p.value <= 0.05]
mux2_27me <- mux2_27me[hvgs, ]

decomp_mux4_atac <- modelGeneVar(mux4_atac)
hvgs <- rownames(decomp_mux4_atac)[decomp_mux4_atac$mean > 0.05 & decomp_mux4_atac$p.value <= 0.05]
mux4_atac <- mux4_atac[hvgs, ]

decomp_mux4_4me3 <- modelGeneVar(mux4_4me3)
hvgs <- rownames(decomp_mux4_4me3)[decomp_mux4_4me3$mean > 0.05 & decomp_mux4_4me3$p.value <= 0.05]
mux4_4me3 <- mux4_4me3[hvgs, ]

decomp_mux4_27me <- modelGeneVar(mux4_27me)
hvgs <- rownames(decomp_mux4_27me)[decomp_mux4_27me$mean > 0.05 & decomp_mux4_27me$p.value <= 0.05]
mux4_27me <- mux4_27me[hvgs, ]

decomp_sat <- modelGeneVar(sat)
hvgs <- rownames(decomp_sat)[decomp_sat$mean > 0.05 & decomp_sat$p.value <= 0.05]
sat <- sat[hvgs, ]

decomp_cut_27ac <- modelGeneVar(cut_27ac)
hvgs <- rownames(decomp_cut_27ac)[decomp_cut_27ac$mean > 0.05 & decomp_cut_27ac$p.value <= 0.05]
cut_27ac <- cut_27ac[hvgs, ]

decomp_cut_27me <- modelGeneVar(cut_27me)
hvgs <- rownames(decomp_cut_27me)[decomp_cut_27me$mean > 0.05 & decomp_cut_27me$p.value <= 0.05]
cut_27me <- cut_27me[hvgs, ]

decomp_cut_4me3 <- modelGeneVar(cut_4me3)
hvgs <- rownames(decomp_cut_4me3)[decomp_cut_4me3$mean > 0.05 & decomp_cut_4me3$p.value <= 0.05]
cut_4me3 <- cut_4me3[hvgs, ]


dim(sra_rna)
dim(sra_atac)
dim(mux2_27ac)
dim(mux2_27me)
dim(mux4_rna)
dim(mux4_atac)
dim(mux4_4me3)
dim(mux4_27me)
dim(sat)
dim(cut_27ac)
dim(cut_27me)
dim(cut_4me3)


head(rownames(as.matrix(logcounts(sra_rna))))
head(rownames(as.matrix(logcounts(sra_atac))))

sra_rna_norm = as.matrix(logcounts(sra_rna))
mux4_rna_norm = as.matrix(logcounts(mux4_rna))
sra_atac_norm = as.matrix(logcounts(sra_atac)); rownames(sra_atac_norm) = paste0('atac-', rownames(sra_atac_norm))
mux4_atac_norm = as.matrix(logcounts(mux4_atac)); rownames(mux4_atac_norm) = paste0('atac-', rownames(mux4_atac_norm))
sat_norm = as.matrix(logcounts(sat)); rownames(sat_norm) = paste0('atac-', rownames(sat_norm))
mux2_27ac_norm = as.matrix(logcounts(mux2_27ac)); rownames(mux2_27ac_norm) = paste0('h3k27ac-', rownames(mux2_27ac_norm))
cut_27ac_norm = as.matrix(logcounts(cut_27ac)); rownames(cut_27ac_norm) = paste0('h3k27ac-', rownames(cut_27ac_norm))
mux2_27me_norm = as.matrix(logcounts(mux2_27me)); rownames(mux2_27me_norm) = paste0('h3k27me3-', rownames(mux2_27me_norm))
mux4_27me_norm = as.matrix(logcounts(mux4_27me)); rownames(mux4_27me_norm) = paste0('h3k27me3-', rownames(mux4_27me_norm))
cut_27me_norm = as.matrix(logcounts(cut_27me)); rownames(cut_27me_norm) = paste0('h3k27me3-', rownames(cut_27me_norm))
mux4_4me3_norm = as.matrix(logcounts(mux4_4me3)); rownames(mux4_4me3_norm) = paste0('h3k4me3-', rownames(mux4_4me3_norm))
cut_4me3_norm = as.matrix(logcounts(cut_4me3)); rownames(cut_4me3_norm) = paste0('h3k4me3-', rownames(cut_4me3_norm))

assay_list <- list(
    sra = rbind(sra_rna_norm, sra_atac_norm),
    mux2 = rbind(mux2_27ac_norm, mux2_27me_norm),
    mux4 = rbind(mux4_rna_norm, mux4_atac_norm, mux4_4me3_norm, mux4_27me_norm),
    sat = sat_norm,
    cut_27ac = cut_27ac_norm,
    cut_27me = cut_27me_norm, 
    cut_4me3 = cut_4me3_norm
)

stab <- stabMap(
    assay_list,
    reference_list = c("sra"),
    plot = FALSE
)

write.csv(as.data.frame(stab), 
    file=paste0(out_dir, 'df_emb.csv'),
    quote=F, row.names=T)
