library(StabMap)
library(magrittr)
library(scater)
library(scran)
library(SingleCellMultiModal)
library(gridExtra)
library(glue)
library(Matrix)

set.seed(2025)

data_dir = '../../../data/processed/Simulations/'
out_dir = '../../../results/embeddings/StabMap/Simulations/'

pj <- file.path

read_mat <- function(dir, prefix){
    mat = readMM(pj(dir, glue('{prefix}_mat.mtx')))
    meta = read.table(pj(dir, glue('{prefix}_meta.csv')), sep=',', header=T, row.names=1)
    feat_name = read.table(pj(dir, glue('{prefix}_feats.csv')), sep=',', header=T)
    rownames(mat) = rownames(meta)
    colnames(mat) = feat_name$X0

    return (t(mat))
}

for (p in c(0.1, 0.2, 0.3, 0.4, 0.5)){
  for (s in c(1, 2, 3)){
    print('===================================')
    print(p)
    print(s)
    print('===================================')
    tmp_dir = list.files(path=data_dir, pattern=glue("^rna-{p}"), full.names=T, recursive=F, include.dirs=T)
    mult_rna_count = read_mat(pj(tmp_dir, s, 'R'), 'bridge_rna')
    mult_adt_count = read_mat(pj(tmp_dir, s, 'R'), 'bridge_atac')
    single_rna_count = read_mat(pj(tmp_dir, s, 'R'), 'test_rna')
    single_adt_count = read_mat(pj(tmp_dir, s, 'R'), 'test_atac')

    mult_rna = SingleCellExperiment(assays = list(counts = mult_rna_count))
    test_rna = SingleCellExperiment(assays = list(counts = single_rna_count))
    mult_adt = SingleCellExperiment(assays = list(counts = mult_adt_count))
    test_adt = SingleCellExperiment(assays = list(counts = single_adt_count))

    mult_rna = logNormCounts(mult_rna)
    mult_adt =  logNormCounts(mult_adt)
    test_rna = logNormCounts(test_rna)
    test_adt =  logNormCounts(test_adt)

    mult_rna_norm = as.matrix(logcounts(mult_rna))
    mult_adt_norm = as.matrix(logcounts(mult_adt))
    test_rna_norm = as.matrix(logcounts(test_rna))
    test_adt_norm = as.matrix(logcounts(test_adt))

    rownames(mult_rna_norm) = paste0('rna-', rownames(mult_rna_norm))
    rownames(mult_adt_norm) = paste0('adt-', rownames(mult_adt_norm))
    rownames(test_rna_norm) = paste0('rna-', rownames(test_rna_norm))
    rownames(test_adt_norm) = paste0('adt-', rownames(test_adt_norm))

    assay_list <- list(
        RNA = test_rna_norm,
        ADT = test_adt_norm,
        Multiome = rbind(mult_rna_norm, mult_adt_norm)
    )

    # mdt <- mosaicDataTopology(assay_list)
    stab <- stabMap(
        assay_list,
        reference_list = c("Multiome"),
        plot = FALSE
    )

    foldn = basename(tmp_dir) 
    out_path = pj(out_dir, foldn, as.character(s))
    dir.create(out_path, recursive = TRUE, showWarnings = FALSE)

    write.csv(as.data.frame(stab),
      file = pj(out_path, 'df_emb.csv'),
      quote = FALSE, row.names = TRUE)
  }
}
