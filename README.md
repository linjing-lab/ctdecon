# ctdecon

Reference-based cell type deconvolution in spatial transcriptomics.

Experiments were executed with PyTorch 2.1.0+cu121 on NVIDIA A40 with 46068MiB memory in linux environment.

## adata_utils

adata_utils file contains 4 functions, preprocess function selects highly variable genes from raw adata, then normalize, log1p, and scale adata or adata_sc(scRNA-seq). The spot_graph function constructs spot-to-spot interactive graph with is_sparse and n_neighbors, is_sparse controls the way of creating interactions matrix, then interaction of graph neighborhood and symmetrical adjacent save in adata.obsm. The contrast function generates contrastive label for spots and save in adata.obsm. The get_feature function augmentes features of adata by permutation after selecting highly variable genes with bool deconvolution and choose whether instance of adata is csc_matrix or csr_matrix.

## deconvo

deconvo file contains 1 class named config, which have train, train_sc, and train_map functions to learn representation of adata and adata_sc. Parameter device sets the device of training process, while learning_rate set for train function and learng_rate_sc set for train_sc function. Parameter dim_output is the output representation of adata, while alpha and beta act on loss functions combination of train function. The train_sc function evaluates loss with mse_loss, while lambda1 and lambda2 control the influence of reconstruction loss and contrastive loss in mapping matrix learning. Class config uses default True in deconvolution and False in is_sparse to control whether uses sparse data.

## reference

reference file contains 2 functions, overlap_gene function computes the overlap genes of adata and adata_sc, while cell2spot projects cell types onto spatial transcriptomics data using mapped matrix in adata.obsm. The overlap_gene function selects and saves overlap data by genes with spatial data and scRNA-seq reference data. The cell2spot function extracts top-k values for each spot with float retain_percent, and using map_matrix.dot(matrix_cell_type) as projection by spot-level. Final mapped results are saved in adata.obs by columns of projection dataframes.
