import pandas
from ._utils import top_value, celltype_matrix

def overlap_gene(adata, adata_sc):
    if 'highly_variable' not in adata.var.keys():
       raise ValueError("'highly_variable' are not existed in adata!")
    else:    
       adata = adata[:, adata.var['highly_variable']]
    if 'highly_variable' not in adata_sc.var.keys():
       raise ValueError("'highly_variable' are not existed in adata_sc!")
    else:    
       adata_sc = adata_sc[:, adata_sc.var['highly_variable']]   
    genes = list(set(adata.var.index) & set(adata_sc.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))
    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes
    adata = adata[:, genes]
    adata_sc = adata_sc[:, genes]
    return adata, adata_sc

def cell2spot(adata, adata_sc, retain_percent=0.1):
    '''\
    Project cell types onto ST data using mapped matrix in adata.obsm
    :param adata: anndata, AnnData object of spatial data.
    :param adatas_sc: anndata, AnnData object of scRNA-seq reference data.
    :param retrain_percent: float, the percentage of cells to retain. default: float=0.1.
    '''
    map_matrix = adata.obsm['map_matrix']
    map_matrix = top_value(map_matrix, retain_percent=retain_percent)
    matrix_cell_type = celltype_matrix(adata_sc)
    matrix_cell_type = matrix_cell_type.values
    matrix_projection = map_matrix.dot(matrix_cell_type)
    cell_type = list(adata_sc.obs['label'].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    df_projection = pandas.DataFrame(matrix_projection, index=adata.obs_names, columns=cell_type)
    df_projection = df_projection.div(df_projection.sum(axis=1), axis=0).fillna(0)
    adata.obs[df_projection.columns] = df_projection