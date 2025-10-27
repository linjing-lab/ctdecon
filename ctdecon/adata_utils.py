import scanpy, numpy, ot
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from ._utils import permutation

def preprocess(adata):
    scanpy.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    scanpy.pp.normalize_total(adata, target_sum=1e4)
    scanpy.pp.log1p(adata)
    scanpy.pp.scale(adata, zero_center=False, max_value=10)

def spot_graph(adata, is_sparse=False, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    if is_sparse:
        n_spot = position.shape[0]
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
        _ , indices = nbrs.kneighbors(position)
        x = indices[:, 0].repeat(n_neighbors)
        y = indices[:, 1:].flatten()
        interaction = numpy.zeros([n_spot, n_spot])
        interaction[x, y] = 1
    else:
        distance_matrix = ot.dist(position, position, metric='euclidean')
        n_spot = distance_matrix.shape[0]
        adata.obsm['distance_matrix'] = distance_matrix
        interaction = numpy.zeros([n_spot, n_spot])  
        for i in range(n_spot):
            vec = distance_matrix[i, :]
            distance = vec.argsort()
            for t in range(1, n_neighbors + 1):
                y = distance[t]
                interaction[i, y] = 1
    adata.obsm['graph_neigh'] = interaction
    adjacent = interaction
    adjacent = adjacent + adjacent.T
    adjacent = numpy.where(adjacent>1, 1, adjacent)
    adata.obsm['adjacent'] = adjacent

def contrast(adata):
    n_spot = adata.n_obs
    one_matrix = numpy.ones([n_spot, 1])
    zero_matrix = numpy.zeros([n_spot, 1])
    contrastive_label = numpy.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['contrastive_label'] = contrastive_label

def get_feature(adata, deconvolution=True):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    feat_a = permutation(feat)
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a