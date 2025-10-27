import numpy, torch, pandas, os, random
from torch.backends import cudnn
import scipy.sparse as sp

def permutation(feature):
    # fix_seed(FLAGS.random_seed) 
    ids = numpy.arange(feature.shape[0])
    ids = numpy.random.permutation(ids)
    feature_permutated = feature[ids]
    return feature_permutated

def fixed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def normalized(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = numpy.array(adj.sum(1))
    d_inv_sqrt = numpy.power(rowsum, -0.5).flatten()
    d_inv_sqrt[numpy.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def sparse2torch(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(numpy.float32)
    indices = torch.from_numpy(numpy.vstack((sparse_mx.row, sparse_mx.col)).astype(numpy.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def preprocess_adj(adj, is_sparse=False):
    if is_sparse:
        adj = sp.coo_matrix(adj)
        adj_ = adj + sp.eye(adj.shape[0])
        rowsum = numpy.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(numpy.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return sparse2torch(adj_normalized)
    else:
        return torch.FloatTensor(normalized(adj)+numpy.eye(adj.shape[0]))

class Discriminator(torch.nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = torch.nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, torch.nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  
        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 1)
        return logits
    
class AvgReadout(torch.nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
        return torch.nn.functional.normalize(global_emb, p=2, dim=1)
    
def top_value(map_matrix, retain_percent = 0.1): 
    '''\
    Filter out cells with low mapping probability
    :param map_matrix: numpy.ndarray, mapped matrix with m spots and n cells.
    :param retain_percent: float, The percentage of cells to retain. default: float=0.1.

    :return: numpy.ndarray, retain top 1% values for each spot with fitered mapped matrix.
    '''
    top_k  = retain_percent * map_matrix.shape[1]
    output = map_matrix * (numpy.argsort(numpy.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)
    return output 

def celltype_matrix(adata_sc):
    label = 'label'
    n_type = len(list(adata_sc.obs[label].unique()))
    zeros = numpy.zeros([adata_sc.n_obs, n_type])
    cell_type = list(adata_sc.obs[label].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    mat = pandas.DataFrame(zeros, index=adata_sc.obs_names, columns=cell_type)
    for cell in list(adata_sc.obs_names):
        ctype = adata_sc.obs.loc[cell, label]
        mat.loc[cell, str(ctype)] = 1
    #res = mat.sum()
    return mat 