import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from ._utils import Discriminator, AvgReadout
from kan.KANLayer import KANLayer # pykan

class encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, is_sparse=False, dropout=0.0, act=torch.nn.functional.elu):
        super(encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.is_sparse = is_sparse
        self.dropout = dropout
        self.act = act
        self.kan_layer1 = KANLayer(self.in_features, self.out_features)
        self.kan_layer2 = KANLayer(self.out_features, self.in_features)
        self.disc = Discriminator(self.out_features)
        self.sigm = torch.nn.Sigmoid()
        self.read = AvgReadout()

    def forward(self, feat, feat_a, adj):
        z = torch.nn.functional.dropout(feat, self.dropout, self.training)
        z, _, _, _ = self.kan_layer1(z)
        z = torch.spmm(adj, z) if self.is_sparse else torch.mm(adj, z)
        hiden_emb = z
        h, _, _, _ = self.kan_layer2(z)
        h = torch.spmm(adj, h) if self.is_sparse else torch.mm(adj, h)
        emb = self.act(z)
        z_a = torch.nn.functional.dropout(feat_a, self.dropout, self.training)
        z_a, _, _, _ = self.kan_layer1(z_a)
        z_a = torch.spmm(adj, z_a) if self.is_sparse else torch.mm(adj, z_a)
        emb_a = self.act(z_a)
        g = self.read(emb, self.graph_neigh) 
        g = self.sigm(g)  
        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)  
        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        return hiden_emb, h, ret, ret_a     

class encoder_sc(torch.nn.Module):
    def __init__(self, dim_input, dropout=0.0):
        super(encoder_sc, self).__init__()
        self.dim_input = dim_input
        self.dim1 = 256
        self.dim2 = 64
        self.dropout = dropout
        self.weight1_en = Parameter(torch.FloatTensor(self.dim_input, self.dim1))
        self.weight2_en = Parameter(torch.FloatTensor(self.dim1, self.dim2))
        self.weight1_de = Parameter(torch.FloatTensor(self.dim2, self.dim1))
        self.weight2_de = Parameter(torch.FloatTensor(self.dim1, self.dim_input))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1_en)
        torch.nn.init.xavier_uniform_(self.weight1_de)
        torch.nn.init.xavier_uniform_(self.weight2_en)
        torch.nn.init.xavier_uniform_(self.weight2_de)
        
    def forward(self, x):
        x = torch.nn.functional.dropout(x, self.dropout, self.training)
        x = torch.mm(x, self.weight1_en)
        x = torch.mm(x, self.weight2_en)
        x = torch.mm(x, self.weight1_de)
        x = torch.mm(x, self.weight2_de)
        return x
    
class encoder_map(torch.nn.Module):
    def __init__(self, n_cell, n_spot):
        super(encoder_map, self).__init__()
        self.n_cell = n_cell
        self.n_spot = n_spot
          
        self.M = Parameter(torch.FloatTensor(self.n_cell, self.n_spot))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.M)
    
    def forward(self):
        x = self.M
        
        return x 