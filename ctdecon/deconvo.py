import torch, pandas, numpy
from tqdm import tqdm
from ._utils import permutation, fixed, preprocess_adj
from ._encoders import encoder, encoder_sc, encoder_map
from .adata_utils import csc_matrix, csr_matrix, preprocess, spot_graph, contrast, get_feature

class config():
    def __init__(self, 
        adata,
        adata_sc,
        device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        learning_rate=0.001,
        learning_rate_sc = 0.01,
        weight_decay=0.00,
        epochs=600, 
        dim_output=64,
        random_seed = 41,
        alpha = 10,
        beta = 1,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = True,
        is_sparse = False
        ):
        '''\
        :param adata: anndata, AnnData object of spatial data.
        :param adata_sc: anndata, AnnData object of scRNA-seq data.
        :param device: str, using GPU or CPU. default: str='cuda'.
        :param learning_rate: float, learning rate for ST representation learning. default: float=0.001.
        :param learning_rate_sc: float, learning rate for scRNA representation learning. default: float=0.01.
        :param weight_decay: float, weight factor to control the influence of weight parameters. default: float=0.00.
        :param epochs: int, epoch for train, train_sc, and train_map. default: float=600.
        :param dim_output: int, dimension of output representation. default: int=64.
        :param random_seed: int, random seed to fix model initialization. default: int=41.
        :param alpha: float, weight factor to control the influence of reconstruction loss in representation learning. default: int=10.
        :param beta: float, weight factor to control the influence of contrastive loss in representation learning. default: int=1.
        :param lamda1: float, weight factor to control the influence of reconstruction loss in mapping matrix learning. default: int=10.
        :param lamda2: float, weight factor to control the influence of contrastive loss in mapping matrix learning. default: int=1.
        :param deconvolution: bool, deconvolution task. default: bool=True.
        :param is_sparse: bool, sparse data. default: bool=False.

        :return: The learned representation 'self.emb_rec'.
        '''
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_sc = learning_rate_sc
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.is_sparse = is_sparse
        fixed(self.random_seed)
        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata)
        if 'adjacent' not in adata.obsm.keys():
            spot_graph(self.adata, self.is_sparse)
        if 'contrastive_label' not in adata.obsm.keys():    
            contrast(self.adata)
        if 'feat' not in adata.obsm.keys():
            get_feature(self.adata, deconvolution=self.deconvolution)
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.contrastive_label = torch.FloatTensor(self.adata.obsm['contrastive_label']).to(self.device)
        self.adjacent = self.adata.obsm['adjacent']
        self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + numpy.eye(self.adjacent.shape[0])).to(self.device)
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        self.adjacent = preprocess_adj(self.adjacent, self.is_sparse).to(self.device)
        if self.deconvolution:
           self.adata_sc = adata_sc.copy() 
           if isinstance(self.adata.X, csc_matrix) or isinstance(self.adata.X, csr_matrix):
              self.feat_sp = adata.X.toarray()[:, ]
           else:
              self.feat_sp = adata.X[:, ]
           if isinstance(self.adata_sc.X, csc_matrix) or isinstance(self.adata_sc.X, csr_matrix):
              self.feat_sc = self.adata_sc.X.toarray()[:, ]
           else:
              self.feat_sc = self.adata_sc.X[:, ]
           self.feat_sc = pandas.DataFrame(self.feat_sc).fillna(0).values
           self.feat_sp = pandas.DataFrame(self.feat_sp).fillna(0).values
           self.feat_sc = torch.FloatTensor(self.feat_sc).to(self.device)
           self.feat_sp = torch.FloatTensor(self.feat_sp).to(self.device)
           if self.adata_sc is not None:
              self.dim_input = self.feat_sc.shape[1] 
           self.n_cell = adata_sc.n_obs
           self.n_spot = adata.n_obs
            
    def train(self):
        self.model = encoder(self.dim_input, self.dim_output, self.graph_neigh, self.is_sparse).to(self.device)
        self.loss_CSL = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        print('Start training spatial transcriptomics data...')
        self.model.train()
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
            self.features_a = permutation(self.features)
            self.hiden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adjacent)
            self.loss_sl_1 = self.loss_CSL(ret, self.contrastive_label)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.contrastive_label)
            self.loss_feat = torch.nn.functional.mse_loss(self.features, self.emb)
            loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2)
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
        print("End training for spatial transcriptomics data.")
        with torch.no_grad():
             self.model.eval()
             if self.deconvolution:
                self.emb_rec = self.model(self.features, self.features_a, self.adjacent)[1]
                return self.emb_rec
             else:
                if self.is_sparse:
                    self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                    self.emb_rec = torch.nn.functional.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
                else:
                    self.emb_rec = self.model(self.features, self.features_a, self.adjacent)[1].detach().cpu().numpy()
                self.adata.obsm['emb'] = self.emb_rec
                return self.adata
         
    def train_sc(self):
        self.model_sc = encoder_sc(self.dim_input).to(self.device)
        self.optimizer_sc = torch.optim.Adam(self.model_sc.parameters(), lr=self.learning_rate_sc)  
        print('Start training scRNA-seq data...')
        for epoch in tqdm(range(self.epochs)):
            self.model_sc.train()
            emb = self.model_sc(self.feat_sc)
            loss = torch.nn.functional.mse_loss(emb, self.feat_sc)
            self.optimizer_sc.zero_grad()
            loss.backward()
            self.optimizer_sc.step()
        print("End learning for cell representation.")
        with torch.no_grad():
            self.model_sc.eval()
            emb_sc = self.model_sc(self.feat_sc)
            return emb_sc
        
    def train_map(self):
        emb_sp = self.train()
        emb_sc = self.train_sc()
        self.adata.obsm['emb_sp'] = emb_sp.detach().cpu().numpy()
        self.adata_sc.obsm['emb_sc'] = emb_sc.detach().cpu().numpy()
        emb_sp = torch.nn.functional.normalize(emb_sp, p=2, eps=1e-12, dim=1)
        emb_sc = torch.nn.functional.normalize(emb_sc, p=2, eps=1e-12, dim=1)
        self.model_map = encoder_map(self.n_cell, self.n_spot).to(self.device)  
        self.optimizer_map = torch.optim.Adam(self.model_map.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        print('Start learning mapping matrix...')
        for epoch in tqdm(range(self.epochs)):
            self.model_map.train()
            self.map_matrix = self.model_map()
            loss_recon, loss_NCE = self.loss(emb_sp, emb_sc)
            loss = self.lamda1*loss_recon + self.lamda2*loss_NCE 
            self.optimizer_map.zero_grad()
            loss.backward()
            self.optimizer_map.step() 
        print("End learning mapping matrix.")
        with torch.no_grad():
            self.model_map.eval()
            emb_sp = emb_sp.cpu().numpy()
            emb_sc = emb_sc.cpu().numpy()
            map_matrix = torch.nn.functional.softmax(self.map_matrix, dim=1).cpu().numpy()
            self.adata.obsm['emb_sp'] = emb_sp
            self.adata_sc.obsm['emb_sc'] = emb_sc
            self.adata.obsm['map_matrix'] = map_matrix.T
            return self.adata, self.adata_sc
    
    def loss(self, emb_sp, emb_sc):
        '''
        :param emb_sp: torch.Tensor, Spatial spot representation matrix.
        :param emb_sc: torch.Tensor, scRNA cell representation matrix.

        :return: loss values.
        '''
        map_probs = torch.nn.functional.softmax(self.map_matrix, dim=1)
        self.pred_sp = torch.matmul(map_probs.t(), emb_sc)
        loss_recon = torch.nn.functional.mse_loss(self.pred_sp, emb_sp, reduction='mean')
        loss_NCE = self.Noise_Cross_Entropy(self.pred_sp, emb_sp)
        return loss_recon, loss_NCE
        
    def Noise_Cross_Entropy(self, pred_sp, emb_sp):
        '''
        Calculate noise cross entropy. Considering spatial neighbors as positive pairs for each spot
        :param pred_sp: torch.Tensor, Predicted spatial gene expression matrix.
        :param emb_sp: torch.Tensor, Reconstructed spatial gene expression matrix.

        :return: loss value.
        '''
        mat = self.cosine_similarity(pred_sp, emb_sp) 
        k = torch.exp(mat).sum(axis=1) - torch.exp(torch.diag(mat, 0))
        p = torch.exp(mat)
        p = torch.mul(p, self.graph_neigh).sum(axis=1)
        ave = torch.div(p, k)
        loss = - torch.log(ave).mean()
        return loss
    
    def cosine_similarity(self, pred_sp, emb_sp):
        '''\
        Calculate cosine similarity based on predicted and reconstructed gene expression matrix.    
        '''
        M = torch.matmul(pred_sp, emb_sp.T)
        Norm_c = torch.norm(pred_sp, p=2, dim=1)
        Norm_s = torch.norm(emb_sp, p=2, dim=1)
        Norm = torch.matmul(Norm_c.reshape((pred_sp.shape[0], 1)), Norm_s.reshape((emb_sp.shape[0], 1)).T) + -5e-12
        M = torch.div(M, Norm)
        if torch.any(torch.isnan(M)):
           M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)
        return M