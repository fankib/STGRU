
import torch
from torch import nn
import numpy as np

from network import RNN, RNN_user, RNN_cls, RNN_cls_user, RNN_cls_st, RNN_cls_st_user, RNN_cls_stgn

class TrainerFactory():
    
    def create(self, cross_entropy, user_embedding, temporal, spatial, lambda_t, lambda_s, is_stgn):
        bpr = not cross_entropy
        
        if is_stgn:
            assert cross_entropy
            assert temporal
            assert spatial
            assert not user_embedding
            return STGNTrainer()

        if bpr:
            assert not temporal
            assert not spatial
            return BprTrainer(user_embedding)
        if cross_entropy:
            if not temporal and not spatial:
                return CrossEntropyTrainer(user_embedding)
            else:
                return SpatialTemporalCrossEntropyTrainer(user_embedding, temporal, spatial, lambda_t, lambda_s)

class Trainer():
    
    def __init__(self, use_user_embedding):
        self.model = None
        self.use_user_embedding = use_user_embedding
    
    def prepare(self, loc_count, user_count, hidden_size, device):
        ''' Initializes the model '''
        pass
    
    def parameters(self):
        return self.model.parameters()
    
    def evaluate(self, x, t, s, y_t, y_s, h, active_users):
        ''' takes a sequence "x" (sequence x users x hidden)
        then does the magic and returns a list of user x locations x sequence
        describing the probabilities in a per user way
        t, s are temporal and spatial data related to x
        y_t, y_s are temporal and spatial data related to y which we will predict
        '''
        pass
    
    def loss(self, x, t, s, y, y_t, y_s, h, active_users):
        ''' takes a sequence "x" (sequence x users x hidden)
        and corresponding labels (location_id) to
        compute the training loss '''

    def validate():
        pass
    
    def debug(self):
        ''' is called after each epoch in order to print debug information '''
        pass
    

class BprTrainer(Trainer):        
    
    def greeter(self):
        if not self.use_user_embedding:
            return 'Use BPR training.'
        return 'Use BPR training with user embeddings.'
    
    def prepare(self, loc_count, user_count, hidden_size, gru_factory, device):
        self.hidden_size = hidden_size
        if self.use_user_embedding:
            self.model = RNN_user(loc_count, user_count, hidden_size, gru_factory).to(device)
        else:
            self.model = RNN(loc_count, hidden_size, gru_factory).to(device)
    
    def evaluate(self, x, t, s, y_t, y_s, h, active_users):
        seq_length = x.shape[0]
        user_length = x.shape[1]
        out, h = self.model(x, h, active_users)
        out_t = out.transpose(0, 1)
        response = []
        Q = self.model.encoder.weight
        for j in range(user_length):
            out_j = out_t[j].transpose(0,1)
            o = torch.matmul(Q, out_j).cpu().detach()
            o = o.transpose(0,1)
            o = o.contiguous().view(seq_length, -1)
            response.append(o)
        return response, h

    def loss(self, x, t, s, y, y_t, y_s, h, active_users):
        out, h = self.model(x, h, active_users)
        y_emb = self.model.encoder(y)
        
        # reshape
        out = out.view(-1, self.hidden_size)
        out_t = out.transpose(0, 1)
        y_emb = y_emb.contiguous().view(-1, self.hidden_size)
        Q = self.model.encoder.weight
        
        neg_o = torch.matmul(Q, out_t)
        pos_o = torch.matmul(y_emb, out_t).diag()
        
        l = torch.log(1 + torch.exp(-(pos_o - neg_o)))
        l = torch.mean(l)
        return l, h
    
class CrossEntropyTrainer(Trainer):
    
    def greeter(self):
        if not self.use_user_embedding:
            return 'Use Cross Entropy training.'
        return 'Use Cross Entropy training with user embeddings.'
    
    def prepare(self, loc_count, user_count, hidden_size, gru_factory, device):
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        if self.use_user_embedding:
            self.model = RNN_cls_user(loc_count, user_count, hidden_size, gru_factory).to(device)
        else:
            self.model = RNN_cls(loc_count, hidden_size, gru_factory).to(device)
    
    def evaluate(self, x, t, s, y_t, y_s, h, active_users):
        out, h = self.model(x, h, active_users)
        out_t = out.transpose(0, 1)
        return out_t, h # model output is directly associated with the ranking per location.
    
    def loss(self, x, t, s, y, y_t, y_s, h, active_users):
        out, h = self.model(x, h, active_users)
        out = out.view(-1, self.loc_count)
        y = y.view(-1)
        l = self.cross_entropy_loss(out, y)
        return l, h

class STGNTrainer(Trainer):
    
    def __init__(self):
        super(STGNTrainer, self).__init__(False)
    
    def greeter(self):
        return 'Do STGN training.'
    
    def debug(self):
        pass
    
    def parameters(self):
        return self.model.parameters()
    
    def prepare(self, loc_count, user_count, hidden_size, gru_factory, device):
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.model = RNN_cls_stgn(loc_count, hidden_size, gru_factory).to(device)

    def evaluate(self, x, t, s, y_t, y_s, h, active_users):
        delta_t = y_t - t
        delta_s = torch.norm(y_s - s, dim=-1)
        out, h = self.model(x, delta_t, delta_s, h)
        out_t = out.transpose(0, 1)
        return out_t, h # model output is directly associated with the ranking per location.
    
    def loss(self, x, t, s, y, y_t, y_s, h, active_users):
        with torch.no_grad():
            delta_t = y_t - t
            delta_s = torch.norm(y_s - s, dim=-1)
        out, h = self.model(x, delta_t, delta_s, h)
        out = out.view(-1, self.loc_count)
        y = y.view(-1)
        l = self.cross_entropy_loss(out, y)
        return l, h

class SpatialTemporalCrossEntropyTrainer(Trainer):
    
    def __init__(self, use_user_embedding, use_temporal, use_spatial, lambda_t, lambda_s):
        super(SpatialTemporalCrossEntropyTrainer, self).__init__(use_user_embedding)
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
    
    def greeter(self):
        if not self.use_user_embedding:
            if self.use_temporal and not self.use_spatial:
                return 'Use Temporal Cross Entropy training.'
            if self.use_spatial and not self.use_temporal:
                return 'Use Spatial Cross Entropy training.'
            return 'Use Spatial and Temporal Cross Entropy training.'
        else:
            if self.use_temporal and not self.use_spatial:
                return 'Use Temporal Cross Entropy training with user embeddings.'
            if self.use_spatial and not self.use_temporal:
                return 'Use Spatial Cross Entropy training with user embeddings.'
            return 'Use Spatial and Temporal Cross Entropy training with user embeddings.'
    
    def debug(self):
        print('As:', self.As)
        print('At:', self.At)
        pass
    
    def parameters(self):
        #return list(self.model.parameters()) + list([self.a, self.b]) 
        return list(self.model.parameters()) + list([self.At, self.As])
        
    
    def prepare(self, loc_count, user_count, hidden_size, gru_factory, device):
        if self.use_temporal:
            f_t = lambda delta_t, user_len: ((torch.cos(delta_t*2*np.pi/86400) + 1) / 2)*torch.exp(-(delta_t/86400*self.lambda_t))
        else:
            f_t = lambda delta_t, user_len: torch.ones(user_len, device=device)
        
        if self.use_spatial:
            f_s = lambda delta_s, user_len: torch.exp(-(delta_s*self.lambda_s))
        else:
            f_s = lambda delta_s, user_len: torch.ones(user_len, device=device)
        
        mu = 1.0
        sd = 0.1
        self.At = nn.Parameter(torch.randn(1, 6, 1)*sd + mu)
        self.As = nn.Parameter(torch.randn(1)*sd + mu)
        
        
        #torch.stack([torch.cos(delta_t*2*np.pi/3600),torch.sin(delta_t*2*np.pi/3600)], dim=0]
        
        USE_SPECIAL = True
        def special_t(delta_t, user_len, At, lambda_t, device):
            # encode time:
            time_emb = torch.stack([torch.cos(delta_t*2*np.pi/3600),\
                    torch.sin(delta_t*2*np.pi/3600),\
                    torch.cos(delta_t*2*np.pi/86400),\
                    torch.sin(delta_t*2*np.pi/86400),\
                    torch.cos(delta_t*2*np.pi/604800),\
                    torch.sin(delta_t*2*np.pi/604800),\
                    ], dim=1).unsqueeze(2)
            weight = torch.sigmoid(torch.matmul(At.transpose(1,2), time_emb)).squeeze()
            decay = torch.exp(-(delta_t/86400*lambda_t))
            return weight*decay
        def special_s(delta_s, user_len, As, lambda_s, device):
            return torch.sigmoid(As*delta_s)*torch.exp(-(delta_s*self.lambda_s))
        if USE_SPECIAL: 
            self.lambda_t = 0.2
            self.lambda_s = 0.2
            f_t = lambda delta_t, user_len: special_t(delta_t, user_len, self.At, self.lambda_t, device)
            f_s = lambda delta_s, user_len: special_s(delta_s, user_len, self.As, self.lambda_s, device)
            
        
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        if self.use_user_embedding:
            self.model = RNN_cls_st_user(loc_count, user_count, hidden_size, f_t, f_s, gru_factory).to(device)
        else:
            self.model = RNN_cls_st(loc_count, hidden_size, f_t, f_s, gru_factory).to(device)
    
    def evaluate(self, x, t, s, y_t, y_s, h, active_users):
        delta_t = y_t - t
        out, h = self.model(x, delta_t, s, y_s, h, active_users)
        out_t = out.transpose(0, 1)
        return out_t, h # model output is directly associated with the ranking per location.
    
    def loss(self, x, t, s, y, y_t, y_s, h, active_users):
        delta_t = y_t - t
        out, h = self.model(x, delta_t, s, y_s, h, active_users)
        out = out.view(-1, self.loc_count)
        y = y.view(-1)
        l = self.cross_entropy_loss(out, y)
        return l, h
    
    