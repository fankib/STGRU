
import torch
from torch import nn

from network import RNN, RNN_user, RNN_cls, RNN_cls_user, RNN_cls_attention

class Trainer():
    
    def __init__(self, use_user_embedding):
        self.model = None
        self.use_user_embedding = use_user_embedding
    
    def prepare(self, loc_count, user_count, hidden_size, device):
        ''' Initializes the model '''
        pass
    
    def parameters(self):
        return self.model.parameters()
    
    def evaluate(self, x, times, y_times, h, active_users):
        ''' takes a sequence x (sequence x users x hidden)
        then does the magic and returns a list of user x locations x sequnce
        describing the probabilities in a per user way
        '''
        pass
    
    def loss(self, x, y, times, y_times, h, active_users):
        ''' takes a sequence x (sequence x users x hidden)
        and corresponding labels (location_id) to
        compute the training loss '''

    def validate():
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
    
    def evaluate(self, x, times, y_times, h, active_users):
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

    def loss(self, x, y, times, y_times, h, active_users):
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
            #self.model = RNN_cls(loc_count, hidden_size, gru_factory).to(device)
            self.model = RNN_cls_attention(loc_count, hidden_size, gru_factory).to(device)
    
    def evaluate(self, x, times, y_times, h, active_users):
        delta_t = (y_times[-1] - times)
        out, h = self.model(x, delta_t, h, active_users)
        out_t = out.transpose(0, 1)
        return out_t, h # model output is directly associated with the ranking per location.
    
    def loss(self, x, y, times, y_times, h, active_users):
        delta_t = (y_times[-1] - times)
        out, h = self.model(x, delta_t, h, active_users)
        seq_len, _, _ = out.shape
        out = out[seq_len-1, :, :] # use only latest in seq len
        y = y[seq_len-1, :]
        out = out.view(-1, self.loc_count)
        y = y.view(-1)
        l = self.cross_entropy_loss(out, y)
        return l, h
    
    