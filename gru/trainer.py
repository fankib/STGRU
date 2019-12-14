
import torch
from torch import nn

from network import RNN, RNN_cls

class Trainer():
    
    def __init__(self):
        self.model = None
    
    def prepare(self, loc_count, hidden_size, device):
        ''' Initializes the model '''
        pass
    
    def set_batch_params(self, seq_length, user_length):
        self.seq_length = seq_length
        self.user_length = user_length
    
    def parameters(self):
        return self.model.parameters()
    
    def evaluate(self, x, h):
        ''' takes a sequence x (sequence x users x hidden)
        then does the magic and returns a list of user x locations x sequnce
        describing the probabilities in a per user way
        '''
        pass
    
    def loss(self, x, y, h):
        ''' takes a sequence x (sequence x users x hidden)
        and corresponding labels (location_id) to
        compute the training loss '''

    def validate():
        pass
    

class BprTrainer(Trainer):
    
    def prepare(self, loc_count, hidden_size, device):
        self.hidden_size = hidden_size
        self.model = RNN(loc_count, hidden_size).to(device)
    
    def evaluate(self, x, h):
        seq_length = x.shape[0]
        user_length = x.shape[1]
        out, h = self.model(x, h)
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

    def loss(self, x, y, h):
        out, h = self.model(x, h)
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
    
    def prepare(self, loc_count, hidden_size, device):
        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.model = RNN_cls(loc_count, hidden_size).to(device)
    
    def evaluate(self, x, h):
        out, h = self.model(x, h)
        out_t = out.transpose(0, 1)
        return out_t, h # model output is directly associated with the ranking per location.
    
    def loss(self, x, y, h):
        out, h = self.model(x, h)
        out = out.view(-1, self.loc_count)
        y = y.view(-1)
        l = self.cross_entropy_loss(out, y)
        return l, h
    
    