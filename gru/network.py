import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import numpy as np

from gru import OwnGRU

class GRU(Enum):
    PYTORCH_GRU = 0
    OWN_GRU = 1
    LSTM = 2
    
    @staticmethod
    def from_string(name):
        if name == 'pytorch':
            return GRU.PYTORCH_GRU
        if name == 'own':
            return GRU.OWN_GRU
        if name == 'lstm':
            return GRU.LSTM
        raise ValueError('{} not supported'.format(name))
        

class GruFactory():
    
    def __init__(self, gru_type_str):
        self.gru_type = GRU.from_string(gru_type_str)
        
    def greeter(self):
        if self.gru_type == GRU.PYTORCH_GRU:
            return 'Use pytorch GRU implementation.'
        if self.gru_type == GRU.OWN_GRU:
            return 'Use *own* GRU implementation.'
        if self.gru_type == GRU.LSTM:
            return 'Use pytorch LSTM implementation.'
        
    def create(self, hidden_size):
        if self.gru_type == GRU.PYTORCH_GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.gru_type == GRU.OWN_GRU:
            return OwnGRU(hidden_size)
        if self.gru_type == GRU.LSTM:
            raise Exception('not yet implemented')
            #return nn.LSTM(hidden_size, hidden_size)
        

class RNN(nn.Module):
    ''' GRU based RNN, using embeddings and one linear output layer '''
    
    def __init__(self, input_size, hidden_size, gru_factory):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = gru_factory.create(hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h, active_user):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)
        out, h = self.gru(x_emb, h)
        #out, (h, c) = self.gru(x_emb) # lstm hack
        y_linear = self.fc(out)
        return y_linear, h
    
class RNN_user(nn.Module):
    ''' GRU based RNN, with user embeddings and one linear output layer '''
    
    def __init__(self, input_size, user_count, hidden_size, gru_factory):
        super(RNN_user, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.user_count = user_count

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.user_encoder = nn.Embedding(user_count, hidden_size)
        self.gru = gru_factory.create(hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h, active_user):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)
        out, h = self.gru(x_emb, h)
        y_linear = self.fc(out)
        p_u = self.user_encoder(active_user)
        p_u = p_u.view(1, user_len, self.hidden_size)
        return (y_linear + p_u), h

class RNN_cls(nn.Module):
    ''' GRU based RNN used for cross entropy loss, using embeddings and one linear output layer '''
    
    def __init__(self, input_size, hidden_size, gru_factory):
        super(RNN_cls, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = gru_factory.create(hidden_size)
        self.fc = nn.Linear(hidden_size, input_size) # create outputs in lenght of locations

    def forward(self, x, delta_t, h, active_user):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)
        out, h = self.gru(x_emb, h)
        y_linear = self.fc(out)
        return y_linear, h

class RNN_cls_attention(nn.Module):
    ''' GRU based RNN used for cross entropy loss, using embeddings and one linear output layer '''
    
    def __init__(self, input_size, hidden_size, gru_factory):
        super(RNN_cls_attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = gru_factory.create(hidden_size)
        self.fc = nn.Linear(hidden_size, input_size) # create outputs in lenght of locations
        
        
        self.fc_a0 = nn.Linear(2, 2)
        
        self.fc_a = nn.Linear(2, 1)
        self.fc_a.weight.data[0] +=1.
        self.fc_a.bias.data[0] -= 1.
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, delta_t, h, active_user):
        seq_len, user_len = x.size()
        # learn attention:
        delta_t = delta_t.unsqueeze(2)
        t_1 = torch.cos(delta_t * 2*np.pi / 86400) # your daily cos
        t_2 = torch.sin(delta_t * 2*np.pi / 86400) # your daily sin
        t = torch.cat([t_1, t_2], dim=2)
        a0 = self.fc_a0(t)
        a0 = self.relu(a0)
        #a0 = a0 * a0
        a = 1. - self.sigmoid(self.fc_a(a0))
        x_emb = self.encoder(x)
        x_emb = a*x_emb
        out, h = self.gru(x_emb, h)
        y_linear = self.fc(out)
        return y_linear, h

class RNN_cls_user(nn.Module):
    ''' GRU based RNN used for cross entropy loss with user embeddings '''
    
    def __init__(self, input_size, user_count, hidden_size, gru_factory):
        super(RNN_cls_user, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.user_count = user_count

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.user_encoder = nn.Embedding(user_count, hidden_size)
        self.gru = gru_factory.create(hidden_size)
        self.fc = nn.Linear(2*hidden_size, input_size) # create outputs in lenght of locations

    def forward(self, x, h, active_user):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)
        out, h = self.gru(x_emb, h)
        p_u = self.user_encoder(active_user)
        p_u = p_u.view(user_len, self.hidden_size)
        # boradcast on sequence (concat user embeddings):
        out_pu = torch.zeros(seq_len, user_len, 2*self.hidden_size, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([out[i], p_u], dim=1)
        y_linear = self.fc(out_pu)        
        return y_linear, h