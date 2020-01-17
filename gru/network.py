import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from enum import Enum

from gru import OwnGRU, OwnLSTM, STGN, STGCN

class GRU(Enum):
    PYTORCH_GRU = 0
    OWN_GRU = 1
    RNN = 2
    LSTM = 3
    OWN_LSTM = 4
    STGN = 5
    STGCN = 6
    
    @staticmethod
    def from_string(name):
        if name == 'pytorch':
            return GRU.PYTORCH_GRU
        if name == 'own':
            return GRU.OWN_GRU
        if name == 'rnn':
            return GRU.RNN
        if name == 'lstm':
            return GRU.LSTM
        if name == 'ownlstm':
            return GRU.OWN_LSTM
        if name == 'stgn':
            return GRU.STGN
        if name == 'stgcn':
            return GRU.STGCN
        raise ValueError('{} not supported'.format(name))
        

class GruFactory():
    
    def __init__(self, gru_type_str):
        self.gru_type = GRU.from_string(gru_type_str)
    
    def is_lstm(self):
        return self.gru_type in [GRU.LSTM, GRU.OWN_LSTM, GRU.STGN, GRU.STGCN]
    
    def is_stgn(self):
        return self.gru_type in [GRU.STGN, GRU.STGCN]
        
    def greeter(self):
        if self.gru_type == GRU.PYTORCH_GRU:
            return 'Use pytorch GRU implementation.'
        if self.gru_type == GRU.OWN_GRU:
            return 'Use *own* GRU implementation.'
        if self.gru_type == GRU.RNN:
            return 'Use vanilla pytorch RNN implementation.'
        if self.gru_type == GRU.LSTM:
            return 'Use pytorch LSTM implementation.'
        if self.gru_type == GRU.OWN_LSTM:
            return 'Use *own* LSTM implementation.'
        if self.gru_type == GRU.STGN:
            return 'Use STGN variant.'
        if self.gru_type == GRU.STGCN:
            return 'Use STG*C*N variant.'
        
    def create(self, hidden_size):
        if self.gru_type == GRU.PYTORCH_GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.gru_type == GRU.OWN_GRU:
            return OwnGRU(hidden_size)
        if self.gru_type == GRU.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.gru_type == GRU.LSTM:
            return nn.LSTM(hidden_size, hidden_size)
        if self.gru_type == GRU.OWN_LSTM:
            return OwnLSTM(hidden_size)
        if self.gru_type == GRU.STGN:
            return STGN(hidden_size)
        if self.gru_type == GRU.STGCN:
            return STGCN(hidden_size)
        

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

    def forward(self, x, h, active_user):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)
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

class RNN_stgn(nn.Module):
    ''' STGN based RNN used for bpr, using own weights '''
    
    def __init__(self, input_size, hidden_size, gru_factory):
        super(RNN_stgn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = gru_factory.create(hidden_size)

    def forward(self, x, delta_t, delta_s, h):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)
        out, h = self.gru(x_emb, delta_t, delta_s, h)
        y_linear = self.fc(out) # use own weights!
        return y_linear, h

class RNN_cls_stgn(nn.Module):
    ''' STGN based RNN used for cross entropy loss and one linear output layer '''
    
    def __init__(self, input_size, hidden_size, gru_factory):
        super(RNN_cls_stgn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = gru_factory.create(hidden_size)
        self.fc = nn.Linear(hidden_size, input_size) # create outputs in lenght of locations

    def forward(self, x, delta_t, delta_s, h):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)
        out, h = self.gru(x_emb, delta_t, delta_s, h)
        y_linear = self.fc(out)
        return y_linear, h

class RNN_cls_st(nn.Module):
    ''' GRU based rnn. applies weighted average using spatial and temporal data '''
    
    def __init__(self, input_size, hidden_size, f_t, f_s, gru_factory):
        super(RNN_cls_st, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.f_t = f_t # function for computing temporal weight
        self.f_s = f_s # function for computing spatial weight

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.gru = gru_factory.create(hidden_size)
        self.fc = nn.Linear(hidden_size, input_size) # create outputs in lenght of locations

    def forward(self, x, t, s, y_t, y_s, h, active_user):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)
        out, h = self.gru(x_emb, h)
        
        # comopute weights per
        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)
            for j in range(i+1):
                dist_t = t[i] - t[j]
                dist_s = torch.norm(s[i] - s[j], dim=-1)
                a_j = self.f_t(dist_t, user_len) # (torch.cos(cummulative_t[j]*2*np.pi / 86400) + 1) / 2 #
                b_j = self.f_s(dist_s, user_len)
                a_j = a_j.unsqueeze(1)
                b_j = b_j.unsqueeze(1)
                w_j = a_j*b_j + 1e-10 # small epsilon to have no 0 division
                sum_w += w_j
                out_w[i] += w_j*out[j] # could be factored out into a matrix!
            # normliaze according to weights
            out_w[i] /= sum_w
        
        y_linear = self.fc(out_w)
        return y_linear, h
    
class RNN_cls_st_user(nn.Module):
    ''' GRU based rnn. applies weighted average using spatial and temporal data WITH user embeddings'''
    
    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, gru_factory):
        super(RNN_cls_st_user, self).__init__()
        self.input_size = input_size
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t # function for computing temporal weight
        self.f_s = f_s # function for computing spatial weight

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.user_encoder = nn.Embedding(user_count, hidden_size)
        self.gru = gru_factory.create(hidden_size)
        self.fc = nn.Linear(2*hidden_size, input_size) # create outputs in lenght of locations

    def forward(self, x, t, s, y_t, y_s, h, active_user):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)        
        out, h = self.gru(x_emb, h)
        
        # comopute weights per
        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)
        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)
            for j in range(i+1):
                dist_t = t[i] - t[j]
                dist_s = torch.norm(s[i] - s[j], dim=-1)
                a_j = self.f_t(dist_t, user_len) # (torch.cos(cummulative_t[j]*2*np.pi / 86400) + 1) / 2 #
                b_j = self.f_s(dist_s, user_len)
                a_j = a_j.unsqueeze(1)
                b_j = b_j.unsqueeze(1)
                w_j = a_j*b_j + 1e-10 # small epsilon to have no 0 division
                sum_w += w_j
                out_w[i] += w_j*out[j] # could be factored out into a matrix!
            # normliaze according to weights
            out_w[i] /= sum_w
        
        # add user embedding:
        p_u = self.user_encoder(active_user)
        p_u = p_u.view(user_len, self.hidden_size)
        out_pu = torch.zeros(seq_len, user_len, 2*self.hidden_size, device=x.device)
        for i in range(seq_len):
            out_pu[i] = torch.cat([out_w[i], p_u], dim=1)
        y_linear = self.fc(out_pu)
        return y_linear, h