
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    ''' GRU based RNN, using embeddings and one linear output layer '''
    
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # move out of model as it is too large
        self.encoder = nn.Embedding(input_size, hidden_size, sparse=True)
        #self.encoder.weight.requires_grad = False
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        seq_len, batch_len = x.size()
        x_emb = self.encoder(x)
        out, h = self.gru(x_emb, h)
        y_linear = self.fc(out)
        return y_linear, h
    
class CpuEncoder(nn.Module):
    ''' wraps embeddings on RAM '''
    
    def __init__(self, input_size, hidden_size):
        super(CpuEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.encoder = nn.Embedding(input_size, hidden_size)
    
    def active_locations(self, P):
        # P is the 'permutation' of active locations:
        return P * self.encoder.weight

    def forward(self, x, P, poi2id):
        pass
