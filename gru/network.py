
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    ''' GRU based RNN, using embeddings and one linear output layer '''
    
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # move out of model as it is too large
        self.encoder = nn.Embedding(input_size, hidden_size)
        #self.encoder.weight.requires_grad = False
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)
        out, h = self.gru(x_emb, h)
        y_linear = self.fc(out)
        return y_linear, h

class RNN_cls(nn.Module):
    ''' GRU based RNN used for cross entropy loss, using embeddings and one linear output layer '''
    
    def __init__(self, input_size, hidden_size):
        super(RNN_cls, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # move out of model as it is too large
        self.encoder = nn.Embedding(input_size, hidden_size)
        #self.encoder.weight.requires_grad = False
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, input_size) # create outputs in lenght of locations

    def forward(self, x, h):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)
        out, h = self.gru(x_emb, h)
        y_linear = self.fc(out)
        return y_linear, h
