import torch
import torch.nn as nn

class OwnGRU(nn.Module):
    
    def __init__(self, hidden_size):
        super(OwnGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # some initialization:
        self.mu = 0
        self.sd = 1/(hidden_size**2)
        
        # input weights:
        self.Wz = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Wr = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Wh = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        
        # hidden state weights:
        self.Uz = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Ur = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Uh = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        
        # bias terms:
        self.bz = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.br = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.bh = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)        
        
    def forward(self, x, h):
        ''' input x: sequence x users x hidden
            input h: 1 x users x hidden '''
        seq_len, user_len, _ = x.size()
        out = []
        h = h[0].view(user_len, self.hidden_size, 1) # convert view on h
        for i in range(seq_len):
            x_t = x[i].view(user_len, self.hidden_size, 1) # users x hidden x 1
            z = torch.sigmoid(torch.matmul(self.Wz, x_t) + torch.matmul(self.Uz, h) + self.bz)
            r = torch.sigmoid(torch.matmul(self.Wr, x_t) + torch.matmul(self.Ur, h) + self.br)
            h_tilde = torch.tanh(torch.matmul(self.Wh, x_t) + r*torch.matmul(self.Uh, h) + self.bh)
            h = z*h + (1-z)*h_tilde
            out.append(h.view(user_len, self.hidden_size))
        out = torch.stack(out, dim=0).view(seq_len, user_len, self.hidden_size)
        return out, h.view(1, user_len, self.hidden_size)
            