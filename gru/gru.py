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
    
class OwnLSTM(nn.Module):
    
    def __init__(self, hidden_size):
        super(OwnLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # some initialization
        self.mu = 0
        self.sd = 1/(hidden_size**2)
        
        # input weights:
        self.Wi = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Wf = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Wo = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Wc = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)

        # hidden state weights:
        self.Ui = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Uf = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Uo = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Uc = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)

        # bias terms:
        self.bi = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.bf = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.bo = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)           
        self.bc = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        
    def forward(self, x, hc):
        seq_len, user_len, _ = x.size()
        h, c = hc
        out = []
        h = h[0].unsqueeze(2)
        c = c[0].unsqueeze(2)
        for j in range(seq_len):
            x_t = x[j].view(user_len, self.hidden_size, 1)
            i = torch.sigmoid(torch.matmul(self.Wi, x_t) + torch.matmul(self.Ui, h) + self.bi)
            f = torch.sigmoid(torch.matmul(self.Wf, x_t) + torch.matmul(self.Uf, h) + self.bf)
            o = torch.sigmoid(torch.matmul(self.Wo, x_t) + torch.matmul(self.Uo, h) + self.bo)
            c_tilde = torch.tanh(torch.matmul(self.Wc, x_t) + torch.matmul(self.Uc, h) + self.bc)
            c = f*c + i*c_tilde
            h = o*torch.tanh(c)
            out.append(h.squeeze())
        out = torch.stack(out, dim=0).view(seq_len, user_len, self.hidden_size)
        return out, (h.view(1, user_len, self.hidden_size), c.view(1, user_len, self.hidden_size))

class STGN(nn.Module):
    ''' The STGN Implementation '''
    
    def __init__(self, hidden_size):
        super(STGN, self).__init__()
        self.hidden_size = hidden_size
        
        # some initialization
        self.mu = 0
        self.sd = 1/(hidden_size**2)
        
        # input weights:
        self.Wi = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Wf = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Wo = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Wc = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Wt1 = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Wd1 = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Wt2 = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Wd2 = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)

        # hidden state weights:
        self.Ui = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Uf = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Uo = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        self.Uc = nn.Parameter(torch.randn(1, hidden_size, hidden_size)*self.sd + self.mu)
        
        # Temporal weights:
        self.Tt1 = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.Tt2 = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.To = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        
        # Spatial weights:
        self.Dd1 = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.Dd2 = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.Do = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)

        # bias terms:
        self.bi = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.bf = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.bo = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)           
        self.bc = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.bt1 = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.bd1 = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.bt2 = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        self.bd2 = nn.Parameter(torch.randn(1, hidden_size, 1)*self.sd + self.mu)
        
    def forward(self, x, delta_t, delta_s, hc):
        seq_len, user_len, _ = x.size()
        h, c = hc
        out = []
        h = h[0].unsqueeze(2)
        c = c[0].unsqueeze(2)
        for j in range(seq_len):
            x_t = x[j].view(user_len, self.hidden_size, 1)
            t_t = delta_t[j].view(user_len, 1, 1)
            s_t = delta_s[j].view(user_len, 1, 1)
            i = torch.sigmoid(torch.matmul(self.Wi, x_t) + torch.matmul(self.Ui, h) + self.bi)
            f = torch.sigmoid(torch.matmul(self.Wf, x_t) + torch.matmul(self.Uf, h) + self.bf)
            o = torch.sigmoid(torch.matmul(self.Wo, x_t) + torch.matmul(self.Uo, h) + t_t*self.To + s_t*self.Do + self.bo)
            c_tilde = torch.tanh(torch.matmul(self.Wc, x_t) + torch.matmul(self.Uc, h) + self.bc)
            T1 = torch.sigmoid(torch.matmul(self.Wt1, x_t) + torch.sigmoid(t_t*self.Tt1) + self.bt1)
            D1 = torch.sigmoid(torch.matmul(self.Wd1, x_t) + torch.sigmoid(s_t*self.Dd1) + self.bd1)
            T2 = torch.sigmoid(torch.matmul(self.Wt2, x_t) + torch.sigmoid(t_t*self.Tt2) + self.bt2)
            D2 = torch.sigmoid(torch.matmul(self.Wd2, x_t) + torch.sigmoid(s_t*self.Dd2) + self.bd2)
            c_hat = f*c + i*T1*D1*c_tilde
            c = f*c + i*T2*D2*c_tilde
            h = o*torch.tanh(c_hat)
            out.append(h.squeeze())
        out = torch.stack(out, dim=0).view(seq_len, user_len, self.hidden_size)
        return out, (h.view(1, user_len, self.hidden_size), c.view(1, user_len, self.hidden_size))
        
        
class STGCN(nn.Module):
    ''' The STGCN Implementation '''
    # TODO
    pass

























        
        
        
        