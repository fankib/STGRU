import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from network import RNN
from dataloader import GowallaLoader
from torch.utils.data import DataLoader

###### parameters ######
epochs = 10000
lr = 0.01
hidden_size = 7
#batch_size = 10
seq_length = 10
########################

gowalla = GowallaLoader(10, 101)
gowalla.load('../../dataset/small-10000.txt')
dataset = gowalla.poi_dataset(seq_length)
dataloader = DataLoader(dataset, batch_size = 1, shuffle=False)

model = RNN(gowalla.locations(), hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.MSELoss()

def sample(idx, steps):
    h = torch.zeros(1, 1, hidden_size)
    
    x, y, _ = dataset.__getitem__(idx)
    x = x[:, 0]
    y = y[:, 0]
    
    offset = 5
    test_input = x[:offset].view(offset, 1)

    for i in range(steps):
        y_ts, h = model(test_input, h)
        y_last = y_ts[-1].transpose(0,1) # latest
        
        probs = torch.matmul(model.encoder.weight, y_last).detach().numpy()
        rank = np.argsort(np.squeeze(-probs))        
        
        print('truth', y[offset+i].item(), 'idx-target', np.where(rank == y[offset+i].item())[0][0] + 1, 'prediction', rank[:5])
        
        test_input = y[offset+i].view(1, 1)

sample(0, 5)

for e in range(epochs):
    h = torch.zeros(1, 10, hidden_size)
    
    for i, (x, y, reset_h) in enumerate(dataloader):
        for j, reset in enumerate(reset_h):
            if reset:
                h[0, j] = torch.zeros(hidden_size)
        
        # reshape
        # sequence already in front!
        #x = x.transpose(1, 0).contiguous()
        #y = y.transpose(1, 0).contiguous()
        #x = x.view(100, batch_size)
        x = x.squeeze()
        y = y.squeeze()
        
        optimizer.zero_grad() # zero out gradients 
        
        out, h = model(x, h)
        #out = out.transpose(0, 1)
        y_emb = model.encoder(y)
        
        # reshape
        out = out.view(-1, hidden_size)
        out_t = out.transpose(0, 1)
        y_emb = y_emb.contiguous().view(-1, hidden_size)
        Q = model.encoder.weight
        
        neg_o = torch.matmul(Q, out_t)
        pos_o = torch.matmul(y_emb, out_t).diag()
        
        loss = torch.log(1 + torch.exp(-(pos_o - neg_o)))
        loss = torch.mean(loss)

        #loss = criterion(out, y_emb)
        loss.backward(retain_graph=True) # backpropagate through time to adjust the weights and find the gradients of the loss function
        
        latest_loss = loss.item()
        
        optimizer.step()
    if (e+1) % 1 == 0:
        print(f'Epoch: {e+1}/{epochs}')
        print(f'Loss: {latest_loss}')
    if (e+1) % 3 == 0:
        sample(0, 5)


def visualize_embedding(embedding):
    pass
        



