import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import argparse

from network import RNN
from dataloader import GowallaLoader, Split
from torch.utils.data import DataLoader

### command line parameters ###
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use')
parser.add_argument('--users', default=10, type=int, help='users to process')
parser.add_argument('--dims', default=10, type=int, help='hidden dimensions to use')
parser.add_argument('--seq_length', default=10, type=int, help='seq-length to process in one pass')
args = parser.parse_args()

###### parameters ######
epochs = 10000
lr = 0.01
hidden_size = args.dims
seq_length = args.seq_length
user_count = args.users
########################

### CUDA Setup ###
device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)
print('use', device)

gowalla = GowallaLoader(user_count, 101)
#gowalla.load('../../dataset/small-10000.txt')
gowalla.load('../../dataset/loc-gowalla_totalCheckins.txt')
dataset = gowalla.poi_dataset(seq_length, Split.TRAIN)
dataset_test = gowalla.poi_dataset(seq_length, Split.TEST)
dataloader = DataLoader(dataset, batch_size = 1, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size = 1, shuffle=False)

model = RNN(gowalla.locations(), hidden_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.MSELoss()

def evaluate(dataloader):
    h = torch.zeros(1, user_count, hidden_size).to(device)
    
    with torch.no_grad():        
        iter_cnt = 0
        recall1 = 0
        recall5 = 0
        recall10 = 0
        average_precision = 0.
        
        reset_count = torch.zeros(user_count)        
        
        for i, (x, y, reset_h) in enumerate(dataloader):
            for j, reset in enumerate(reset_h):
                if reset:
                    reset_count[j] += 1
            
            ####
            #### Attention: the modulo stuff lets us evaluate certain things much more often!!
            ####
            
            # squeeze for reasons of "loader-batch-size-is-1"
            x = x.squeeze().to(device)
            y = y.squeeze()
        
            out, h = model(x, h)
        
            # reshape
            out = out.view(-1, hidden_size)
            out_t = out.transpose(0, 1)
            y = y.contiguous().view(-1)
            Q = model.encoder.weight
            o = torch.matmul(Q, out_t).cpu().detach().numpy()
            rank = np.argsort(-1*o, axis=0)
            
            for i in range(len(y)):
                user = i // seq_length
                if (reset_count[user] > 1):
                    continue
                
                r = torch.tensor(rank[:, i])
                t = y[i]
                
                iter_cnt += 1
                recall1 += t in r[:1]
                recall5 += t in r[:5]
                recall10 += t in r[:10]
                idx_target = np.where(r == t)[0][0]
                precision = 1./(idx_target+1)
                average_precision += precision
            
        print('recall@1:', recall1/iter_cnt)
        print('recall@5:', recall5/iter_cnt)
        print('recall@10:', recall10/iter_cnt)
        print('MAP', average_precision/iter_cnt)
        print('predictions:', iter_cnt)
            

def sample(idx, steps):
   
    with torch.no_grad(): 
        h = torch.zeros(1, 1, hidden_size).to(device)
        x, y, _ = dataset_test.__getitem__(idx)
        x = x[:, 0].to(device)
        y = y[:, 0].to(device)
        
        offset = 5
        test_input = x[:offset].view(offset, 1)
    
        for i in range(steps):
            y_ts, h = model(test_input, h)
            y_last = y_ts[-1].transpose(0,1) # latest
            
            probs = torch.matmul(model.encoder.weight, y_last).cpu().detach().numpy()
            rank = np.argsort(np.squeeze(-probs))        
            
            print('truth', y[offset+i].item(), 'idx-target', np.where(rank == y[offset+i].item())[0][0] + 1, 'prediction', rank[:5])
            
            test_input = y[offset+i].view(1, 1)

# try before train
evaluate(dataloader)
sample(0, 5)

# train!
for e in range(epochs):
    h = torch.zeros(1, user_count, hidden_size).to(device)
    
    for i, (x, y, reset_h) in enumerate(dataloader):
        for j, reset in enumerate(reset_h):
            if reset:
                h[0, j] = torch.zeros(hidden_size)
        
        # reshape
        # sequence already in front!
        #x = x.transpose(1, 0).contiguous()
        #y = y.transpose(1, 0).contiguous()
        #x = x.view(100, batch_size)
        x = x.squeeze().to(device)
        y = y.squeeze().to(device)
        
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
    if (e+1) % 5 == 0:
        sample(0, 5)
        print('~~~ Training Evaluation ~~~')
        evaluate(dataloader)
        print('~~~ Test Set Evaluation ~~~')
        evaluate(dataloader_test)


def visualize_embedding(embedding):
    pass
        



