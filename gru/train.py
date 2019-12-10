import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import argparse
import time

from network import RNN
from dataloader import GowallaLoader, Split, Usage
from torch.utils.data import DataLoader

### command line parameters ###
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use')
parser.add_argument('--users', default=6, type=int, help='users to process')
parser.add_argument('--dims', default=7, type=int, help='hidden dimensions to use')
parser.add_argument('--seq-length', default=10, type=int, help='seq-length to process in one pass (batching)')
parser.add_argument('--user-length', default=3, type=int, help='user-length to process in one pass (batching)')
parser.add_argument('--min-checkins', default=101, type=int, help='amount of checkins required')
parser.add_argument('--validate-on-latest', default=False, const=True, nargs='?', type=bool, help='use only latest sequence sample to validate')
parser.add_argument('--validate-epoch', default=3, type=int, help='run validation after this amount of epochs')
parser.add_argument('--report-user', default=1, type=int, help='report every x user on evaluation')
args = parser.parse_args()

###### parameters ######
epochs = 10000
lr = 0.01
hidden_size = args.dims
seq_length = args.seq_length
user_count = args.users
user_length = args.user_length
########################

### CUDA Setup ###
device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)
print('use', device)

gowalla = GowallaLoader(user_count, args.min_checkins)
#gowalla.load('../../dataset/small-10000.txt')
gowalla.load('../../dataset/loc-gowalla_totalCheckins.txt')
#gowalla.load('../../dataset/loc-gowalla_totalCheckins_Pcore50_50.txt')
dataset = gowalla.poi_dataset(seq_length, user_length, Split.TRAIN, Usage.MAX_SEQ_LENGTH)
dataset_test = gowalla.poi_dataset(seq_length, user_length, Split.TEST, Usage.MAX_SEQ_LENGTH)
dataloader = DataLoader(dataset, batch_size = 1, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size = 1, shuffle=False)

model = RNN(gowalla.locations(), hidden_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.MSELoss()

def evaluate(dataloader):
    dataset_test.reset()
    h = torch.zeros(1, user_length, hidden_size).to(device)
    
    with torch.no_grad():        
        iter_cnt = 0
        recall1 = 0
        recall5 = 0
        recall10 = 0
        average_precision = 0.
        
        u_iter_cnt = np.zeros(args.users)
        u_recall1 = np.zeros(args.users)
        u_recall5 = np.zeros(args.users)
        u_recall10 = np.zeros(args.users)
        u_average_precision = np.zeros(args.users)        
        reset_count = torch.zeros(user_count)
        
        for i, (x, y, reset_h, active_users) in enumerate(dataloader):
            active_users = active_users.squeeze()
            for j, reset in enumerate(reset_h):
                if reset:
                    h[0, j] = torch.zeros(hidden_size)
                    reset_count[active_users[j]] += 1
            
            if i % 10 == 0:
                print('active on batch', i, active_users)
            
            # for user location selections:
            #Ps = dataset.Ps
            #Qs = dataset.Qs
            
            # squeeze for reasons of "loader-batch-size-is-1"
            x = x.squeeze().to(device)
            y = y.squeeze()
        
            out, h = model(x, h)
            
            out_t = out.transpose(0, 1)
            Q = model.encoder.weight
            
            for j in range(user_length):                
                out_j = out_t[j].transpose(0,1)
                
                # with filtering on seen locations:
                #PQ = torch.matmul(Ps[j].to(device), Q)
                #PQs = torch.matmul(Ps[j], Qs).squeeze().long().numpy()
                
                # w/o filtering on seen locations:
                PQ = Q

                o = torch.matmul(PQ, out_j).cpu().detach()
                o = o.transpose(0,1)
                o = o.contiguous().view(10, -1)
                #start = time.time()
                rank = np.argsort(-1*o.numpy(), axis=1)
                #duration = time.time() - start
                #print('argsort for', active_users[j], 'in', duration)
                
                y_j = y[:, j]
                
                for k in range(len(y_j)):                    
                    if (reset_count[active_users[j]] > 1):
                        continue
                    
                    if args.validate_on_latest and (i+1) % seq_length != 0:
                        continue
                    
                    #if Ps[j].size()[0] == 1:
                    #    continue # skip user with single location.
                    
                    r = rank[k, :]
                    # with filtering on seen locations:
                    #r = torch.tensor(PQs[r]) # transform to given locations
                    # w/o filtering on seen locations:
                    r = torch.tensor(r)
                    t = y_j[k]
                    
                    if not t in r:
                        print('we have a problem with user', active_users[j], ': t is', t, 'rank is', r)
                    
                    #if (j == 1):
                    #    print('at user 1, t:', t,'r[0]:', r[0])
                    
                    u_iter_cnt[active_users[j]] += 1
                    u_recall1[active_users[j]] += t in r[:1]
                    u_recall5[active_users[j]] += t in r[:5]
                    u_recall10[active_users[j]] += t in r[:10]
                    idx_target = np.where(r == t)[0][0]
                    precision = 1./(idx_target+1)
                    u_average_precision[active_users[j]] += precision
        
        formatter = "{0:.2f}"
        for j in range(args.users):
            if (j % args.report_user > 0):
                continue
            iter_cnt += u_iter_cnt[j]
            recall1 += u_recall1[j]
            recall5 += u_recall5[j]
            recall10 += u_recall10[j]
            average_precision += u_average_precision[j]
            #print('Report user', j, 'recall@1', u_recall1[j]/u_iter_cnt[j], 'recall@5', u_recall5[j]/u_iter_cnt[j], 'recall@10', u_recall10[j]/u_iter_cnt[j], 'MAP', u_average_precision[j]/u_iter_cnt[j], sep='\t')
            print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1', formatter.format(u_recall1[j]/u_iter_cnt[j]), 'MAP', formatter.format(u_average_precision[j]/u_iter_cnt[j]), sep='\t')
            
            
        print('recall@1:', formatter.format(recall1/iter_cnt))
        print('recall@5:', formatter.format(recall5/iter_cnt))
        print('recall@10:', formatter.format(recall10/iter_cnt))
        print('MAP', formatter.format(average_precision/iter_cnt))
        print('predictions:', iter_cnt)
            

def sample(idx):
    assert idx < user_length # does not yet work if we wrap around users!
    dataset_test.reset()
   
    with torch.no_grad(): 
        h = torch.zeros(1, 1, hidden_size).to(device)
        
        resets = 0
        
        for i, (x, y, reset_h, _) in enumerate(dataloader_test):
            if reset_h[idx]:
                resets += 1
            
            if resets > 1:
                return
                           
            x = x.squeeze()[:, idx].to(device)
            y = y.squeeze()[:, idx].to(device)
        
            offset = 1
            test_input = x[:offset].view(offset, 1)
    
            for i in range(10):
                y_ts, h = model(test_input, h)
                y_last = y_ts[-1].transpose(0,1) # latest
            
                probs = torch.matmul(model.encoder.weight, y_last).cpu().detach().numpy()
                rank = np.argsort(np.squeeze(-probs))        
            
                print('in', test_input.item(), 'expected', y[offset+i-1].item(), ': idx-target', np.where(rank == y[offset+i-1].item())[0][0] + 1, 'prediction', rank[:5])
            
                test_input = y[offset+i-1].view(1, 1)

# test user idx
sample_user_id = 2
train_seqs = dataset.sequences_by_user(sample_user_id)
test_seqs = dataset_test.sequences_by_user(sample_user_id)
print('~~~ train ~~~', train_seqs)
print('~~~ test ~~~', test_seqs)

# try before train
sample(sample_user_id)
evaluate(dataloader_test)

# train!
for e in range(epochs):
    h = torch.zeros(1, user_length, hidden_size).to(device)
    
    for i, (x, y, reset_h, active_users) in enumerate(dataloader):
        for j, reset in enumerate(reset_h):
            if reset:
                h[0, j] = torch.zeros(hidden_size)
        
        if i % 100 == 0:
            print('active on batch', i, active_users[0])
        
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
    if (e+1) % args.validate_epoch == 0:
        sample(sample_user_id)
        #print('~~~ Training Evaluation ~~~')
        #evaluate(dataloader)
        print('~~~ Test Set Evaluation ~~~')
        evaluate(dataloader_test)


def visualize_embedding(embedding):
    pass
        



