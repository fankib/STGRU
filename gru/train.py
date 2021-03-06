import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import argparse
import time

from dataloader import GowallaLoader, Split, Usage
from torch.utils.data import DataLoader
from trainer import TrainerFactory
from network import GRU, GruFactory
from h_strategy import ZeroStrategy, FixNoiseStrategy, PersistUserStateStrategy, h_strategy_from_string

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
parser.add_argument('--regularization', default=0.0, type=float, help='regularization weight')
parser.add_argument('--lr', default = 0.01, type=float, help='learning rate')
parser.add_argument('--epochs', default=1000, type=int, help='amount of epochs')
parser.add_argument('--cross-entropy', default=False, const=True, nargs='?', type=bool, help='use cross entropy loss instead of BPR loss for training')
parser.add_argument('--skip-sanity', default=False, const=True, nargs='?', type=bool, help='skip sanity tests')
parser.add_argument('--user-embedding', default=False, const=True, nargs='?', type=bool, help='activate user embeddings')
parser.add_argument('--temporal', default=False, const=True, nargs='?', type=bool, help='activate use of temporal data')
parser.add_argument('--spatial', default=False, const=True, nargs='?', type=bool, help='activate use of spatial data')
parser.add_argument('--dataset', default='loc-gowalla_totalCheckins.txt', type=str, help='the dataset under ../../dataset/<dataset.txt> to load')
parser.add_argument('--gru', default='pytorch', type=str, help='the GRU implementation to use: [pytorch|own|rnn|lstm]')
parser.add_argument('--h0', default='fixnoise', type=str, help='h0 strategy to use: [zero|fixnoise|zero-persist|fixnoise-persist], zero: use zero vector, fixnoise: use normal noise, -persist: propagate latest train state to test')
parser.add_argument('--lambda_t', default=1.0, type=float, help='decay factor for temporal data')
parser.add_argument('--lambda_s', default=1.0, type=float, help='decay factor for spatial data')
parser.add_argument('--validate-recall', default=False, const=True, nargs='?', type=bool, help='activate (slower) recall@1,5,10 additional to MAP')
args = parser.parse_args()

###### parameters ######
epochs = args.epochs
lr = args.lr
hidden_size = args.dims
seq_length = args.seq_length
user_count = args.users
user_length = args.user_length
weight_decay = args.regularization
use_cross_entropy = args.cross_entropy
skip_sanity = args.skip_sanity
use_user_embedding = args.user_embedding
use_temporal = args.temporal
use_spatial = args.spatial
dataset_file = '../../dataset/{}'.format(args.dataset)
gru_factory = GruFactory(args.gru)
do_map_only = not args.validate_recall
do_recall = args.validate_recall
########################

### CUDA Setup ###
device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)
print('use', device)

trainer_factory = TrainerFactory()
trainer = trainer_factory.create(use_cross_entropy, use_user_embedding, use_temporal, use_spatial, args.lambda_t, args.lambda_s, gru_factory.is_stgn())
#print('{}'.format(trainer.greeter()))
print('{} {}'.format(trainer.greeter(), gru_factory.greeter()))

gowalla = GowallaLoader(user_count, args.min_checkins)
gowalla.load(dataset_file)
user_count = gowalla.user_count()
print('use users:', user_count)
#DEBUG:
#dataset = gowalla.poi_dataset(seq_length, user_length, Split.TRAIN, Usage.CUSTOM, 3)
#dataset_test = gowalla.poi_dataset(seq_length, user_length, Split.TEST, Usage.CUSTOM, 3)

dataset = gowalla.poi_dataset(seq_length, user_length, Split.TRAIN, Usage.MAX_SEQ_LENGTH)
dataset_test = gowalla.poi_dataset(seq_length, user_length, Split.TEST, Usage.MAX_SEQ_LENGTH)
#dataset_test = gowalla.poi_dataset(seq_length, user_length, Split.TRAIN, Usage.MIN_SEQ_LENGTH) # DEBUG-ONLY! converge on train
dataloader = DataLoader(dataset, batch_size = 1, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size = 1, shuffle=False)

#h0_strategy = PersistUserStateStrategy(hidden_size, user_count, FixNoiseStrategy(hidden_size))
h0_strategy = h_strategy_from_string(args.h0, hidden_size, user_count, gru_factory.is_lstm())

# setup trainer
trainer.prepare(gowalla.locations(), user_count, hidden_size, gru_factory, device)

optimizer = torch.optim.Adam(trainer.parameters(), lr = lr, weight_decay = weight_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.8)

def evaluate_test():
    dataset_test.reset()
    h = h0_strategy.on_init(user_length, device)
    
    with torch.no_grad():        
        iter_cnt = 0
        recall1 = 0
        recall5 = 0
        recall10 = 0
        average_precision = 0.
        
        u_iter_cnt = np.zeros(user_count)
        u_recall1 = np.zeros(user_count)
        u_recall5 = np.zeros(user_count)
        u_recall10 = np.zeros(user_count)
        u_average_precision = np.zeros(user_count)        
        reset_count = torch.zeros(user_count)
        
        for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(dataloader_test):
            active_users = active_users.squeeze()
            for j, reset in enumerate(reset_h):
                if reset:
                    if gru_factory.is_lstm():
                        hc = h0_strategy.on_reset_test(active_users[j], device)
                        h[0][0, j] = hc[0]
                        h[1][0, j] = hc[1]
                    else:
                        h[0, j] = h0_strategy.on_reset_test(active_users[j], device)
                    reset_count[active_users[j]] += 1
            
            #if i % 10 == 0:
            #    print('active on batch', i, active_users)
            
            # squeeze for reasons of "loader-batch-size-is-1"
            x = x.squeeze().to(device)
            t = t.squeeze().to(device)
            s = s.squeeze().to(device)            
            y = y.squeeze()
            y_t = y_t.squeeze().to(device)
            y_s = y_s.squeeze().to(device)
            
            active_users = active_users.to(device)            
        
            # evaluate:
            out, h = trainer.evaluate(x, t, s, y_t, y_s, h, active_users)
            
            for j in range(user_length):  
                # o contains a per user list of votes for all locations for each sequence entry
                o = out[j]
                
                # Only compute MAP is significantly faster as we ommit sorting.                
                if (do_map_only):
                    y_j = y[:, j]
                    
                    for k in range(len(y_j)):                    
                        if (reset_count[active_users[j]] > 1):
                            continue
                        
                        if args.validate_on_latest and (i+1) % seq_length != 0:
                            continue

                        r_kj = o[k, :].cpu().numpy()

                        t = y_j[k]
                        
                        t_val = r_kj[t]
                        upper = np.where(r_kj > t_val)[0]
                        precision = 1. / (1+len(upper))
                        u_iter_cnt[active_users[j]] += 1
                        u_average_precision[active_users[j]] += precision
                    
                
                if (do_recall):
                    # partition elements
                    o_n = o.cpu().detach().numpy()
                    ind = np.argpartition(o_n, -10, axis=1)[:, -10:] # top 10 elements
                                       
                    y_j = y[:, j]
                    
                    for k in range(len(y_j)):                    
                        if (reset_count[active_users[j]] > 1):
                            continue
                        
                        if args.validate_on_latest and (i+1) % seq_length != 0:
                            continue
                        
                        #if Ps[j].size()[0] == 1:
                        #    continue # skip user with single location.
                        
                        # resort indices for k:
                        ind_k = ind[k]
                        r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)] # sort top 10 elements descending
                        
                        # with filtering on seen locations:
                        #r = torch.tensor(PQs[r]) # transform to given locations
                        # w/o filtering on seen locations:
                        r = torch.tensor(r)
                        t = y_j[k]
                        
                        # compute MAP:
                        r_kj = o_n[k, :]
                        t_val = r_kj[t]
                        upper = np.where(r_kj > t_val)[0]
                        precision = 1. / (1+len(upper))
                        
                        # store
                        u_iter_cnt[active_users[j]] += 1
                        u_recall1[active_users[j]] += t in r[:1]
                        u_recall5[active_users[j]] += t in r[:5]
                        u_recall10[active_users[j]] += t in r[:10]
                        u_average_precision[active_users[j]] += precision
        
        formatter = "{0:.8f}"
        for j in range(user_count):
            iter_cnt += u_iter_cnt[j]
            recall1 += u_recall1[j]
            recall5 += u_recall5[j]
            recall10 += u_recall10[j]
            average_precision += u_average_precision[j]

            if (j % args.report_user == 0):
                print('Report user', j, 'preds:', u_iter_cnt[j], 'recall@1', formatter.format(u_recall1[j]/u_iter_cnt[j]), 'MAP', formatter.format(u_average_precision[j]/u_iter_cnt[j]), sep='\t')
            
        if (do_recall):
            print('recall@1:', formatter.format(recall1/iter_cnt))
            print('recall@5:', formatter.format(recall5/iter_cnt))
            print('recall@10:', formatter.format(recall10/iter_cnt))
        print('MAP', formatter.format(average_precision/iter_cnt))
        print('predictions:', iter_cnt)
            
                

# test user idx
sample_user_id = 2
train_seqs = dataset.sequences_by_user(sample_user_id)
test_seqs = dataset_test.sequences_by_user(sample_user_id)
print('~~~ train ~~~', train_seqs)
print('~~~ test ~~~', test_seqs)

# try before train
if not skip_sanity:
    #sample(sample_user_id)
    evaluate_test()

# train!
for e in range(epochs):
    h = h0_strategy.on_init(user_length, device)
    
    dataset.shuffle_users() # shuffle users before each epoch!
    for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(dataloader):
        for j, reset in enumerate(reset_h):
            if reset:
                if gru_factory.is_lstm():
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])
        
        x = x.squeeze().to(device)
        t = t.squeeze().to(device)
        s = s.squeeze().to(device)
        y = y.squeeze().to(device)
        y_t = y_t.squeeze().to(device)
        y_s = y_s.squeeze().to(device)                
        active_users = active_users.to(device)
        
        optimizer.zero_grad()
        loss, h = trainer.loss(x, t, s, y, y_t, y_s, h, active_users)
        loss.backward(retain_graph=True) # backpropagate through time to adjust the weights and find the gradients of the loss function
        latest_loss = loss.item()        
        optimizer.step()
        
        # persist state for test
        h0_strategy.persist_state(h, active_users[0])
    
    # debug info:
    trainer.debug()
    
    # statistics:
    if (e+1) % 1 == 0:
        print(f'Epoch: {e+1}/{epochs}')
        print(f'Loss: {latest_loss}')
    if (e+1) % args.validate_epoch == 0:
        #sample(sample_user_id)
        print('~~~ Test Set Evaluation ~~~')
        evaluate_test()


def visualize_embedding(embedding):
    pass
        




# TODO: fixme!!
def sample(idx):
    assert idx < user_length # does not yet work if we wrap around users!
    dataset_test.reset()
   
    with torch.no_grad(): 
        h = h0_strategy.on_reset_test(idx, device).view(1, 1, hidden_size) # FIXME
        
        resets = 0
        
        for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(dataloader_test):
            if reset_h[idx]:
                resets += 1
            
            if resets > 1:
                return
                           
            x = x.squeeze()[:, idx].to(device)            
            t = t.squeeze()[:, idx].to(device)
            s = s.squeeze()[:, idx].to(device)
            y = y.squeeze()[:, idx].to(device)
            y_t = y_t.squeeze()[:, idx].to(device)
            y_s = y_s.squeeze()[:, idx].to(device)
            active_user = active_users.squeeze()[idx].to(device).view(1,1)
        
            offset = 1
            test_input = x[:offset].view(offset, 1)
            test_t = t[:offset].view(offset, 1)
            test_s = s[:offset].view(offset, 2)
            test_y_t = y_t[:offset].view(offset, 1)
            test_y_s = y_s[:offset].view(offset, 2)
    
            for i in range(seq_length):
                #y_ts, h = model(test_input, h)
                
                out, h = trainer.evaluate(test_input, test_t, test_s, test_y_t, test_y_s, h, active_user)
                
                #if use_cross_entropy:
                #    probs = y_ts[-1].transpose(0, 1)
                #else:
                #    y_last = y_ts[-1].transpose(0,1) # latest
                #    probs = torch.matmul(model.encoder.weight, y_last).cpu().detach().numpy()
                probs = out[0][-1].cpu().detach().numpy()
                rank = np.argsort(np.squeeze(-probs))        
            
                print('in', test_input.item(), 'expected', y[offset+i-1].item(), ': idx-target', np.where(rank == y[offset+i-1].item())[0][0] + 1, 'prediction', rank[:5])
            
                test_input = y[offset+i-1].view(1, 1)
                test_t = t[offset+i-1].view(1, 1)
                test_s = s[offset+i-1].view(1, 2)
                test_y_t = y_t[offset+i-1].view(1, 1)
                test_y_s = y_s[offset+i-1].view(1, 2)