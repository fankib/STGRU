import torch
from torch.utils.data import Dataset
from enum import Enum

class Split(Enum):
    TRAIN = 0
    TEST = 2
    USE_ALL = 3
    

class PoiDataset(Dataset):
    
    def __init__(self, users, locs, seq_length, split, loc_count):
        self.users = users
        self.locs = locs
        self.labels = []
        self.sequences = []
        self.sequences_labels = []
        self.sequences_count = []
        self.Ps = []
        self.Qs = torch.zeros(loc_count, 1)
        
        # align labels to locations
        for i, loc in enumerate(locs):
            self.locs[i] = loc[:-1]
            self.labels.append(loc[1:])
            
        # collect locations:
        for i in range(loc_count):
            self.Qs[i, 0] = i        
        
        # collect available locations per user
        for i, loc in enumerate(self.locs):
            ps = []
            ls = []            
            for j, l in enumerate(loc):
                if not l in ls:
                    ls.append(l)
                    p = torch.zeros(loc_count).float()
                    p[l] = 1
                    ps.append(p)
            self.Ps.append(torch.stack(ps, dim=0))
        
        # split to training / test phase:
        for i, (loc, label) in enumerate(zip(self.locs, self.labels)):
            train_thr = int(len(loc) * 0.8)
            if (split == Split.TRAIN):
                self.locs[i] = loc[:train_thr]
                self.labels[i] = label[:train_thr]
            if (split == Split.TEST):
                self.locs[i] = loc[train_thr:]
                self.labels[i] = label[train_thr:]
            if (split == Split.USE_ALL):
                pass # do nothing
            
        # split location and labels to sequences:
        self.max_seq_count = 0
        self.min_seq_count = 10000000
        for i, (loc, label) in enumerate(zip(self.locs, self.labels)):
            seq_count = len(loc) // seq_length
            seqs = []
            seq_lbls = []
            for j in range(seq_count):
                start = j * seq_length
                end = (j+1) * seq_length
                seqs.append(loc[start:end])
                seq_lbls.append(label[start:end])
            self.sequences.append(seqs)
            self.sequences_labels.append(seq_lbls)
            self.sequences_count.append(seq_count)
            self.max_seq_count = max(self.max_seq_count, seq_count)
            self.min_seq_count = min(self.min_seq_count, seq_count)
        print('load', len(users), 'users with max_seq_count', self.max_seq_count)
    
    def __len__(self):
        #return self.min_seq_count
        return self.max_seq_count
    
    def __getitem__(self, idx):
        seqs = []
        lbls = []
        reset_h = []
        for i in range(len(self.users)):
            j = idx % self.sequences_count[i]
            reset_h.append(j == 0)
            seqs.append(torch.tensor(self.sequences[i][j]))
            lbls.append(torch.tensor(self.sequences_labels[i][j]))
        return torch.stack(seqs, dim=1), torch.stack(lbls, dim=1), reset_h, self.Ps
        #return torch.tensor(self.locs[idx]), torch.tensor(self.labels[idx])


class GowallaLoader():
    
    def __init__(self, max_users = 0, min_checkins = 0):
        self.max_users = max_users
        self.min_checkins = min_checkins
        
        self.user2id = {}
        self.poi2id = {}
        
        self.users = []
        self.locs = []
    
    def poi_dataset(self, seq_length, split):
        dataset = PoiDataset(self.users, self.locs, seq_length, split, len(self.poi2id)) # crop latest in time
        return dataset
    
    def locations(self):
        return len(self.poi2id)
    
    def load(self, file):
        # collect all users with min checkins:
        self.load_users(file)
        # collect checkins for all collected users:
        self.load_pois(file)
    
    def load_users(self, file):
        f = open(file, 'r')
        lines = f.readlines()
    
        prev_user = int(lines[0].split('\t')[0])
        visit_cnt = 0
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if user == prev_user:
                visit_cnt += 1
            else:
                if visit_cnt >= self.min_checkins:
                    self.user2id[prev_user] = len(self.user2id)
                prev_user = user
                visit_cnt = 1
                if self.max_users > 0 and len(self.user2id) >= self.max_users:
                    break # restrict to max users
    
    def load_pois(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        
        # store location ids
        user_loc = []
        
        prev_user = int(lines[0].split('\t')[0])
        prev_user = self.user2id.get(prev_user)
        for i, line in enumerate(lines):
            tokens = line.strip().split('\t')
            user = int(tokens[0])
            if self.user2id.get(user) is None:
                continue # user is not of interrest
            user = self.user2id.get(user)

            location = int(tokens[4]) # location nr
            if self.poi2id.get(location) is None: # get-or-set locations
                self.poi2id[location] = len(self.poi2id)
            location = self.poi2id.get(location)
    
            if user == prev_user:
                # insert in front!
                #user_loc.insert(0, location)
                user_loc.append(location)
            else:
                self.users.append(prev_user)
                self.locs.append(user_loc)
                prev_user = user
                user_loc = [location] # resart
                
        # process also the latest user in the for loop
        self.users.append(prev_user)
        self.locs.append(user_loc)