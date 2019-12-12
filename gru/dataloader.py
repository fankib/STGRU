import torch
from torch.utils.data import Dataset
from enum import Enum

class Split(Enum):
    TRAIN = 0
    TEST = 2
    USE_ALL = 3

class Usage(Enum):
    MIN_SEQ_LENGTH = 0
    MAX_SEQ_LENGTH = 1
    CUSTOM = 2
    

class PoiDataset(Dataset):
    
    def reset(self):
        # reset training state:
        self.next_user_idx = 0 # current user index to add
        self.active_users = [] # current active users
        self.active_user_seq = [] # current active users sequences
        
        # set active users:
        for i in range(self.user_length):
            self.next_user_idx += 1
            self.active_users.append(i) 
            self.active_user_seq.append(0)
        
    
    def __init__(self, users, locs, seq_length, user_length, split, usage, loc_count, custom_seq_count):
        self.users = users
        self.locs = locs
        self.labels = []
        self.sequences = []
        self.sequences_labels = []
        self.sequences_count = []
        self.Ps = []
        self.Qs = torch.zeros(loc_count, 1)
        self.usage = usage
        self.user_length = user_length
        self.loc_count = loc_count
        self.custom_seq_count = custom_seq_count
        #self.loc_ids_of_user = [] # sets of unique locations per user

        self.reset()

        # collect locations:
        for i in range(loc_count):
            self.Qs[i, 0] = i    
            
        # collect unique locations per user:
        #for i in range(len(users)):
        #    locs_of_user = set()
        #    for j in self.locs[i]:
        #        locs_of_user.add(j)
        #    self.loc_ids_of_user.append(locs_of_user)
        
        # collect available locations per user
        #for i, loc in enumerate(self.locs):
        #    pss = []
        #    ls = []
        #    for j, l in enumerate(loc):
        #        if not l in ls:
        #            ls.append(l)
        #            p = torch.zeros(loc_count).float()
        #            p[l] = 1
        #            pss.append(p)
        #    print('user', i, 'has ', len(pss), 'distinct locations')
        #    self.Ps.append(torch.stack(pss, dim=0))
        
        # align labels to locations
        for i, loc in enumerate(locs):
            self.locs[i] = loc[:-1]
            self.labels.append(loc[1:])
        
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
        self.capacity = 0
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
            self.capacity += seq_count
            self.max_seq_count = max(self.max_seq_count, seq_count)
            self.min_seq_count = min(self.min_seq_count, seq_count)
        
        # statistics
        if (self.usage == Usage.MIN_SEQ_LENGTH):
            print('load', len(users), 'users with min_seq_count', self.min_seq_count, 'batches:', self.__len__())
        if (self.usage == Usage.MAX_SEQ_LENGTH):
            print('load', len(users), 'users with max_seq_count', self.max_seq_count, 'batches:', self.__len__())
        if (self.usage == Usage.CUSTOM):
            print('load', len(users), 'users with custom_seq_count', self.custom_seq_count, 'Batches:', self.__len__())
            
    
    def sequences_by_user(self, idx):
        return self.sequences[idx]
    
    def __len__(self):
        if (self.usage == Usage.MIN_SEQ_LENGTH):
            # min times amount_of_user_batches:
            return self.min_seq_count * (len(self.users) // self.user_length)
        if (self.usage == Usage.MAX_SEQ_LENGTH):
            # estimated capacity:
            estimated = self.capacity // self.user_length
            return max(self.max_seq_count, estimated)
        if (self.usage == Usage.CUSTOM):
            return self.custom_seq_count * (len(self.users) // self.user_length)
        raise Exception('Piiiep')
    
    def __getitem__(self, idx):
        seqs = []
        lbls = []
        reset_h = []
        for i in range(self.user_length):
            i_user = self.active_users[i]
            j = self.active_user_seq[i]
            max_j = self.sequences_count[i_user]
            if (self.usage == Usage.MIN_SEQ_LENGTH):
                max_j = self.min_seq_count
            if (self.usage == Usage.CUSTOM):
                max_j = min(max_j, self.custom_seq_count) # use either the users maxima count or limit by custom count
            if (j >= max_j):
                # repalce this user in current sequence:
                i_user = self.next_user_idx
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                while self.next_user_idx in self.active_users:
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.users)
                # TODO: throw exception if wrapped around!
            # use this user:
            reset_h.append(j == 0)
            seqs.append(torch.tensor(self.sequences[i_user][j]))
            lbls.append(torch.tensor(self.sequences_labels[i_user][j]))
            self.active_user_seq[i] += 1
        
        #if idx % 10 == 0:
        #    print('active on batch ', idx, self.active_users)
        
        # collect active locations:
        '''active_locs = set()
        for i in range(self.user_length):
            i_user = self.active_users[i]
            active_locs.update(self.loc_ids_of_user[i_user])
        P = torch.zeros(len(active_locs), self.loc_count)
        poi2id = torch.zeros(self.loc_count)
        for i, l in enumerate(active_locs):
            P[i, l] = 1
            poi2id[l] = i'''
            
        return torch.stack(seqs, dim=1), torch.stack(lbls, dim=1), reset_h, torch.tensor(self.active_users) #, P, poi2id
        #for i in range(len(self.users)):
        #    j = idx % self.sequences_count[i]
        #    reset_h.append(j == 0)
        #    seqs.append(torch.tensor(self.sequences[i][j]))
        #    lbls.append(torch.tensor(self.sequences_labels[i][j]))
        #return torch.tensor(self.locs[idx]), torch.tensor(self.labels[idx])


class GowallaLoader():
    
    def __init__(self, max_users = 0, min_checkins = 0):
        self.max_users = max_users
        self.min_checkins = min_checkins
        
        self.user2id = {}
        self.poi2id = {}
        
        self.users = []
        self.locs = []
    
    def poi_dataset(self, seq_length, user_length, split, usage, custom_seq_count = 1):
        dataset = PoiDataset(self.users, self.locs, seq_length, user_length, split, usage, len(self.poi2id), custom_seq_count) # crop latest in time
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
                user_loc.insert(0, location)
            else:
                self.users.append(prev_user)
                self.locs.append(user_loc)
                prev_user = user
                user_loc = [location] # resart
                
        # process also the latest user in the for loop
        self.users.append(prev_user)
        self.locs.append(user_loc)