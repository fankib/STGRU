
import torch

''' H0 strategies are to handle the special state h0:
    
    - use 0 vector for initialization
    - use gaussian noise vector for initialization
    - persist state per user in order to propagate from train to test
    - reset state after each sequence
'''

def h_strategy_from_string(type_name, hidden_size, user_count, is_lstm):
    if is_lstm:
        if type_name == 'zero':
            return LstmStrategy(hidden_size, ZeroStrategy(hidden_size), ZeroStrategy(hidden_size))
        if type_name == 'fixnoise':
            return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
        raise ValueError('{} not defined for lstm usage!')
    else:
        if type_name == 'zero':
            return ZeroStrategy(hidden_size)
        if type_name == 'fixnoise':
            return FixNoiseStrategy(hidden_size)
        if type_name == 'zero-persist':
            return PersistUserStateStrategy(hidden_size, user_count, ZeroStrategy(hidden_size))
        if type_name == 'fixnoise-persist':
            return PersistUserStateStrategy(hidden_size, user_count, FixNoiseStrategy(hidden_size))
        raise ValueError('{} not defined'.format(type_name))

class H0Strategy():
    
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
    
    def on_init(self, user_len, device):
        pass
    
    def on_reset(self, user):
        pass
    
    def on_reset_test(self, user, device):
        return self.on_reset(user)
    
    def persist_state(self, h, active_users):
        pass
    
    def on_sequence(self, user):
        pass

class ZeroStrategy(H0Strategy):
    ''' uses the 0 vector for initialization and reset '''
    
    def on_init(self, user_len, device):
        return torch.zeros(1, user_len, self.hidden_size, requires_grad=False).to(device)
    
    def on_reset(self, user):
        return torch.zeros(self.hidden_size, requires_grad=False)
    
class FixNoiseStrategy(H0Strategy):
    ''' use fixed normal noise as initialization '''
    
    def __init__(self, hidden_size):
        super(FixNoiseStrategy, self).__init__(hidden_size)
        mu = 0
        sd = 1/self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu
    
    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)
    
    def on_reset(self, user):
        return self.h0

class LstmStrategy(H0Strategy):
    ''' creates h0 and c0 using the inner strategy '''
    
    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy
    
    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return (h,c)
    
    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return (h,c)

class PersistUserStateStrategy(H0Strategy):
    ''' persists the state per user to propagate to test evaluation '''
    
    def __init__(self, hidden_size, user_count, inner_strategy):
        super(PersistUserStateStrategy, self).__init__(hidden_size)
        self.inner_strategy = inner_strategy
        self.user_count = user_count
        
        # init state
        self.user_state = []
        for i in range(user_count):
            self.user_state.append(self.inner_strategy.on_reset(i))
        
    
    def on_init(self, user_len, device):
        return self.inner_strategy.on_init(user_len, device)
    
    def on_reset(self, user):
        return self.inner_strategy.on_reset(user)
    
    def on_reset_test(self, user, device):
        return self.user_state[user].to(device)
    
    def persist_state(self, h, active_users):
        for j in range(len(active_users)):
            self.user_state[active_users[j]] = h[0, j].detach().cpu()
    
        
        
        
        