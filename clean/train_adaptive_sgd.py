import torch
from torch.utils.data import DataLoader
import numpy as np

from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from network import create_h0_strategy
from evaluation import Evaluation

### wandb LOGGER ###
import wandb
import getpass

def local_machine():
    return getpass.getuser() == 'fsb1'

class Logger():
    
    def __init__(self, project, log_wandb, group='', print_frequency = 100):
        self.project = project     
        self.log_frequency = 5
        self.print_frequency = print_frequency
        self.current_step = 0
        self.train_losses = []
        self.log_wandb = log_wandb
        self.group = group
        if log_wandb:
            wandb.init(project=project, group=group)
    
    def log_args(self, args):
        if self.log_wandb:
            wandb.config.update(args)
        print('~~~ args ~~~')
        args_d = args.__dict__
        for k in args_d.keys():
            print('{}: {}'.format(k, args_d[k]))
    
    def step(self):
        self.current_step += 1
    
    def log(self, ord_dict):
        if self.log_wandb:
            wandb.log(ord_dict, step=self.current_step)
        else:
            print(ord_dict)
    
    def log_iteration(self, iteration):
        return (iteration+1) % self.print_frequency == 0
    
    def is_current_step(self, frequency):
        return (self.current_step) % frequency == 0
    
    def is_current_step_in_range(self, frequency, max_value):
        return (self.current_step) % frequency < max_value
    
    # custom adaptive sgd:
    def adaptive_learning_sgd(self, learning_rate, momentum):
        self.log({'learning_rate': learning_rate,\
                  'momentum': momentum})
    
    def adaptive_train_loss(self, loss):
        self.log({'train_loss': loss})
    
    def adaptive_test_energy(self, loss):
        self.log({'test_energy': loss})
        
    def adaptive_weight_decay(self, weight_decay):
        self.log({'regularization': weight_decay})
            
    # custom metrics:    
    def train_loss(self, epoch, iteration, loss):        
        self.train_losses.append(loss)
        if self.log_wandb and (iteration+1) % self.log_frequency == 0:
            wandb.log({'train_loss': loss}, step=self.current_step)
        if (iteration+1) % self.print_frequency == 0:        
            print('Epoch {}, iter {}: loss {:0.3f}'.format(epoch, iteration+1, loss))
    
    def epoch_end(self, epoch):  
        if self.log_wandb:
            wandb.log({'zz_epoch_loss_mean': np.mean(self.train_losses), 'zz_epoch_loss_std': np.std(self.train_losses)}, step=self.current_step)
        print('Epoch {} done: avg loss: {:0.3f} ({:0.3f})'.format(epoch, np.mean(self.train_losses), np.std(self.train_losses)))
        self.train_losses = []
        
    def evaluate_train(self, epoch, total, correct, avg_loss=0.):
        self.evaluate('train', epoch, total, correct, avg_loss)
    
    def evaluate_test(self, epoch, total, correct, avg_loss=0.):
        self.evaluate('test', epoch, total, correct, avg_loss)
        
    def evaluate(self, prefix, epoch, total, correct, avg_loss):
        acc = correct/total
        if self.log_wandb:
            wandb.log({'{}_accuracy'.format(prefix): acc,\
                       '{}_avg_loss'.format(prefix): avg_loss,\
                   'zz_{}_total'.format(prefix): total,\
                   'zz_{}_correct'.format(prefix): correct}, step=self.current_step)
        print('Epoch {} {} accuracy: {}/{}={:0.3f}'.format(epoch, prefix, correct, total, acc))

# Init logger
print_frequency = 100 if not local_machine() else 10
logger = Logger('stgru-adaptive-sgd', not local_machine(), group='p10_10', print_frequency=print_frequency)

### Adaptive SGD ###

# settings:
inner_iters = 5

class Buffer():
    partial_A_lambda = 'partial_A_lambda_buffer'
    partial_M_lambda = 'partial_M_lambda_buffer'
    partial_A_alpha = 'partial_A_alpha_buffer'
    partial_A_beta = 'partial_A_beta_buffer'
    partial_M_beta = 'partial_M_beta_buffer'
    weight_decay = 'weight_decay_buffer'
    learning_rate = 'lr_buffer'
    momentum = 'momentum_buffer_hyperparam'
    
class HyperOptimizer(torch.optim.Optimizer):
    ''' controls lambda (weight decay) and learning rate
        using an adaptive heuristic
    '''    
    
    def __init__(self, params, defaults):
        self.inner_step = 0
        super(HyperOptimizer, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(HyperOptimizer, self).__setstate__(state)                    

    def hyper_zero_grad(self):
        pass   
        # ev make partial buffers zero?             
    
    def hyper_momentum(self, group, name, step, beta=0.9):
        if name not in group:
            group[name] = torch.tensor(0.0)
        buf = group[name]
        buf.mul_(beta).add_(1.0, step)
        return buf
    
    def wd_buffer(self, group):
        if Buffer.weight_decay not in group:
            value = torch.tensor(group['weight_decay'])            
            wd_buf = group[Buffer.weight_decay] = value.log()
        else:
            wd_buf = group[Buffer.weight_decay]
        return wd_buf
    
    def get_wd(self, group):
        wd_buf = self.wd_buffer(group)
        return wd_buf.exp().item()
    
    def lr_buffer(self, group):
        if Buffer.learning_rate not in group:
            lr = group['lr']
            lr_buf = group[Buffer.learning_rate] = torch.log(torch.tensor(lr))
        else:
            lr_buf = group[Buffer.learning_rate]
        return lr_buf
    
    def get_lr(self, group):
        lr_buf = self.lr_buffer(group)
        return lr_buf.exp().item()
    
    def momentum_buffer(self, group):
        if Buffer.momentum not in group:
            value = torch.tensor(group['momentum'])
            value = (value / (1-value)).log()
            momentum_buf = group[Buffer.momentum] = value
        else:
            momentum_buf = group[Buffer.momentum]
        return momentum_buf
    
    def get_momentum(self, group):
        momentum_buf = self.momentum_buffer(group)
        return momentum_buf.sigmoid()        
    
    def register_param(self, beta, param_state, param):
        # accumulate hyper gradients (simple approximation)        
        if Buffer.partial_A_lambda not in param_state:
            partial_A_lambda = param_state[Buffer.partial_A_lambda] = torch.zeros_like(torch.flatten(param))
            partial_M_lambda = param_state[Buffer.partial_M_lambda] = torch.zeros_like(torch.flatten(param))
        else:
            partial_A_lambda = param_state[Buffer.partial_A_lambda]                                
            partial_M_lambda = param_state[Buffer.partial_M_lambda]
        partial_M_lambda.mul_(beta)
        partial_M_lambda.add_(1., torch.clone(torch.flatten(param)).detach())        
        partial_A_lambda.add_(-1., partial_M_lambda.detach()) # do not scale by lr
    
    def register_step(self, param_state, step):
        if Buffer.partial_A_alpha not in param_state:
            partial_A_alpha = param_state[Buffer.partial_A_alpha] = torch.zeros_like(torch.flatten(step))
        else:
            partial_A_alpha = param_state[Buffer.partial_A_alpha]
        partial_A_alpha.add_(-1., torch.clone(torch.flatten(step)).detach())  
    
    def register_momentum(self, beta, param_state, momentum, gradient):
        if Buffer.partial_A_beta not in param_state:
            partial_A_beta = param_state[Buffer.partial_A_beta] = torch.zeros_like(torch.flatten(momentum))
            partial_M_beta = param_state[Buffer.partial_M_beta] = torch.zeros_like(torch.flatten(momentum))
        else:
            partial_A_beta = param_state[Buffer.partial_A_beta]
            partial_M_beta = param_state[Buffer.partial_M_beta]
        partial_M_beta.mul_(beta)        
        partial_M_beta.add_(1.0, torch.clone(torch.flatten(momentum - gradient)).detach())
        partial_A_beta.add_(-1., partial_M_beta.detach())
    
    def hyper_step(self):                       
        
        for group in self.param_groups:            
            
            use_wd = Buffer.weight_decay in group
            use_lr = Buffer.learning_rate in group
            use_momentum = Buffer.momentum in group
            
            lambda_grads = []
            alpha_grads = []
            beta_grads = []
            lambda_grad = 0
            alpha_grad = 0
            beta_grad = 0
            
            for p in group['params']:                            
                if p.grad is None:
                    continue
                d_p = p.grad.data

                # partial lambda:
                param_state = self.state[p]
                if use_wd:
                    partial_A_lambda = param_state[Buffer.partial_A_lambda]
                    lambda_grads.append((partial_A_lambda*torch.flatten(d_p)).sum())
                    param_state[Buffer.partial_A_lambda] = torch.zeros_like(torch.flatten(p))
                    param_state[Buffer.partial_M_lambda] = torch.zeros_like(torch.flatten(p))
                
                # partial alpha:                
                if use_lr:
                    partial_A_alpha = param_state[Buffer.partial_A_alpha]
                    alpha_grads.append((partial_A_alpha*torch.flatten(d_p)).sum())
                    param_state[Buffer.partial_A_alpha] = torch.zeros_like(torch.flatten(p))     
                
                # partial beta:
                if use_momentum:
                    partial_A_beta = param_state[Buffer.partial_A_beta]
                    beta_grads.append((partial_A_beta*torch.flatten(d_p)).sum())
                    param_state[Buffer.partial_A_beta] = torch.zeros_like(torch.flatten(p))                               
                    param_state[Buffer.partial_M_beta] = torch.zeros_like(torch.flatten(p))                               
                    
            
            if use_wd:
                wd_buf = group[Buffer.weight_decay]
                lambda_grad = torch.tensor(lambda_grads).sum()                
                
                #stgru
                wd_buf.add_(-0.2/inner_iters, self.hyper_momentum(group, 'lambda_grad_momentum', lambda_grad)) # more lineary
                logger.adaptive_weight_decay(self.get_wd(group))
            
            if use_lr:
                lr_buf = group[Buffer.learning_rate]                
                alpha_grad = torch.tensor(alpha_grads).sum()     
                # stgru:
                lr_buf.add_(-0.1/1, self.hyper_momentum(group, 'apha_grad_momentum', alpha_grad))
            
            if use_momentum:
                momentum_buf = group[Buffer.momentum]
                beta_grad = torch.tensor(beta_grads).sum()                
                
                #stgru:
                momentum_buf.add_(-0.1/1, self.hyper_momentum(group, 'beta_grad_momentum', beta_grad))
            
            if use_lr and use_momentum:
                logger.adaptive_learning_sgd(lr_buf.exp().item(), momentum_buf.sigmoid().item())
            
            logger.log({
                    'lambda_grad': lambda_grad,\
                    'alpha_grad': alpha_grad,\
                    'beta_grad': beta_grad})
            
            #wd_grad = torch.cat(wd_grads).sum()
            #wd_buf.add_(-group['hlr'], wd_grad)
            
            
            #p_dot_Nabla_E_norm = p_dot_Nabla_E / (np.sqrt(l2_p) + np.sqrt(l2_Nabla_E))
            #print('dot-product:', p_dot_Nabla_E, 'angle:', np.degrees(np.arccos(p_dot_Nabla_E_norm)))
            
            #if p_dot_Nabla_E < 0:
            #    lambda_step = -0.01
            #else:
            #    lambda_step = 0.01            
            #wd_buf.add_(1.0, self.hyper_momentum(group, 'lambda_grad_momentum', lambda_step))            
            #wd_buf.add_(-1000.0, p_dot_Nabla_E_norm*weight_decay)
    

class AdaptiveSGD(HyperOptimizer):

    def __init__(self, params, lr=0.0, hlr=0.0, momentum=0, weight_decay=0):
        if lr == 0.0 or lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if hlr == 0.0 or hlr < 0.0:
            raise ValueError("Invalid hyper learning rate: {}".format(hlr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, hlr=hlr, momentum=momentum, weight_decay=weight_decay)
        super(AdaptiveSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:            
            # use tuned wd and lr:            
            weight_decay = self.get_wd(group)
            lr = self.get_lr(group)    
            momentum = self.get_momentum(group)

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data                
                
                param_state = self.state[p]                                
                             
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)                    
                
                # accumulate partial lambda                
                self.register_param(momentum, param_state, p) # before update!                
                                
                # add momentum
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']                                
                self.register_momentum(momentum, param_state, buf, d_p) # accumulate partial beta (use d_p instead of 0 at first..)                
                buf.mul_(momentum).add_(d_p)
                d_p = buf
                p.data.add_(-lr, d_p)
                
                # accumulate partial alpha:
                self.register_step(param_state, d_p)
                
                #if 'partial_M_lambda' not in param_state:
                #    partial_M_lambda = param_state['partial_M_lambda_buffer'] = torch.zeros_like(p)
                #else:
                #    partial_M_lambda = param_state['partial_M_lambda_buffer']                
                
                #partial_M_lambda.mul_(momentum).add_(torch.clone(p).detach()).add_(weight_decay, partial_A_lambda)
                #partial_A_lambda.add_(-lr, partial_M_lambda)

        return loss

### parse settings and create trainer ###
setting = Setting()
setting.parse()
print(setting)

### load dataset ###
poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)
poi_loader.read(setting.dataset_file)
dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
assert setting.batch_size < poi_loader.user_count(), 'batch size must be lower than the amount of available users'

### create flashback trainer ###
trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s)
print('{} {}'.format(trainer, setting.rnn_factory))
h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory, setting.device)
evaluation_test = Evaluation(dataset_test, dataloader_test, poi_loader.user_count(), h0_strategy, trainer, setting)

###  training loop ###
optimizer = AdaptiveSGD(trainer.parameters(), lr=setting.learning_rate, hlr=999, momentum=0.8, weight_decay=setting.weight_decay)

dataset_test.shuffle_users()
test_iterator = iter(dataloader_test)
test_available = len(dataloader_test)

for e in range(setting.epochs):
    h = h0_strategy.on_init(setting.batch_size, setting.device)    
    t_h = h0_strategy.on_init(setting.batch_size, setting.device) 
    dataset.shuffle_users() # shuffle users before each epoch!
    
    losses = []
    
    for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(dataloader):
        
        # logger update
        logger.step()
        
        # reset hidden states for newly added users
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])
        
        x = x.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)
        y = y.squeeze().to(setting.device)
        y_t = y_t.squeeze().to(setting.device)
        y_s = y_s.squeeze().to(setting.device)                
        active_users = active_users.to(setting.device)
        
        optimizer.zero_grad()
        loss, h = trainer.loss(x, t, s, y, y_t, y_s, h, active_users)
        loss.backward(retain_graph=True) # backpropagate through time to adjust the weights and find the gradients of the loss function
        losses.append(loss.item())
        logger.adaptive_train_loss(loss.item())
        optimizer.step()    
        
        # add hyper step after 5 iterations:
        if (i+1) % inner_iters == 0:
            
            # refresh test_loader
            if test_available == 0:
                dataset_test.shuffle_users()
                test_iterator = iter(dataloader_test)
                test_available = len(dataloader_test)
            test_available -= 1
            
            # compute hypergradients:            
            (t_x, t_t, t_s, t_y, t_y_t, t_y_s, t_reset_h, t_active_users) = next(test_iterator)
            # reset hidden states for newly added users
            for j, reset in enumerate(t_reset_h):
                if reset:
                    if setting.is_lstm:
                        hc = h0_strategy.on_reset(t_active_users[0][j])
                        t_h[0][0, j] = hc[0]
                        t_h[1][0, j] = hc[1]
                    else:
                        t_h[0, j] = h0_strategy.on_reset(t_active_users[0][j])
            
            t_x = t_x.squeeze().to(setting.device)
            t_t = t_t.squeeze().to(setting.device)
            t_s = t_s.squeeze().to(setting.device)
            t_y = t_y.squeeze().to(setting.device)
            t_y_t = t_y_t.squeeze().to(setting.device)
            t_y_s = t_y_s.squeeze().to(setting.device)                
            t_active_users = active_users.to(setting.device)
            
            optimizer.hyper_zero_grad()
            optimizer.zero_grad()
            E, t_h = trainer.loss(t_x, t_t, t_s, t_y, t_y_t, t_y_s, t_h, t_active_users)
            E.backward(retain_graph = True)
            optimizer.hyper_step()
            logger.adaptive_test_energy(E.item())              
        
    
    # statistics:
    if (e+1) % 1 == 0:
        epoch_loss = np.mean(losses)
        print(f'Epoch: {e+1}/{setting.epochs}')
        print(f'Avg Loss: {epoch_loss}')
    if (e+1) % setting.validate_epoch == 0:
        #sample(sample_user_id)
        print(f'~~~ Test Set Evaluation (Epoch: {e+1}) ~~~')
        evaluation_test.evaluate()
