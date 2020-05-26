
class Logger:
    
    def __init__(self, log_frequency):
        self.iteration = 0
        self.log_frequency = log_frequency
    
    def step(self):
        self.iteration += 1
    
    def prefix(self):
        return '[Logger] iteration: {}'.format(self.iteration)
    
    def train_loss(self, loss):
        if (self.iteration+1) % self.log_frequency == 0:
            print(self.prefix(), 'train loss: {}'.format(loss))
    
    def evaluate_map(self, map_value):
        if (self.iteration+1) % self.log_frequency == 0:
            print(self.prefix(), 'map: {}'.format(map_value))
    
    def evaluate_recalls(self, recall_1, recall_5, recall_10):
        pass
        #if (self.iteration+1) % self.log_frequency == 0:
            #print(self.prefix(), 'map: {}'.format(map_value))