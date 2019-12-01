import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

text = "MathMathMathMathMath"

character_list = list(set(text))   # get all of the unique letters in our text variable
vocabulary_size = len(character_list)   # count the number of unique elements
character_dictionary = {char:e for e, char in enumerate(character_list)}  # create a dictionary mapping each unique char to a number
encoded_chars = [character_dictionary[char] for char in text] #integer representation of our vocabulary 

def one_hot_encode(encoded, vocab_size):
    result = torch.zeros((len(encoded), vocab_size))
    for i, idx in enumerate(encoded):
        result[i, idx] = 1.0
    return result

# One hot encode our encoded charactes
batch_size = 2
seq_length = 3
num_samples = (len(encoded_chars) - 1) // seq_length # time lag of 1 for creating the labels
vocab_size = 4

data = one_hot_encode(encoded_chars[:seq_length*num_samples], vocab_size).reshape((num_samples, seq_length, vocab_size))
num_batches = len(data) // batch_size
X = data[:num_batches*batch_size].reshape((num_batches, batch_size, seq_length, vocab_size))
# swap batch_size and seq_length axis to make later access easier
print('x-before', X.size())
X = X.transpose(1, 2)
print('x-after', X.size()) # shape and data correct after this stuff!

# +1 shift the labels by one so that given the previous letter the char we should predict would be or next char
labels = torch.tensor(encoded_chars[1:seq_length*num_samples+1])
y = labels.reshape((num_batches, batch_size, seq_length))
y = y.transpose(1, 2) # transpose the first and second index


## parameters:
feature_size = 4
hidden_size = 2

class GruModel(nn.Module):
    
    def __init__(self):
        super(GruModel, self).__init__()
        self.gru = nn.GRU(feature_size, hidden_size)
        self.fc = nn.Linear(2, 4)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        y_linear = self.fc(out)
        y_t = F.softmax(y_linear, dim=2)
        return y_t, h

model = GruModel()

def sample(primer, length_chars_predict):
    
    word = primer

    primer_dictionary = [character_dictionary[char] for char in word]
    test_input = one_hot_encode(primer_dictionary, vocab_size)
    test_input = test_input.view(len(primer), 1, 4)

    h = torch.zeros(1, 1, hidden_size)

    for i in range(length_chars_predict):
        y_ts, h = model(test_input, h)
        out = y_ts[-1]
        choice = np.random.choice(vocab_size, p=out[0].detach().numpy())
        word += character_list[choice]
        test_input = one_hot_encode([choice],vocab_size).view(1, 1, 4)
    return word

# Optimizer and training loop
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

criterion = nn.NLLLoss()

def cross_entropy(yhat, y):
    return -torch.mean(torch.sum(y * torch.log(yhat), dim=0))
  
def total_loss(predictions, y_true):
    total_loss = 0.0
    for prediction, label in zip(predictions, y_true):
        cross = cross_entropy(prediction, label)
        total_loss += cross
    return total_loss / len(predictions)  

max_epochs = 1000  # passes through the data
for e in range(max_epochs):
    h = torch.zeros(1, batch_size, hidden_size)
    
    latest_loss = 0
    
    for i in range(num_batches):
        x_in = X[i]
        y_in = y[i]
        
        optimizer.zero_grad() # zero out gradients 
        
        out, h = model(x_in, h)
        
        # reshape
        out = out.view(6, 4)
        y_in = y_in.contiguous().view(6)

        loss = criterion(out, y_in)
        loss.backward(retain_graph=True) # backpropagate through time to adjust the weights and find the gradients of the loss function
        
        latest_loss = 1+loss.item()
        
        optimizer.step()
    if e % 10 == 0:
        print(f'Epoch: {e+1}/{max_epochs}')
        print(f'Loss: {latest_loss}')
        other_loss = total_loss(out, one_hot_encode(y_in, 4))
        print(f'Other Loss: {other_loss}')
        print(sample('Ma', 10))