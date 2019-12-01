
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
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
print('x-after', X.size())

# +1 shift the labels by one so that given the previous letter the char we should predict would be or next char
labels = one_hot_encode(encoded_chars[1:seq_length*num_samples+1], vocab_size) 
y = labels.reshape((num_batches, batch_size, seq_length, vocab_size))
y = y.transpose(1, 2) # transpose the first and second index
y,y.shape

############
# next part
############

#torch.manual_seed(1) # reproducibility

####  Define the network parameters:
hiddenSize = 2 # network size, this can be any number (depending on your task)
numClass = 4 # this is the same as our vocab_size

#### Weight matrices for our inputs 
Wz = Variable(torch.randn(vocab_size, hiddenSize), requires_grad=True)
Wr = Variable(torch.randn(vocab_size, hiddenSize), requires_grad=True)
Wh = Variable(torch.randn(vocab_size, hiddenSize), requires_grad=True)

## Intialize the hidden state
# this is for demonstration purposes only, in the actual model it will be initiated during training a loop over the 
# the number of bacthes and updated before passing to the next GRU cell.
h_t_demo = torch.zeros(batch_size, hiddenSize) 

#### Weight matrices for our hidden layer
Uz = Variable(torch.randn(hiddenSize, hiddenSize), requires_grad=True)
Ur = Variable(torch.randn(hiddenSize, hiddenSize), requires_grad=True)
Uh = Variable(torch.randn(hiddenSize, hiddenSize), requires_grad=True)

#### bias vectors for our hidden layer
bz = Variable(torch.zeros(hiddenSize), requires_grad=True)
br = Variable(torch.zeros(hiddenSize), requires_grad=True)
bh = Variable(torch.zeros(hiddenSize), requires_grad=True)

#### Output weights
Wy = Variable(torch.randn(hiddenSize, numClass), requires_grad=True)
by = Variable(torch.zeros(numClass), requires_grad=True)

##########
# GRU Part
##########

# h gets updated and then we calculate for the next 
h_t_1 = []
h = h_t_demo
for i,sequence in enumerate(X[0]):   # iterate over each sequence in the batch to calculate the hidden state h 
    z = torch.sigmoid(torch.matmul(sequence, Wz) + torch.matmul(h, Uz) + bz)
    r = torch.sigmoid(torch.matmul(sequence, Wr) + torch.matmul(h, Ur) + br)
    h_tilde = torch.tanh(torch.matmul(sequence, Wh) + torch.matmul(r * h, Uh) + bh)
    h = z * h + (1 - z) * h_tilde
    h_t_1.append(h)
    print(f'h{i}:{h}')
h_t_1 = torch.stack(h_t_1)


def gru(x, h):
    outputs = []
    for i,sequence in enumerate(x): # iterates over the sequences in each batch
        z = torch.sigmoid(torch.matmul(sequence, Wz) + torch.matmul(h, Uz) + bz)
        r = torch.sigmoid(torch.matmul(sequence, Wr) + torch.matmul(h, Ur) + br)
        h_tilde = torch.tanh(torch.matmul(sequence, Wh) + torch.matmul(r * h, Uh) + bh)
        h = z * h + (1 - z) * h_tilde

        # Linear layer
        y_linear = torch.matmul(h, Wy) + by
# fsb1: where is the max-reduction??
        # Softmax activation function
        y_t = F.softmax(y_linear, dim=1)

        outputs.append(y_t)
    return torch.stack(outputs), h

def sample(primer, length_chars_predict):
    
    word = primer

    primer_dictionary = [character_dictionary[char] for char in word]
    test_input = one_hot_encode(primer_dictionary, vocab_size)

    h = torch.zeros(1, hiddenSize)

    for i in range(length_chars_predict):
        outputs, h = gru(test_input, h)
        choice = np.random.choice(vocab_size, p=outputs[-1][0].detach().numpy())
        word += character_list[choice]
        test_input = one_hot_encode([choice],vocab_size)
    return word

def cross_entropy(yhat, y):
    return -torch.mean(torch.sum(y * torch.log(yhat), dim=1))
  
def total_loss(predictions, y_true):
    total_loss = 0.0
    for prediction, label in zip(predictions, y_true):
        cross = cross_entropy(prediction, label)
        total_loss += cross
    return total_loss / len(predictions)   

      

# Attached variables 
params = [Wz, Wr, Wh, Uh, Uz, Ur, bz, br, bh, Wy, by] # iterable of parameters that require gradient computation

# Optimizer and training loop
#optimizer = torch.optim.SGD(params, lr = 0.002, momentum=0.7)
optimizer = torch.optim.Adam(params, lr = 0.01)

#optimizer = torch.optim.SGD(model.parameters(), lr = 0.02, momentum=0.8)
#optimizer = torch.optim.LBFGS(params, lr=0.8)
max_epochs = 1000  # passes through the data
for e in range(max_epochs):
    h = torch.zeros(batch_size, hiddenSize)
    for i in range(num_batches):
        x_in = X[i]
        y_in = y[i]
        
        optimizer.zero_grad() # zero out gradients 
        
        out, h = gru(x_in, h)
        #out, h = model(x_in, h)
        loss = total_loss(out, y_in)
        loss.backward(retain_graph=True) # backpropagate through time to adjust the weights and find the gradients of the loss function
        optimizer.step()
    if e % 10 == 0:
        print(f'Epoch: {e+1}/{max_epochs}')
        print(f'Loss: {loss}')
        print(sample('Ma', 10))