import torch
import torch.nn as nn
import torch.functional as f

# option 1 (create nn modules) just one hidden layer
class NeuroNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuroNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) # neuro
        self.relu = nn.ReLU() # activation function
        self.linear2 = nn.Linear(hidden_size, 1) # neuro
        self.sigmoid = nn.Sigmoid() # activation function

    def forward(self, x):
        out = self.linear1(x) 
        out = self.relu(out) # output of activation
        out = self.linear2(out)
        out = self.sigmoid(out) 
        y_pred = out
        return y_pred


# option 2 (use activation function directly in forward pass)
class NeuroNet(nn.Module):
    def __init__(self, input_size, hidden_size): # construct the structure of NN
        super(NeuroNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        y_pred = out
        return y_pred