import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0) # for every colum(列) softmax

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * predicted)
    return loss

loss = nn.CrossEntropyLoss()
# 3 samples

Y = torch.tensor([2, 0, 1])

# nsamples x nclass
y_predected_good = torch.tensor([[2.0, 2.0, 4.1],[4.0, 2.0, 0.1],[1.0, 3.0, 2.0]])
y_predected_bad = torch.tensor([[1.0, 2.0, 0.5],[1.0, 2.0, 3.0],[1.0, 0.3, 0.2]])

loss_good = loss(y_predected_good, Y)
_, loss_good_position = torch.max(y_predected_good, 1)
good_one = loss_good, loss_good_position

loss_bad = loss(y_predected_bad, Y)
_, loss_bad_position = torch.max(y_predected_bad, 1)
bad_one = loss_bad, loss_bad_position
print(good_one)
print(bad_one)
