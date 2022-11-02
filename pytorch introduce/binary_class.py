from turtle import forward
import torch
import torch.nn as nn

# Binary classfication just one hidden layer
class NeuroNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuroNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred

model = NeuroNet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()