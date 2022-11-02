from turtle import forward
from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'CPU')

# Huperparamater
num_epochs = 3
batch_size = 4
learning_rate = 0.001

# dataset has PILImage image of range [0, 1]
# we transform them to Tensor of normalized [-1, 1]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train = True, download=True, transform=transform)
data_size = len(train_dataset)
print(data_size)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
print(len(train_loader))

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trunck')

# implement conv net
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # layer of mask, 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
n_total_steps = len(train_loader)

# training
for epoch in range(num_epochs):
    since = time.time()
    loss_value = 0.0
    correct_value = 0.0

    for i, (images, label) in enumerate(train_loader):
        # forward
        images = images.to(device)
        label = label.to(device)

        output = model(images)
        loss = criterion(output, label)
        _,preds = torch.max(output, 1)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # staristic
        loss_value += loss.item()
        correct_value += torch.sum(preds == label)

    epoch_loss = loss_value/data_size
    epoch_correct = correct_value/data_size
    print('Epoch Loss: {:.4f} and Eopch correct: {:.4f}'.format(epoch_loss, epoch_correct))
used_time = time.time() - since

print('The total training time {}'.format(used_time))


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predictions == label).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predictions[i]
            if (label == pred):
                n_class_correct[label] += 1
                n_class_samples[label] += 1

    acc = 100.0 * n_correct/n_samples
    print(f'Accuracy of network: {acc}%')

    for i in range(10):
        acc = 100.0 * n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy of class {classes[i]}: {acc}%')