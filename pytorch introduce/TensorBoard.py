from random import shuffle
from xml.etree.ElementTree import Comment
import torch
import torch.nn as nn
import torchvision 
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import sys
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter(comment='TensorBoard')

# Hyperparameters
input_size = 784
hidden_size = 500
num_epoch = 6
batch_size = 4
lr = 0.1

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
# Datasets
data_sets = torchvision.datasets.CIFAR10('./data', train = True, download=True, transform=transforms.ToTensor())
data_loader = torch.utils.data.DataLoader(data_sets, shuffle= True, batch_size = batch_size)
data_size = len(data_sets)
n_total_steps = len(data_loader)
examples = iter(data_loader)
examples_data, examples_label = examples.next()
print(len(examples_data))
show_steps = 1000

# ConvNet structure

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward (self, inputs):
        inputs = self.pool(F.relu(self.conv1(inputs)))
        inputs = self.pool(F.relu(self.conv2(inputs)))
        inputs = inputs.view(-1, 16*5*5)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        inputs = self.fc3(inputs)
        return inputs

# # show the batch image
# img_grid = torchvision.utils.make_grid(examples_data)
# writer.add_image('CIFAR10', img_grid)
# model = ConvNet()
# writer_input = examples_data
# writer.add_graph(model, writer_input)
# writer.close()

# Training process

def train_model(model, dataloder, optimizer, criterion, lr_scheduler_step, num_epochs=num_epoch):
    model = model.to(device)

    for epoch in range(num_epochs):
        print('Eopoch {}'.format(epoch+1))
        print('--'*10)
        loss_value = 0.0
        correct_value = 0.0
        for i, (images, labels) in enumerate(dataloder):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _,preds = torch.max(outputs, 1)
            
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            lr_scheduler_step.step()

            loss_value += loss.item()
            correct_value += torch.sum(preds == labels)
            
            if (i+1) % show_steps == 0:
                print(preds)
                print('\nThis is step {} in epoch {}'.format(i+1, epoch+1))
                print('The loss is : {:.4f}, the correct is : {:.4f} '.format(loss_value/show_steps, correct_value/show_steps/batch_size))

                # # visualisierung
                # writer.add_scalar('training loss', loss_value/show_steps, epoch * n_total_steps + i)
                # writer.add_scalar('training correct', correct_value/show_steps/batch_size, epoch * n_total_steps + i)
                    
                loss_value = 0.0
                correct_value = 0.0

        # epoch_loss = loss_value/data_size
        # epoch_correct = correct_value/data_size
        # print('Epoch :{}, Loss: {:.4f} and correct: {:.4f}'.format(epoch+1, epoch_loss, epoch_correct))
    writer.close()
    model_sturcture = copy.deepcopy(model.state_dict())
    model.load_state_dict(model_sturcture)
    return model


# example

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.5)

trained_model = train_model(model, dataloder=data_loader, optimizer=optimizer, criterion=criterion, lr_scheduler_step=step_lr_scheduler,num_epochs=num_epoch)

# test process

test_data = torchvision.datasets.CIFAR10('./data', train=False, transform=transform,download=True)
test_data_loader = torch.utils.data.DataLoader(test_data, shuffle= True,batch_size=batch_size)
test_data_amount = len(test_data_loader)
test_correct = 0.0
test_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = trained_model(inputs)
        loss = criterion(outputs, labels)
        _, prediction = torch.max(outputs, 1)

        # statistics 
        
        test_correct += torch.sum(prediction == labels)

    correct_rate = test_correct / test_data_amount/batch_size
    test_loss = loss.item()

    print('The total loss of model is {:.4f} and the rate of correct is {:.4f}'.format(test_loss, correct_rate))


