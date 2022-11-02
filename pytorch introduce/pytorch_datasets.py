from turtle import shape
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self):
        #data loading
        xy = np.loadtxt('C:/Users/shuai.tan/Documents/Python Scripts/data/wine/wine.csv', delimiter=",", dtype=np.float32,skiprows=1)
        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,0])
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        #dataset[0]
        return self.x[index], self.y[index] #tuple, it deceides the dataforme that returns
    
    def __len__(self):
        #len(dataset)
        return self.n_samples


dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=4,shuffle=True)

 datatiter = iter(dataloader)
 data = datatiter.next()
 features ,lables = data
 print(features, lables)

# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples,n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, lablels) in enumerate(dataloader):
        # forward, backward, update
        if (i+1) % 5 == 0:
            print(f'epoch {epoch+1}/{num_epochs},  step {i+1}/{n_iterations}, inputs {inputs.shape}')

torchvision.datasets.MNIST()
