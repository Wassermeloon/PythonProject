from turtle import shape
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, transform = None): # transform can be a function
        # data loading
        xy = np.loadtxt('C:/Users/shuai.tan/Documents/Python Scripts/data/wine/wine.csv', delimiter=",", dtype=np.float32,skiprows=1)
        self.n_samples = xy.shape[0]
        # note that we do not convert to tensor here
        self.x = xy[:, 1:]
        self.y = xy[:,[0]]
        self.transform = transform
        
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def __len__(self):
        # len(dataset)
        return self.n_samples

class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

dataset = WineDataset(transform=None)
first_data = dataset[0]
features, lables = first_data
print(type(features), type(lables))
