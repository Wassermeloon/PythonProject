import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from draw_box_utils import draw_objs
import transform as transforms
from PIL import Image
import torchvision.transforms as ts
import json
from my_dataset import VOC2012DataSet
import os

# read the jsonfile
try:
    json_file = open(".\pascal_voc_classes.json")
    classes = json.load(json_file)
    classes = dict((v, k) for k, v in classes.items())
except Exception as e:
    print(e)
    exit(-1)


data_transforms = {"train": transforms.Compose([transforms.ToTensor(),
                            transforms.RandomHorizontalFlip(prob=0.6)]),
                   "val": transforms.ToTensor()}

# load dataset
train_data_set = VOC2012DataSet(voc_root=os.getcwd(), tranforms=data_transforms["train"], train_set=False)
length = len(train_data_set)
target, image = train_data_set[1000]
print(image.shape)
