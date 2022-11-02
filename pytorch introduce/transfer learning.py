from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.optim import lr_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cup')

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transform = {
    'train': transforms.Compose([transforms.RandomResizedCrop(224), # cut image randomly into 224 small images
                                 transforms.RandomHorizontalFlip(), # 沿着水平方向随机翻转
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean, std)]),

    'val': transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)]),
}

# import data
data_dir = 'C:\\Users\\shuai.tan\\Desktop\\py\\hymenoptera_data'                 # 文件路径
sets = ['train', 'val']                                                          # 两个数据集   
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), transform=data_transform[x]) # 读取数据
                            for x in ['train', 'val']
}

dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 4, shuffle=True) # 整理数据
                                   for x in ['train', 'val']}

datasets_size = {
    x: len(image_datasets[x]) for x in ['train', 'val']
}

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
print(classes)

# datasetss_size = {
#     x: len(dataloaders[x]) for x in ['train', 'val'] 这里的size是之前的1/4， 原因就是4个batchsize组成了一个大的array
# }
class_names = image_datasets['train'].classes # 返回的是文件夹train下的子文件夹的名字
print(class_names)
print(datasets_size['train'])

def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict()) # 建立的神经网络的参数和结构 相当于建立了变量
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch{epoch}/{num_epochs-1}')
        print('-'*10)

        # Each epoch has a training and validatiion pahse
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # set model to training mode

            else:
                model.eval() # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward over data
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            if phase =='train':
                scheduler.step()

            epoch_loss = running_loss / datasets_size[phase]
            epoch_acc = running_corrects.double() / datasets_size[phase]

            print('{} Loss: {:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))


            # deep copy the modell
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val ACC: {:4f}'.format(best_acc))

    # load best model weughts
    model.load_state_dict(best_model_wts)
    return model

###
model = models.resnet18(pretrained=True)



num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Scheduler
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)

model = train_model(model,criterion, optimizer, step_lr_scheduler, num_epochs = 5)


# how to use scheduler
# the structure will be like that:
# for epoch in range(100):
#   train() # optimizer.step()
#   evaluate()
#   scheduler.step()