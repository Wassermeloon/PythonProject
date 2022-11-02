import torch.nn as nn
from torchvision import transforms, datasets, utils
import torch.optim as optim
import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import AlexNet

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('The train is based on ' + str(device))

    # hyper parameters
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    batch_size = 32
    learning_rate =0.0002
    epochs = 2

    dataclass_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.RandomCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.RandomCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)]),
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(),".."))
    imagpath = os.path.join(data_root, "data_set", "flower_data")
    train_dataset = datasets.ImageFolder(root=imagpath + "/train", transform=dataclass_transform["train"])
    train_num = len(train_dataset)

    val_dataset = datasets.ImageFolder(root=imagpath + "/val", transform=dataclass_transform["val"])
    val_num = len(val_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dic = dict((val, key) for key, val in flower_list.items())

    json_str = json.dumps(cla_dic, indent=4)
    with open("class_indice.json", 'w') as json_file:
        json_file.write(json_str)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=4)

    #as an example to show the images

    test_data_iter = iter(val_loader)
    test_imag, test_label = test_data_iter.next()
    print(test_label.shape)

    def imshow(imag):
        imag = imag/2 + 0.5
        imag = imag.numpy()
        plt.imshow(np.transpose(imag, (1, 2, 0)))
        plt.show()

    print(' '.join('%5s' % cla_dic[test_label[i].item()] for i in range(4)))

    # imshow(utils.make_grid(test_imag))

    # for step, (images, labels) in enumerate(train_loader):
    #     print(step)
    #     print(labels.shape)


    net = AlexNet(num_classes=5, init_weights=False)
    net.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    save_path = './AlexNet.pth'

    best_acc = 0.0

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        t1 = time.perf_counter()
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            los = criterion(outputs, labels)

            optimizer.zero_grad()
            los.backward()
            optimizer.step()

            # print statistics
            running_loss += los.item()

            # print train process
            rate = (step + 1)/len(train_loader)
            a = "*" * int(rate*50)
            b = "." * int((1-rate)*50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100), a, b, los), end="")
        print()
        print('The training time is : {:.4f}'.format(time.perf_counter()-t1))

        # validate no more grad need
        net.eval()
        acc = 0.0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                outputs = net(val_images)
                preds = torch.max(outputs, dim=1)[1]
                acc += (preds == val_labels).sum().item()

            accurate_test = acc/len(val_loader)
            if accurate_test > best_acc:
                best_acc = accurate_test
                torch.save(net.state_dict(),save_path)
            print('{} train_loss: {:.4f} test_accuracy: {:.4f}'.format(epoch+1, running_loss/step, acc/val_num))

    print('Finised Training')

if __name__ == '__main__':
    main()