import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
from model import resnet34, resnet101
import os
import json
from torch.utils import data

# net = resnet101()
# model_weights = "./resnet101_pre.pth"
# missingkeys, unexpected_keys = net.load_state_dict(torch.load(model_weights), strict=False)

class_num = 5
class_list = ["train", "val"]
learning_rate = 0.0001
epochs = 3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("This training is based on " + str(device))

# data preprocessing
image_transforms = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])])
}

# get data

data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
image_path = {
    "train": os.path.join(data_root, "data_set", "flower_data", "train"),
    "val": os.path.join(data_root, "data_set", "flower_data", "val")
}

print(image_path["train"])
image_folder = {
    "train": datasets.ImageFolder(root=image_path["train"], transform=image_transforms["train"]),
    "val": datasets.ImageFolder(root=image_path["val"], transform=image_transforms["val"])
}
image_loader = {
    "train": data.DataLoader(image_folder["train"], batch_size=16, shuffle=True),
    "val": data.DataLoader(image_folder["val"], batch_size=16, shuffle=False)
}
flower_list = image_folder["train"].class_to_idx
flower_list = dict((value, key) for key, value in flower_list.items())
# write to json file
json_str = json.dumps(flower_list, indent=4)
with open("flower_list.json", 'w') as json_file:
    json_file.write(json_str)


net = resnet34()
model_weights = "./resnet34_pre.pth"
missingkeys, unexpected_keys = net.load_state_dict(torch.load(model_weights), strict=False)

# change the model
inchanel = net.fc.in_features
net.fc = nn.Linear(inchanel, class_num)

# train process
net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
best_acc = 0.0
save_path = "./resnet34_trained"

for epoch in range(epochs):
    running_loss = 0.0
    for x in class_list:
        if x == "train":
            net.train()

            for step, (images, labels) in enumerate(image_loader["train"]):
                net = net.to(device)
                images = images.to(device)
                labels = labels.to(device)

                outputs = net(images)
                train_loss = criterion(outputs, labels)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # statistics
                running_loss += train_loss.item()
                rate = (step+1)/len(image_loader["train"])
                a = "*" * int(rate * 50)
                b = "o" * int((1-rate) * 50)

                print('\rCurrent epoch: {} || process: {:^3.0f}% [{}->{}] || current loss: {:.3f}'.format
                      (epoch+1, int(rate*100), a, b, train_loss), end="")

            print()
        else:
            net.eval()
            acc = 0.0
            with torch.no_grad():
                for images, labels in image_loader["val"]:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = net(images)

                    prediction = torch.max(outputs, dim=1)[1]
                    acc += torch.sum(prediction==labels)

                accuracy = acc / len(image_folder["val"])
                print("Current epoch accuracy: {:.2f}".format(accuracy))

                if accuracy > best_acc:
                    torch.save(net.state_dict(), save_path)
print("Training finished")
