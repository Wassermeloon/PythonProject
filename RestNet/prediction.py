from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import json
from model import resnet34
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = "E:\code\PythonProject\daisy.jpg"
image = Image.open(data_root)
print(type(image))
plt.imshow(image)

# data processing
data_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225])])

image = data_transforms(image)
image = torch.unsqueeze(image, dim=0)
# print(image.shape)

# load the labels
try:
    json_file = open("./flower_list.json")
    flower_list = json.load(json_file)
    print(flower_list)

except Exception as e:
    print("please check the data_str")
    exit(-1)

# load the weights
model_path = "./resnet34_trained"
model = resnet34(classes_num=5)
# load the weights
misskeys, unexpectkeys = model.load_state_dict(torch.load(model_path), strict=False)
model.eval()
with torch.no_grad():
    image = image.to(device)
    model = model.to(device)
    outputs = torch.softmax(torch.squeeze(model(image)), dim=0).numpy()
    prediction = np.argmax(outputs)
    flower = flower_list[str(prediction)]
    accuracy = outputs[prediction]

print("The predicted flower is :{}, and the accuracy is :{:.2f}%".format(flower, accuracy*100))
plt.show()