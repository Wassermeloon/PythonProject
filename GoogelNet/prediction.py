# prediction
import torch
from model import GoogleNet
import matplotlib.pyplot as plt
from PIL import Image
import json
from torchvision import transforms

im_height = 224
im_width = 224
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
# load the data
sample_root = "E:\\code\\PythonProject\\daisy.jpg"
sample = Image.open(sample_root)
plt.imshow(sample)
plt.show()

sample_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])

sample = sample_transforms(sample)
sample = torch.unsqueeze(sample, dim=0)
print(sample.shape)

try:
    json_file = open("./flower_list.json")
    class_name = json.load(json_file)
    print(class_name)
except Exception as e:
    print(e)
    print(-1)

model = GoogleNet(num_classes=5, aux_logits=False, init_weights=False)
weight_path = "./googleNet.pth"
missing_keys, unexpected_keys = model.load_state_dict(torch.load(weight_path), strict=False)

model.eval()
with torch.no_grad():
    outputs = model(sample)
    pred = torch.max(outputs, dim=1)[1]
    pred_class = class_name[str(pred.item())]
    print("the predicted image is {}".format(pred_class))


# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# from model import GoogleNet
# import json
#
# im_height = 224
# im_width = 224
# image_root = "E:\\code\\PythonProject\\daisy.jpg"
# image = Image.open(image_root)
# image = image.resize((im_height, im_width))
# plt.imshow(image)
# plt.show()
#
# image = np.array(image)
# image = (image / 255 - 0.5) / 0.5
#
# # tensorflow the range is NHWC
# image = np.expand_dims(image, axis=0)
# print(image.shape)
#
# try:
#     json_file = open("./flower_list.json", "r")
#     class_name = json.load(json_file)
#     print(class_name)
# except Exception as e:
#     print(e)
#     print(-1)
#
# # load the model
# model = GoogleNet(im_height=im_height, im_width=im_width, class_num=5,aux_logits=False)
# weight_path = "./save_weights/GoogelNet.ckpt"
#
# # load model weights
# model.load_weights(weight_path)
#
# output = model(image)
# outputs = np.squeeze(output)
# prediction = np.argmax(output, axis=1)[0]
# print("The predicted class is: {}".format(class_name[str(prediction)]))