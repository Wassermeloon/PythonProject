# import json
# import torch
# from model import vgg
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import torch

# img = Image.open("E:\code\PythonProject\daisy.jpg")
# print(type(img))
# plt.show(img)
#
# img = torch.unsqueeze(img, dim=0)
#
# # read the class
# try:
#     json_file = open("./flower_list.json", 'r')
#     class_name = json.load(json_file)
# except Exception as e:
#     print(e)
#     exit(-1)
#
# # create model
# model = vgg('vgg13', num_classes=5)
# model_weights_path = './vgg13.path'
# model.load_state_dict(torch.load(model_weights_path))
#
# model.eval()
# with torch.no_grad():
#     output = torch.softmax(model(img), dim=0)
#     prediction = torch.max(output, dim=1)[1]
#     pred = class_name[str(prediction)]
#     probalibity = output[prediction].item()
#
# print("||Class:{} || Accuracy:{} ||".format(pred, probalibity))
# plt.show()

import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import vgg

im_height = 224
im_width = 224
model_name = 'vgg16'
class_num = 5

# load sample
sample_root = "E:\\code\\PythonProject\\daisy.jpg"
sample_image = Image.open(sample_root)
# resize
sample_image = sample_image.resize((im_height, im_width))
plt.show(sample_image)

sample_image = np.array(sample_image) / 255.0
sample_image = (np.expand_dims(sample_image, axis=0))

try:
    json_file = open("./flower_list.json", 'r')
    class_name = json.load(json_file)

except Exception as e:
    print('e')
    exit(-1)
model = vgg(model_name=model_name, im_height=im_height, im_width=im_width, class_num=class_num)
weight_path = "./save_weights/VGGNet_{}.h5".format(model_name)
model.load_weights(weight_path)

output = model(sample_image)
output = np.squeeze(output)
pred = np.argmax(output)

print("The predicted class is {} || the accuracy is {:.4f} %".format(class_name[str(pred)], output[pred] * 100))

plt.show()


