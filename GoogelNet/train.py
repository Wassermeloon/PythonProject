# import json
# from model import GoogleNet
# import torch
# from torchvision import transforms, utils, datasets
# import torch.nn as nn
# from torch.utils.data import DataLoader
# import torch.optim as optim
# import os
# import numpy as np
# import matplotlib.pyplot as plt
#
# # hyper parameters
# data_list = ["train", "val"]
# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]
# batch_size = 64
# epochs = 2
# learning_rate = 0.0002
#
# # create  the model
# net = GoogleNet(num_classes=5, aux_logits=True, init_weights=True)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("This training is based on "+str(device))
#
# # get the dir of the dataseet
# data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
# image_path = {
#     x: os.path.join(data_root, "data_set", "flower_data", x) for x in data_list}
#
# # load data and have some pre-processing
# train_data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
#                                             transforms.RandomHorizontalFlip(),
#                                             transforms.ToTensor(),
#                                             transforms.Normalize(mean=mean, std=std)])
#
# val_data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize(mean=mean, std=std)])
#
# data_transforms = {
#     "train": train_data_transforms,
#     "val": val_data_transforms}
# image_folder = {
#     x: datasets.ImageFolder(root=image_path[x], transform=data_transforms[x]) for x in data_list}
#
# image_data = {
#     "train": DataLoader(image_folder["train"], batch_size=batch_size, shuffle=True),
#     "val": DataLoader(image_folder["val"], batch_size=4, shuffle=False)}
#
# # list classification
# flower_list = image_folder["train"].class_to_idx
# flower_list = dict((value, key) for key, value in flower_list.items())
# # write into json file
# json_str = json.dumps(flower_list, indent=4)
# with open("flower_list.json", 'w') as json_file:
#     json_file.write(json_str)
#
# # # show the example
# # example = iter(image_data["val"])
# # example_images, example_labels = example.next()
# # example_images = example_images/2 + 0.5
# # show_image = utils.make_grid(example_images)
# #
# # show_image = show_image.numpy()
# # show_image = np.transpose(show_image, (1, 2, 0))
# # plt.imshow(show_image)
# # plt.show()
#
# # train the model
# net = net.to(device)
# train_optimizer = optim.Adam(net.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()
#
# acc_best = 0.0
# save_path = "./googleNet.pth"
# train_loss_list = []
# total_step = []
#
# plt.ion()
# plt.figure()
# plt.grid()
# plt.xlabel("steps")
# plt.ylabel("loss")
# plt.title("Training loss")
#
# for epoch in range(epochs):
#     running_loss = 0.0
#     for x in data_list:
#         if x == "train":
#             net.train()
#             for step, (train_images, train_labels) in enumerate(image_data["train"]):
#                 train_images = train_images.to(device)
#                 train_labels = train_labels.to(device)
#                 outputs, aux1, aux2 = net(train_images)
#                 train_loss = criterion(outputs, train_labels)
#                 aux1_loss = criterion(aux1, train_labels)
#                 aux2_loss = criterion(aux2, train_labels)
#                 loss = train_loss + 0.3 * aux1_loss + 0.3 * aux2_loss
#
#                 train_optimizer.zero_grad()
#                 loss.backward()
#                 train_optimizer.step()
#                 # statistics
#                 running_loss += train_loss.item()
#
#                 # statistics
#                 rate = (step + 1) / len(image_data["train"])
#                 a = "o" * int(rate * 50)
#                 b = "·" * int((1-rate) * 50)
#                 print("\rEpoch: {} | training process: {:^3.0f}% [{}>>{}] | current loss: {:.3f}".format
#                       (epoch, int(rate * 100), a, b, train_loss), end="")
#
#                 # # plot evey 100 steps
#                 # if step % 100 == 0:
#                 #     plt.clf()
#                 #     total_step.append(epoch * step + step)
#                 #     train_loss_list.append(running_loss/100.0)
#                 #     plt.plot(total_step, train_loss_list, '-r')
#                 #     plt.plot(epoch * step + step, running_loss/100.0, '*r')
#                 #     plt.draw()
#                 #     plt.pause(0.1)
#                 #     running_loss = 0.0
#
#
#             print()
#
#
#         else:
#             net.eval()
#             acc = 0.0
#             with torch.no_grad():
#                 for val_images, val_labels in image_data["val"]:
#                     val_images = val_images.to(device)
#                     val_labels = val_labels.to(device)
#
#                     outputs = net(val_images)
#                     pred = torch.max(outputs, dim=1)[1]
#                     acc += (pred == val_labels).sum().item()
#
#                 accuracy = acc / len(image_data["val"])
#                 if accuracy > acc_best:
#                     acc_best = accuracy
#                     torch.save(net.state_dict(), save_path)
#
#
# print('Finished Training')

# this part is based on tensorflow framework

from model import GoogleNet
from keras.preprocessing.image import ImageDataGenerator
import os
import json
import tensorflow as tf

# hyper parameters
im_height = 224
im_width = 224
class_num = 5
batch_size = dict(train=32, val=4)
lr = 0.0003
epochs = 3
data_list = ["train", "val"]

net = GoogleNet(im_height=im_height, im_width=im_width, class_num=class_num, aux_logits=True)
# net.summary()

# data_root
data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
image_path = dict((x, os.path.join(data_root, "data_set", "flower_data", x)) for x in data_list)

# get_data
data_preprocess = dict((x, ImageDataGenerator(rescale=((1./255)-0.5)/0.5, horizontal_flip=True)) for x in data_list)
image_data = {"train": data_preprocess["train"].flow_from_directory(directory=image_path["train"],
                                                                    target_size=(im_height, im_width),
                                                                    batch_size=batch_size["train"],
                                                                    shuffle=True,
                                                                    class_mode='categorical'),
              "val": data_preprocess["val"].flow_from_directory(directory=image_path["val"],
                                                                target_size=(im_height, im_width),
                                                                batch_size=batch_size["val"],
                                                                shuffle=False)}
# data_length
image_len = dict((x, image_data[x].n) for x in data_list)
# labels issue
flower_list = image_data["train"].class_indices
flower_list = dict((value, key) for key, value in flower_list.items())
# write into jsonfile
json_str = json.dumps(flower_list, indent=4)
with open("flower_list.json", "w") as jsonfile:
    jsonfile.write(json_str)

# use low API to train the model
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_acc = tf.keras.metrics.CategoricalAccuracy(name="train_accuracy")

val_loss = tf.keras.metrics.Mean(name="val_loss")
val_acc = tf.keras.metrics.CategoricalAccuracy(name="val_accuracy")


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        aux1, aux2, outputs = net(images, training=True)
        loss1 = loss_func(aux1, labels)
        loss2 = loss_func(aux2, labels)
        loss3 = loss_func(outputs, labels)
        total_loss = loss3 + 0.3 * loss1 + 0.3 * loss2
    gradients = tape.gradient(total_loss, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))

    train_loss(total_loss)
    train_acc(outputs, labels)


@tf.function
def val_step(images, labels):
    # for validation we just need the finale outputs, the branch classifier is not needed
    _, _, outputs = net(images, training=False)
    validation_loss = loss_func(outputs, labels)

    val_loss(validation_loss)
    val_acc(outputs, labels)


# training process
acc_best = 0.0
for epoch in range(epochs):
    train_loss.reset_states()
    train_acc.reset_states()
    val_loss.reset_states()
    val_acc.reset_states()
    for x in data_list:
        if x == "train":
            for steps in range(image_len[x] // batch_size[x]):
                train_image, train_labels = next(image_data[x])
                train_step(train_image, train_labels)
                rate = (steps + 1) / (image_len[x] // batch_size[x])
                a = "." * int(rate * 50)
                b = "o" * int(50 * (1 - rate))
                print("\rEpoch: {} | current process: {:^3.0f}% [{} -> {}] | current training loss: {:.3f}".format
                      (epoch+1, int(rate*100), a, b, train_loss.result()), end="")

            print()

        else:
            net.trainable = False
            for steps in range(image_len[x] // batch_size[x]):
                val_image, val_labels = next(image_data[x])
                val_step(val_image, val_labels)
                rate = (steps + 1) / (image_len[x] // batch_size[x])
                a = "." * int(rate * 50)
                b = "o" * int((1 - rate) * 50)
                print("\rEpoch: {} | current process: {:^3.0f} [{} -> {}] | current validation accuracy: {:.3f}".format
                      (epoch+1, rate, a, b, val_acc.result()), end="")
            print()

    info = "Summary: Training loss: {:.3f} |Validation loss: {:.3f}| |Validation accuracy: {:.2f}%|"
    print(info.format(train_loss.result(), val_loss.result(), val_acc.result()*100))

    if val_acc.result() > acc_best:
        acc_best = val_acc.result()
        net.save_weights("./save_weights/GoogelNet.ckpt", save_format='tf')