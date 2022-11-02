# this training is based on pytorch framework
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import json
import os
import torch.optim as optim
from model import vgg
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('This training is working on '+str(device))

# hyper parameters
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
batch_size = 32
learning_rate = 0.0001
epochs = 10

data_list = ['train', 'val']

data_transform = {x: transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)]) for x in data_list
}

data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
imag_path = {
    x: os.path.join(data_root, "data_set", "flower_data", x) for x in data_list}

# data preparation
data_folder = {
    x: datasets.ImageFolder(root=imag_path[x], transform=data_transform[x]) for x in data_list}
data_loader = {
    x: torch.utils.data.DataLoader(data_folder[x], batch_size=batch_size, shuffle=True) for x in data_list}

# get the list of flower
flower_list = data_folder['train'].class_to_idx
class_indices = dict((value, key) for key, value in flower_list.items())

# write dict into json file
json_str = json.dumps(class_indices, indent=4)
with open('flower_list.json', 'w') as json_file:
    json_file.write(json_str)

# show one example in train data set
example = iter(data_loader['train'])
example_images, example_labels = example.next()

# def plotImages(images):
#     #images = images.transpose(1, 3)
#     images = images.permute(0, 2, 3, 1)
#     # print(images.shape)
#     images = images.numpy()/2 + 0.5
#     fig, ax = plt.subplots(2, 4, figsize=(20, 20))
#     print(type(ax))
#     ax = ax.flatten()
#     for img, ax in zip(images, ax):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
# plotImages(images=example_images[:8])


def plotImages(images):
    images = images/2 + 0.5
    print(images.shape)
    images = images.numpy()
    images = np.transpose(images, (1, 2, 0))
    print(images.shape)
    plt.imshow(images)
    plt.show()

plotImages(utils.make_grid(example_images[:4]))

# training process
model_name = 'vgg13'
net = vgg(model_name=model_name, class_num=5, init_weights=False).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
best_acc = 0.0
save_path = './{}Net.path'.format(model_name)
print("|| This is {} model training ||".format(model_name))

best_acc = 0.0
train_loss_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(epochs):
    running_loss = 0.0
    running_loss_val = 0.0
    running_correct = 0.0
    t_start = time.time()
    for x in data_list:
        if x == 'train':
            net.train()
            for step, (images, labels) in enumerate(data_loader["train"]):
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                train_loss = criterion(outputs, labels)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # statistics
                running_loss += train_loss.item()

                # print train process
                rate = (step + 1)/len(data_loader["train"])
                a = "=" * int(rate*50)
                b = "*" * int((1-rate)*50)
                print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, train_loss), end="")
            print()
            print("Epoch {} || the training time is {:.3f}".format(epoch+1, time.time()-t_start))
            train_loss_list.append(running_loss)

        else:
            net.eval()
            with torch.no_grad():
                for images, labels in data_loader["val"]:
                    images = images.to(device)
                    labels = labels.to(device)

                    pred = torch.max(net(images), dim=1)[1]
                    running_correct += torch.sum(pred==labels)
                    running_loss_val += criterion(labels, net(images)).item()

                acc = running_correct/len(data_loader["val"])
                val_acc_list.append(acc)
                val_loss_list.append(running_loss_val)
                print('{} || validation loss {:.4f} || test_accuracy: {:.2f} %'.format
                      (epoch+1, running_loss_val, acc*100))

                if acc > best_acc:
                    best_acc = acc
                    torch.save(net.sate_dict(), save_path)

print('Training finished!')

plt.figure()
plt.plot(range(len(train_loss_list)), train_loss_list, label="train_loss")
plt.plot(range(len(val_loss_list)), val_loss_list, label="val_loss")
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title("Loss figure")
plt.grid()

plt.figure()
plt.plot(range(len(val_acc_list)), val_acc_list, label="val_accuracy")
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title("Accuracy figure")
plt.grid()

plt.show()

# this training is based on tensorflow framework
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from model import vgg
import tensorflow as tf
import json
import os

# hyper parameters
# class_num = 5
# batch_size = 32
# learning_rate = 0.0001
# epochs = 10
# im_height = 224
# im_width = 224
# model_name = 'vgg16'
#
# data_list = ['train', 'val']
#
# data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
# imag_path = {
#     x: os.path.join(data_root, "data_set", "flower_data", x) for x in data_list}
#
# # pre processing
# train_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
# val_image_generator = ImageDataGenerator(rescale=1./225)
#
# # get the dataset
# train_data = train_image_generator.flow_from_directory(directory=imag_path["train"],
#                                                        batch_size=batch_size,
#                                                        shuffle=True,
#                                                        target_size=(im_height, im_width),
#                                                        class_mode="categorical")
# val_data = val_image_generator.flow_from_directory(directory=imag_path["val"],
#                                                    batch_size=4,
#                                                    shuffle=False,
#                                                    target_size=(im_height,im_width))
# # get the catagory of flower
# flower_list = train_data.class_indices
# # get the number of get trained number
# total_train = train_data.n
# total_val = val_data.n
#
# # write dict into  json file
# class_indices = dict((value, key) for key, value in flower_list.items())
#
# json_str = json.dumps(class_indices, indent=4)
# with open("flower_list.json", "w") as json_file:
#     json_file.write(json_str)
#
# model = vgg(model_name=model_name, im_height=im_height, im_width=im_width, class_num=class_num)
# model.summary()
#
# # using high API to train the model
# save_path = "./save_weights/VGGNet_{}.h5".format(model_name)
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
#
# model.compile(optimizer=optimizer, loss=criterion, metrics=['accuracy'])
# callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
#                                                 save_best_only=True,
#                                                 save_weights_only=True,
#                                                 monitor='val_loss')]
# history = model.fit(x=train_data, steps_per_epoch=total_train // batch_size,
#                     epochs=epochs, validation_data=val_data, validation_steps=total_val // batch_size,
#                     callbacks=callbacks)
#
# # plot loss and accuracy image
# history_dict = history.history
# train_loss = history_dict["loss"]
# train_accuracy = history_dict["accuracy"]
# val_loss = history_dict["val_loss"]
# val_accuracy = history_dict["val_accuracy"]
#
# # figure
# plt.figure()
# plt.plot(range(epochs), train_loss, label="train_loss")
# plt.plot(range(epochs), val_loss, label="val_loss")
# plt.legend()
# plt.xlabel("epochs")
# plt.ylabel("loss")
#
# plt.figure()
# plt.plot(range(epochs), train_accuracy, label="train_accuracy")
# plt.plot(range(epochs), val_accuracy, label="val_accuracy")
# plt.legend()
# plt.xlabel("epochs")
# plt.ylabel("accuracy")
# plt.grid()
# plt.show()