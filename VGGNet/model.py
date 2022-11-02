# # this model is based on pytorch framework
#
# import torch
# import torch.nn as nn
# from torchsummary import summary
# class VGG(nn.Module):
#     def __init__(self, features, class_num=10000, init_weights=False):
#         super(VGG, self).__init__()
#         self.features = features
#         # define the full connected layers
#         self.clssifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(512*7*7, 2048),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(2048, 2048),
#             nn.ReLU(inplace=True),
#             nn.Linear(2048, class_num),
#         )
#
#         if init_weights:
#             self._init_weights()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.clssifier(x)
#         return x
#
#     def _init_weights(self):
#         for x in self.modules():
#             if isinstance(x, nn.Conv2d):
#                 nn.init.kaiming_normal_(x.weight, mode='fan_out', nonlinearity='relu')
#                 if x.bias is not None:
#                     nn.init.constant(x.bias, 0)
#             elif isinstance(x, nn.Linear):
#                 nn.init.normal(x.weight, 0, 0.01)
#                 nn.init.constant(x.bias, 0)
#
# # for construct the feature net
# cfgs = {
#     'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
#
# def make_features(cfgs: list):
#     layers = []
#     in_channels = 3
#     for x in cfgs:
#         if x == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1)
#             layers += [conv2d, nn.ReLU(True)]
#             in_channels = x
#     return nn.Sequential(*layers)
#
#
# def vgg(model_name='vgg16', **kwargs):
#     try:
#         cfg = cfgs[model_name]
#     except:
#         print("Warning: you have choose wrong vgg net!")
#         exit(-1)
#
#     model = VGG(make_features(cfg), **kwargs)
#     return model
#
# # net = vgg(model_name='vgg13')
# # summary(net,(3, 224, 224), batch_size=32)

# this part, model is based on framework tensorflow
from keras import layers, models, Model, Sequential

def VGG(features, im_height=224, im_width=224, class_num=1000):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    x = features(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(class_num)(x)
    outputs = layers.Softmax()(x)

    model = models.Model(inputs=input_image, outputs=outputs)
    return model

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def features(cfg):
    features_layers = []
    for i in cfg:
        if i == 'M':
            features_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        else:
            features_layers.append(layers.Conv2D(i, kernel_size=3, padding='same', activation='relu'))
    return Sequential(features_layers, name='features')

def vgg(model_name='vgg13', im_height=224, im_width=224, class_num=1000):
    try:
        cfg = cfgs[model_name]
    except Exception as e:
        print(e)
        exit(-1)
    model = VGG(features(cfg), im_height=im_height, im_width=im_width,class_num=class_num)
    return model

# model = vgg('vgg16')
# model.summary()