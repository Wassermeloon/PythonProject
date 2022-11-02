# # resnet structure
# # Batch Normalization
#
# # this part is based on pytorch
#
# import torch
# import torch.nn as nn
#
#
# class BasicResNetBlock(nn.Module):
#     # it is used for 18 or 34 layers. Other layers has something diffrient with the branch(it has downsample)
#     expansion = 1
#
#     def __init__(self,
#                  in_channel,
#                  out_channel,
#                  stride=1,
#                  downsample=None):
#         super(BasicResNetBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                                kernel_size=3, stride=stride, padding=1, bias=False)
#
#
#         # dash arrow the stride is 2, but inside the layer the stride is 1
#         # if stride=1, the output size will not be changed. output = (input-3+2*1)/stride+1
#         # so stride must be a variable
#
#         self.bn1 = nn.BatchNorm2d(out_channel)
#         self.activation_relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#                                kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channel)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.activation_relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         out += identity
#         out = self.activation_relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     # it is used for 50, 101, 152, layers
#
#     expansion = 4
#
#     def __init__(self,
#                  in_channel,
#                  out_channel,
#                  stride=1,
#                  downsample=None):
#         # the out channel means the first layer channel output channel
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                                kernel_size=1, stride=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channel)
#
#         self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#                                kernel_size=3, stride=stride, bias=False, padding=1)
#
#         # by dash arrow the stride is 2, but inside the structure the stride is 1
#         # so stride must be a variable
#
#         self.bn2 = nn.BatchNorm2d(out_channel)
#
#         self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
#                                kernel_size=1, stride=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
#
#         self.activation_relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#
#     def forward(self, x):
#         identity = x
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.activation_relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.activation_relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         out += identity
#
#         out = self.activation_relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self,
#                  block,
#                  block_num,
#                  classes_num=1000,
#                  include_top=True):
#         # include_top: we can build some more complex net, which is based on Resnet
#         # block_num is a list
#         super(ResNet, self).__init__()
#         self.include_top = include_top
#         self.in_channel = 64
#
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2,
#                                padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.in_channel)
#
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         # the size of feature will not change. so the first layer, the stride is 1
#         # and for 18 and 34 layer the depth will not change, it means, first layer 18, 34 layer don't need dowsample
#         # but for 50 layer and more the depth is changed, so it needs downsample.
#         # but for later layers, the depth and the size are changed. so the stride must be a changeable variable.
#         self.layer1 = self._make_layer(block=block, channel=64, block_num=block_num[0], stride=1)
#         self.layer2 = self._make_layer(block=block, channel=128, block_num=block_num[1], stride=2)
#         self.layer3 = self._make_layer(block=block, channel=256, block_num=block_num[2], stride=2)
#         self.layer4 = self._make_layer(block=block, channel=512, block_num=block_num[3], stride=2)
#
#         if self.include_top:
#             self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#             self.flatten = nn.Flatten(start_dim=1)
#             self.fc = nn.Linear(512 * block.expansion, classes_num)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#
#         self.activation_relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.activation_relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         if self.include_top:
#             x = self.avgpool(x)
#             x = self.flatten(x)
#             x = self.fc(x)
#
#         return x
#
#
#     def _make_layer(self, block, channel, block_num, stride=1):
#
#         # this stride, chanel is the first layer of block
#         # just the first input layer it needs downsample
#         # inside the block the size of feature are always the same
#
#         downsample = None
#         if stride != 1 or self.in_channel != channel * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channel, channel * block.expansion, stride=stride, kernel_size=1, bias=False),
#                 nn.BatchNorm2d(channel * block.expansion)
#             )
#         # think the whole layer als one structure, the input is which block you use, the input channel, the input stride
#         # the number of channel.
#         # but for 18 and 34 layer, the first layer don't need downsample, condition: stride=1, input channel is the same
#         # to output channel
#         # for the rest layers, it need downsample, condition: stride=2, input channel is same to output channel
#
#         # but for 50, 101, 140 layer, the first layer, need dowsample, but feature size will not be changed, condition:
#         # stride=1, input channel is not same to output channel
#         # for the rest channels, it needs dowmsample. condition: stride=2, input channel is not the same to output channel
#         # so the criterion: input channel is not same to the output channel, stride = 2
#
#         layers = []
#         layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
#         self.in_channel = channel * block.expansion
#
#         for _ in range(1, block_num):
#             layers.append(block(self.in_channel, channel))
#             # inside the block , there is always without downsample
#             # the channel, this variable means the first layer channel of each block
#
#         return nn.Sequential(*layers)
#
#
# def resnet34(classes_num=1000, include_top=True):
#     return ResNet(block=BasicResNetBlock, block_num=[3, 4, 6, 3], classes_num=classes_num, include_top=include_top)
#
#
# def resnet101(classes_num=1000, include_top=True):
#     return ResNet(block=Bottleneck, block_num=[3, 4, 23, 3], classes_num=classes_num, include_top=include_top)
#
#
# this part is based on tensprflow framework

from keras import models, Model, layers, Sequential

class BasicResnetBlock(layers.Layer):
    expansion = 1.0

    def __int__(self, out_chanel, stride=1, dowsample=None, **kwargs):
        super(BasicResnetBlock, self).__int__()

        self.conv1 = layers.Conv2D(out_chanel, kernel_size=3, padding="SAME", strides=stride, use_bias=False)
        self.bn1 = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        # ---------------------------------------------------
        self.conv2 = layers.Conv2D(out_chanel, kernel_size=3, padding="SAME", strides=1, use_bias=False)
        self.bn2 = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        # ---------------------------------------------------
        self.downsample = dowsample
        self.activation_relu =layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False, *args, **kwargs):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.activation_relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.activation_relu(x)

        return x

class Bottleneck(layers.Layer):
    expansion = 4

    def __init__(self, out_chanel, stride=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__()

        self.conv1 = layers.Conv2D(out_chanel, kernel_size=1, strides=1, use_bias=False, name="conv1")
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")

        # -------------------------------------------------
        self.conv2 = layers.Conv2D(out_chanel, kernel_size=3, strides=stride, use_bias=False, padding="SAME",
                                   name="conv2")
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")

        # -------------------------------------------------
        self.conv3 = layers.Conv2D(out_chanel*self.expansion, kernel_size=1, strides=1, use_bias=False,
                                   name="conv3")
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")

        # -------------------------------------------------
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()

    def call(self, inputs, training=False,*args, **kwargs):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([x, identity])
        x = self.relu(x)

        return x


def _resnet(block,
            block_num,
            im_width = 224,
            im_height = 224,
            class_num = 1000,
            include_top = True):
    # in tensorflow the channels of picture is arranged by NHWC
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1")(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    # --------------------------------------------------------------
    x = _make_layers(block=block, in_channels=x.shape[-1], channel=64, block_num=[0], stride=1, name="block1")(x)
    x = _make_layers(block=block, in_channels=x.shape[-1], channel=128, block_num=[0],stride=2, name="block2")(x)
    x = _make_layers(block=block, in_channels=x.shape[-1], channel=256, block_num=[0],stride=2, name="block3")(x)
    x = _make_layers(block=block, in_channels=x.shape[-1], channel=512, block_num=[0], stride=1, name="block4")(x)

    if include_top:
        x = layers.GlobalAvgPool2D()(x)
        x = layers.Dense(class_num, name="logits")(x)
        predict = layers.Softmax()(x)

    else:
        predict = x
    model = Model(inputs=input_image, outputs=predict)

    return model


def _make_layers(block,
                 in_channels,
                 channel,
                 block_num,
                 name,
                 stride=1, **kwargs):
    downsample = None
    if stride != 1 or in_channels != channel * block.expansion:
        downsample = Sequential(
            [layers.Conv2D(channel * block.expansion, kernel_size=1, strides=stride,
                           use_bias=False, name="conv1"),
             layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="BatchNorm")],
            name="shortcut"
        )

    layers_list = []
    layers_list.append(block(channel, stride=stride, downsample=downsample, name="unit_1"))

    for index in range(1, block_num):
        layers_list.append(block(channel, stride=1, name="unit_"+str(index+1)))

    return Sequential(layers_list, name=name)


def resnet34(im_width=224, im_height=224, class_num=1000, include_top=True):
    return _resnet(BasicResnetBlock, [3, 4, 6, 3], im_width, im_height, class_num, include_top=include_top)


def resnet50(im_width=224, im_height=224, class_num=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 6, 3], im_width, im_height, class_num, include_top=include_top)


def resnet101(im_width=224, im_height=224, class_num=1000, include_top=True):
    return _resnet(BasicResnetBlock, [3, 4, 23, 3], im_width, im_height, class_num, include_top=include_top)
