# # # there is so called paralle structure: Inception
# # # Inception: previous layers is divided into 4 parts, in each part, the output has the same height and width
# # # 4 parts: 1x1 convolutions 3x3 convolutions 5x5 convolutions, 3x3 Maxpooling, then arrange them by depth
# # # be attention there is so called dimension reduction opreation, by using 1x1 filter
# # # with less channels to get this perpose
# #
# # # this part is based on pytorch framework
#
#
# import torch
# import torch.nn as nn
#
# class GoogleNet(nn.Module):
#     def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
#         super(GoogelNet, self).__init__()
#         self.aux_logits = aux_logits
#
#         self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
#
#         self.conv2 = BasicConv2d(64, 64, kernel_size=1)
#         self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
#         self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
#
#         self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
#         self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
#         self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
#
#         self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
#         self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
#         self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
#         self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
#         self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
#         self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
#
#         self.inception5a =Inception(832, 256, 160, 320, 32, 128, 128)
#         self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
#
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout(0.4)
#         self.fc = nn.Linear(1024, num_classes)
#
#         if aux_logits:
#             self.aux1 = InceptionAux(512, num_classes)
#             self.aux2 = InceptionAux(528, num_classes)
#
#         if init_weights:
#             self._init_weights()
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.maxpool2(x)
#
#         x = self.inception3a(x)
#         x = self.inception3b(x)
#         x = self.maxpool3(x)
#         x = self.inception4a(x)
#
#         if self.training and self.aux_logits:
#             aux1 = self.aux1(x)
#
#         x = self.inception4b(x)
#         x = self.inception4c(x)
#         x = self.inception4d(x)
#
#         if self.training and self.aux_logits:
#             aux2 = self.aux2(x)
#
#         x = self.inception4e(x)
#         x = self.maxpool4(x)
#         x = self.inception5a(x)
#         x = self.inception5b(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, start_dim=1)
#         x = self.dropout(x)
#         x = self.fc(x)
#
#         if self.training and self.aux_logits:
#             return x, aux1, aux2
#
#         return x
#
#     def _init_weights(self):
#         for x in self.modules():
#             if isinstance(x, nn.Conv2d):
#                 nn.init.kaiming_normal_(x.weight, mode='fan_out', nonlinearity='relu')
#                 if x.bias is not None:
#                     nn.init.constant_(x.bias, 0)
#             elif isinstance(x, nn.Linear):
#                 nn.init.normal_(x.weight, 0, 0.01)
#                 nn.init.constant_(x.bias, 0)
#
# class BasicConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, **kwargs):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.relu(x)
#         return x
#
# class Inception(nn.Module):
#     def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
#         super(Inception, self).__init__()
#
#         self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
#
#         self.branch2 = nn.Sequential(
#             BasicConv2d(in_channels, ch3x3red, kernel_size=1),
#             BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
#         )
#         self.branch3 = nn.Sequential(
#             BasicConv2d(in_channels, ch5x5red, kernel_size=1),
#             BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
#         )
#         self.branch4 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#             BasicConv2d(in_channels, pool_proj, kernel_size=1)
#         )
#
#     def forward(self, x):
#         branch1 = self.branch1(x)
#         branch2 = self.branch2(x)
#         branch3 = self.branch3(x)
#         branch4 = self.branch4(x)
#
#         outputs = [branch1, branch2, branch3, branch4]
#         return torch.cat(outputs, dim=1)
#
# class InceptionAux(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(InceptionAux, self).__init__()
#         self.features = nn.Sequential(
#             nn.AvgPool2d(kernel_size=5, stride=3),
#             BasicConv2d(in_channels, 128, kernel_size=1),
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(2048, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(1024, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, start_dim=1)
#         outputs = self.classifier(x)
#
#         return outputs
#
# # # if you use torch.dropout you don't need to set training = self.training
# # # but if you use F.dropout, you must set that

# this part we build the GoogleNet based on Tensorflow framework

from tensorflow.python.keras import layers, models, Model, Sequential


def GoogleNet(im_height, im_width, class_num=1000, aux_logits=False):
    inputs = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    # (none, 224, 224, 3)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='SAME', activation='relu', name="conv2d_1")(inputs)
    # (none, 112, 112, 64)
    x = layers.MaxPool2D(pool_size=2, strides=2, padding="SAME", name="maxpool_1")(x)
    # (none, 56, 56, 64)
    x = layers.Conv2D(64, kernel_size=1, activation='relu', name="conv2d_2")(x)
    # (none, 56, 56, 64)
    x = layers.Conv2D(192, kernel_size=3, padding="SAME", activation="relu", name="conv2d_3")(x)
    # (none, 56, 56, 192)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_2")(x)
    # (none, 28, 28, 192)

    # Inception Structure
    x = Inception(64, 96, 128, 16, 32, 32, name="inception_3a")(x)
    # (none, 28, 28, 256)
    x = Inception(128, 128, 192, 32, 96, 64, name="inception_3b")(x)
    # (none, 28, 28, 480)

    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="maxpool_3")(x)
    # (none, 14, 14, 480)

    # Inception Structure
    x = Inception(192, 96, 208, 16, 48, 64, name="inception_4a")(x)
    # (none, 14, 14, 512)
    if aux_logits:
        aux1 = AuxClassifier(num_class=class_num, name="aux1")(x)

    x = Inception(160, 112, 224, 24, 64, 64, name="inception_4b")(x)
    # (none, 14, 14, 512)
    x = Inception(128, 128, 256, 24, 64, 64, name="inception_4c")(x)
    # (none, 14, 14, 512)
    x = Inception(112, 144, 288, 32, 64, 64, name="inception_4d")(x)
    # (none, 14, 14, 528)

    if aux_logits:
        aux2 = AuxClassifier(num_class=class_num, name="aux2")(x)
    x = Inception(256, 160, 320, 32, 128, 128, name="inception_4e")(x)
    # (none, 14, 14, 532)

    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME", name="max_pool4")(x)
    # (none, 7, 7, 832)
    x = Inception(256, 160, 320, 32, 128, 128, name="inception_5a")(x)
    # (none, 7, 7, 832)
    x = Inception(384, 192, 384, 48, 128, 128, name="inception_5b")(x)
    # (none, 7, 7, 1024)

    x = layers.AvgPool2D(pool_size=7, name="avg_pool")(x)
    # (none, 1, 1, 1024)

    x = layers.Flatten(name="output_flatten")(x)
    x = layers.Dropout(rate=0.5, name="dropout_0.5")(x)
    x = layers.Dense(class_num, name="output_dense")(x)
    aux3 = layers.Softmax(name="aux3")(x)

    if aux_logits:
        model = models.Model(inputs=inputs, outputs=[aux1, aux2, aux3])
    else:
        model = models.Model(inputs=inputs, outputs=aux3)

    return model


class Inception(layers.Layer):
    def __init__(self, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, **kwargs):
        # kwargs: the name of defined layers
        super(Inception, self).__init__()
        self.branch1 = layers.Conv2D(ch1x1, kernel_size=1, activation='relu')

        self.branch2 = Sequential([
            layers.Conv2D(ch3x3red, kernel_size=1, activation='relu'),
            layers.Conv2D(ch3x3, kernel_size=3, padding='SAME', activation='relu')])

        self.branch3 = Sequential([
            layers.Conv2D(ch5x5red, kernel_size=1, activation='relu'),
            layers.Conv2D(ch5x5, kernel_size=3, padding='SAME', activation='relu')])

        self.branch4 = Sequential([
            layers.MaxPool2D(pool_size=3, strides=1, padding="SAME"),
            layers.Conv2D(pool_proj, kernel_size=1, padding="SAME", activation="relu")])

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)

        outputs = layers.concatenate([branch1, branch2, branch3, branch4])

        return outputs


class AuxClassifier(layers.Layer):
    def __init__(self, num_class, **kwargs):
        super(AuxClassifier, self).__init__()
        self.avgpool = layers.AvgPool2D(pool_size=5, strides=3)
        self.conv = layers.Conv2D(128, kernel_size=1, activation='relu')

        self.fc1 = layers.Dense(1024, activation='relu')
        self.fc2 = layers.Dense(num_class)

        self.softmax = layers.Softmax()

    def call(self, inputs, **kwargs):
        outputs = self.avgpool(inputs)
        outputs = self.conv(outputs)
        outputs = layers.Flatten()(outputs)
        outputs = layers.Dropout(rate=0.5)(outputs)
        outputs = self.fc1(outputs)
        outputs = layers.Dropout(rate=0.5)(outputs)
        outputs = self.fc2(outputs)
        outputs = self.softmax(outputs)

        return outputs