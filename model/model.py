import warnings

import torch
import numpy as np
import torch.nn as nn

from torch.nn import functional as F


def conv_layer(in_channels, out_channels, kernel_size, stride, padding, pooling=False):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(inplace=True)
    )

    if pooling:
        layer.add_module('pooling', nn.MaxPool2d(kernel_size=3, stride=2))

    return layer


def fc_layer(in_features, out_features, dropout=0.5):
    layer = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, out_features),
        nn.ReLU(inplace=True)
    )

    return layer


class AlexNet(nn.Module):
    def __init__(self, num_class=1000, dropout=0.5):
        super(AlexNet, self).__init__()

        self.layer1 = conv_layer(3, 96, 11, 4, 0, pooling=True)
        self.layer2 = conv_layer(96, 256, 5, 1, 2, pooling=True)
        self.layer3 = conv_layer(256, 384, 3, 1, 1)
        self.layer4 = conv_layer(384, 384, 3, 1, 1)
        self.layer5 = conv_layer(384, 256, 3, 1, 1, pooling=True)

        self.layer6 = fc_layer(6 * 6 * 256, 4096, dropout=dropout)
        self.layer7 = fc_layer(4096, 4096, dropout=dropout)
        self.layer8 = fc_layer(4096, num_class, dropout=0)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        return x


class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))

        return torch.concat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, num_class=1000):
        super().__init__()
        self.layer1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3, pooling=True)
        self.layer2 = nn.Sequential(conv_layer(64, 64, kernel_size=1, stride=1, padding=0),
                                    conv_layer(64, 192, kernel_size=3, stride=1, padding=1, pooling=True))
        self.layer3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                                    Inception(256, 128, (128, 192), (32, 96), 64),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                    Inception(512, 160, (112, 224), (24, 64), 64),
                                    Inception(512, 128, (128, 256), (24, 64), 64),
                                    Inception(512, 128, (128, 256), (24, 64), 64),
                                    Inception(528, 256, (160, 320), (32, 128), 128),
                                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                    Inception(832, 384, (192, 384), (48, 128), 128))
        self.layer6 = nn.Sequential(nn.AvgPool2d(7, stride=1), nn.Dropout(p=0.4),
                                    nn.Linear(in_features=1024, out_features=num_class), nn.Softmax())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        return x


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    image = torch.Tensor(np.random.random(size=(3, 227, 227)))
    image = torch.unsqueeze(image, 0)
    network = AlexNet(num_class=1000)
    x = network(image)
    print(x.shape)

