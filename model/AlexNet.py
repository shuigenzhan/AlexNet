import warnings

import torch
import numpy as np
import torch.nn as nn


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


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    image = torch.Tensor(np.random.random(size=(3, 227, 227)))
    image = torch.unsqueeze(image, 0)
    network = AlexNet(num_class=1000)
    x = network(image)
    print(x.shape)

