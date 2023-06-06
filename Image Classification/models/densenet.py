import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, places):
        super().__init__()

        self.conv = nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class _DenseLayer(nn.Module):
    def __init__(self, input_features, growth_rate, bn_size, drop_rate=0):
        super(_DenseLayer, self).__init__()

        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_features, out_channels=bn_size * growth_rate, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1,
                      bias=False),
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        y = self.dense_layer(x)
        if self.drop_rate > 0:
            y = self.dropout(y)
        return torch.cat([x, y], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_features, growth_rate, bn_size, drop_rate=0):
        super(DenseBlock, self).__init__()

        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(input_features + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _TransitionLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=input_features, out_channels=output_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition_layer(x)


class DenseNet(nn.Module):
    def __init__(self, init_channels=64, growth_rate=32, blocks=None, num_classes=1000):
        super(DenseNet, self).__init__()

        blocks *= 4

        bn_size = 4
        drop_rate = 0

        self.conv1 = BasicConv2d(in_planes=3, places=init_channels)
        num_features = init_channels

        self.layer1 = DenseBlock(num_layers=blocks[0], input_features=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate

        self.transition1 = _TransitionLayer(input_features=num_features, output_features=num_features // 2)
        num_features = num_features // 2

        self.layer2 = DenseBlock(num_layers=blocks[1], input_features=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate

        self.transition2 = _TransitionLayer(input_features=num_features, output_features=num_features // 2)
        num_features = num_features // 2

        self.layer3 = DenseBlock(num_layers=blocks[2], input_features=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate

        self.transition3 = _TransitionLayer(input_features=num_features, output_features=num_features // 2)
        num_features = num_features // 2

        self.layer4 = DenseBlock(num_layers=blocks[3], input_features=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[3] * growth_rate

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.transition1(x)
        x = self.layer2(x)
        x = self.transition2(x)
        x = self.layer3(x)
        x = self.transition3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def densenet121():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 24, 16])


def densenet161():
    return DenseNet(init_channels=96, growth_rate=48, blocks=[6, 12, 36, 24])


def densenet169():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 32, 32])


def densenet201():
    return DenseNet(init_channels=64, growth_rate=32, blocks=[6, 12, 48, 32])