import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 特征提取部分
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool2d = nn.MaxPool2d(kernel_size=3, stride=2)

        # 平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # 分类部分
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256 * 6 * 6, 4096)
        self.fc1 = nn.Linear(4096, 4096)
        self.fc = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool2d(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc(x)

        return x


def alexnet(num_classes=1000):
    return AlexNet(num_classes=num_classes)