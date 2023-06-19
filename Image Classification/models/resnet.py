import torch
import torch.nn as nn


# 基本残差结构块
class BasicBlock(nn.Module):
    expansion= 1 # 扩展系数

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64, dilation=1, norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or width_per_group != 64:
            raise ValueError("BasicBlock only supports groups=1 and width_per_group=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(out_channel)
        self.downsample = downsample # 下采样

    def forward(self, x):
        identity = x

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 主路径和残差路径
        out += identity
        out = self.relu(out)

        return out


# 瓶颈残差结构块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64, dilation=1, norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(out_channel * (width_per_group / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)

        # 组卷积的数
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, dilation=dilation, bias=False, padding=1)
        self.bn2 = norm_layer(width)

        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(out_channel * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, include_top=True):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channel = 64
        self.dilation = 1

        # 残差块是否使用空洞卷积
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )

        # 分组卷积数
        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2, dilate=replace_stride_with_dilation[2])

        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化卷积层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 残差块BN层零初始化
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, channel, block_num, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        # 创建下采样模块
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(channel * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channel, channel, stride, downsample,
                            self.groups, self.width_per_group, previous_dilation, norm_layer))
        self.in_channel = channel * block.expansion

        # 添加残差块
        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel, groups=self.groups, width_per_group=self.width_per_group,
                    dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=1000, replace_stride_with_dilation=None, norm_layer=None, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes,
                  replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, include_top=include_top)


def resnet34(num_classes=1000, replace_stride_with_dilation=None, norm_layer=None, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes,
                  replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, include_top=include_top)


def resnet50(num_classes=1000, replace_stride_with_dilation=None, norm_layer=None, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes,
                  replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, include_top=include_top)


def resnet101(num_classes=1000, replace_stride_with_dilation=None, norm_layer=None, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes,
                  replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, include_top=include_top)


def resnet152(num_classes=1000, replace_stride_with_dilation=None, norm_layer=None, include_top=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes,
                  replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, include_top=include_top)


def resnext50_32x4d(num_classes=1000, replace_stride_with_dilation=None, norm_layer=None, include_top=True):
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, groups=groups, width_per_group=width_per_group,
                  replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, include_top=include_top)


def resnext101_32x8d(num_classes=1000, replace_stride_with_dilation=None, norm_layer=None, include_top=True):
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, groups=groups, width_per_group=width_per_group,
                  replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, include_top=include_top)


def resnext101_64x4d(num_classes=1000, replace_stride_with_dilation=None, norm_layer=None, include_top=True):
    groups = 64
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, groups=groups, width_per_group=width_per_group,
                  replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, include_top=include_top)


def wide_resnet50_2(num_classes=1000, replace_stride_with_dilation=None, norm_layer=None, include_top=True):
    width_per_group = 64 * 2
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, width_per_group=width_per_group,
                  replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, include_top=include_top)


def wide_resnet101_2(num_classes=1000, replace_stride_with_dilation=None, norm_layer=None, include_top=True):
    width_per_group = 64 * 2
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, width_per_group=width_per_group,
                  replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer, include_top=include_top)