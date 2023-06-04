import os
import urllib.request
import torch
import torch.nn as nn

from models.googlenet import googlenet
from models.inception import inception_v3
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.resnext import resnext50_32x4d, resnext101_32x8d, resnext101_64x4d


# 模型名称
model_list = {
    'googlenet': googlenet,
    'inception_v3': inception_v3,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnext101_64x4d': resnext101_64x4d
}

# 无训练权重的模型
no_pre_model = {
}

# 预训练权重下载地址
model_urls = {
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth', # IMAGENET1K_V1
    'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth', # IMAGENET1K_V1
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth', # IMAGENET1K_V1
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth', # IMAGENET1K_V1
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth', # IMAGENET1K_V1
    'resnet50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth', # IMAGENET1K_V2
    # 'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth', # IMAGENET1K_V1
    'resnet101': 'https://download.pytorch.org/models/resnet101-cd907fc2.pth', # IMAGENET1K_V2
    # 'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth', # IMAGENET1K_V1
    'resnet152': 'https://download.pytorch.org/models/resnet152-f82ba261.pth', # IMAGENET1K_V2
    # 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth', # IMAGENET1K_V1
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth', # IMAGENET1K_V2
    # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth', # IMAGENET1K_V1
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth', # IMAGENET1K_V2
    'resnext101_64x4d': 'https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth' # IMAGENET1K_V1
}

# 下载预训练权重
def download_with_progress(url, save_path):
    # 下载进度
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = downloaded / total_size * 100
        progress = int(downloaded / total_size * 50)
        print(f"\r[{'=' * progress}{' ' * (50 - progress)}] {percent:.2f}% ", end='')

    urllib.request.urlretrieve(url, save_path, reporthook=show_progress)


def net(number, model, device='cpu', weight=None, transfer_learning=False):
    if model in model_list:
        net = model_list[model]()

        if model in no_pre_model:
            transfer_learning = False
            print("The model has no pre-training weight")

        if transfer_learning:
            if weight:
                model_weight_path = weight
                print("Loaded pretrained weights from: {}".format(model_weight_path))
            else:
                # 检查预训练权重存放路径
                if not os.path.exists('weights'):
                    os.makedirs('weights')

                # 检查预训练权重
                model_weight_name = os.path.basename(model_urls.get(model))
                model_weight_path = os.path.join('weights', model_weight_name)
                if not os.path.exists(model_weight_path):
                    print(f"Downloading weight file for {model}...")
                    download_with_progress(model_urls.get(model), model_weight_path)
                else:
                    print(f"Weight file for {model} already exists, skipping download.")

            assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

            # 加载预训练权重
            missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)

            # 所有参数的梯度设置为不可训练
            # for param in net.parameters():
            #     param.requires_grad = False

        # 修改全连接层
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, number)  # 类别个数

        net.to(device)

        return net
    else:
        raise ValueError('Invalid model selection.')