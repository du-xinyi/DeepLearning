import os
import urllib.request
import torch
import torch.nn as nn

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


# 模型名称
model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152
}

# 预训练权重下载地址
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
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
    if model in model_dict:
        net = model_dict[model]()

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