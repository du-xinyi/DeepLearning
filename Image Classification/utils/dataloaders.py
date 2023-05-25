import os
import torch
import json

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from PIL import Image


data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224), # 随机裁剪缩放图片为224
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),  # 长宽比例不变，将最小边缩放到256
                               transforms.CenterCrop(224),  # 在中心裁减一个224*224大小的图片
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "test": transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# 类别索引
def class_indices(dataset, save_dir):
    # 获取类别索引
    flower_list = dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    # 保存识别类别
    json_str = json.dumps(cla_dict, indent=4)
    with open(os.path.join(save_dir, 'class_indices.json'), 'w') as json_file:
        json_file.write(json_str)


# 训练集数量
def dataset_num(train_dataset, val_dataset):
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    return train_num, val_num


# K折交叉验证数据变换
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.dataset[self.indices[index]]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.indices)


class DataLoaders:
    def __init__(self, data_dir, batch_size, num_workers, type, k_fold, save_dir):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.type = type
        self.k_fold = k_fold
        self.save_dir = save_dir

    def load_data(self):
        train_loaders = []
        val_loaders = []

        train_num = 0
        val_num = 0

        if self.type == 'train_val':
            # 数据集载入
            train_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "train"),
                                                 transform=data_transform["train"])
            val_dataset = datasets.ImageFolder(os.path.join(self.data_dir, "val"), transform=data_transform["val"])

            class_indices(train_dataset, self.save_dir)

            train_num, val_num = dataset_num(train_dataset, val_dataset)

            # 数据装载
            train_loaders = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers)
            val_loaders = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.num_workers)

        elif self.type == 'k_fold':
            dataset = datasets.ImageFolder(self.data_dir)

            class_indices(dataset, self.save_dir)

            # 获取数据集的标签和样本数量
            labels = dataset.targets
            num_samples = len(labels)

            # 使用 KFold 进行数据集划分
            kfold = KFold(n_splits=self.k_fold, shuffle=True)

            print_flag = True  # 循环标志位

            for train_index, val_index in kfold.split(range(num_samples)):
                # 根据索引划分数据集
                train_dataset = TransformedSubset(dataset, train_index, transform=data_transform["train"])
                val_dataset = TransformedSubset(dataset, val_index, transform=data_transform["val"])

                if (print_flag == True):
                    train_num, val_num = dataset_num(train_dataset, val_dataset)
                    print_flag = False

                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.num_workers)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=self.num_workers)

                train_loaders.append(train_loader)
                val_loaders.append(val_loader)

        return train_loaders, val_loaders, train_num, val_num


class ImageTransform:
    def __init__(self):
        self.data_transform = data_transform

    def transform(self, img):
        img = Image.open(img)
        img = img.convert('RGB')
        img = data_transform["test"](img)
        img = torch.unsqueeze(img, dim=0)

        return img

    def transform_image(self, source):
        images = []  # 变换后的图片
        names = []  # 图片名称
        paths = []  # 图片路径
        # 单个图片
        if os.path.isfile(source):
            image = self.transform(source)
            images.append(image)
            names.append(os.path.basename(source))
            paths.append(source)
        # 文件夹
        if os.path.isdir(source):
            file_names = os.listdir(source)
            for file_name in file_names:
                file_path = os.path.join(source, file_name)
                if os.path.isfile(file_path):
                    image = self.transform(file_path)
                    images.append(image)
                    names.append(os.path.basename(file_path))
                    paths.append(file_path)

        return images, names, paths