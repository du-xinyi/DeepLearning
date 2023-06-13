import argparse
import torch
import sys
import os
import yaml
import json
import csv
import numpy as np

from pathlib import Path
from utils.model import net
from utils.dataloaders import ImageTransform


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

class_totals = {} # 存储每个类别的总预测数
class_corrects = {} # 存储每个类别的正确预测数

# 命令行参数
def parse_opt(known=False):
    # 创建解析器对象
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', type=str, default='', help='model path')
    parser.add_argument('--source', type=str, default='', help='file/dir')
    parser.add_argument('--data', type=str, default='', help='yaml path')
    parser.add_argument('--indices', type=str, default='', help='class indices path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to train the model (default: "cuda" if available, else "cpu")')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save to project')
    parser.add_argument('--name', default='exp', help='save to project/name')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    # 检查样本路径
    assert os.path.exists(opt.source), "file {} path does not exist.".format(opt.source)

    # 检查模型路径
    assert os.path.exists(opt.weights), "file {} path does not exist.".format(opt.weights)

    # 如果data参数未指定，则使用weights路径的上一级目录下的yaml文件
    if not opt.data:
        opt.data = os.path.join(os.path.dirname(os.path.dirname(opt.weights)), 'cfg.yaml')
        assert os.path.exists(opt.data), "file {} path does not exist.".format(opt.data)

    # 如果indices参数未指定，则使用weights路径的上一级目录下的json文件
    if not opt.indices:
        opt.indices = os.path.join(os.path.dirname(os.path.dirname(opt.weights)), 'class_indices.json')
        assert os.path.exists(opt.indices), "file {} path does not exist.".format(opt.indices)

    # 读取文件
    try:
        with open(opt.data, 'r') as data_file, open(opt.indices, 'r') as json_file:
            data = yaml.safe_load(data_file)
            class_indict = json.load(json_file)
    except Exception as exception:
        print(exception)
        exit(-1)

    # 检查保存路径
    if not os.path.exists(opt.project):
        os.makedirs(opt.project)

    folder_count = len([name for name in os.listdir(opt.project)
                        if os.path.isdir(os.path.join(opt.project, name))])  # 获取指定文件夹内的文件夹数量
    save_dir = os.path.join(opt.project, opt.name + f"{folder_count + 1}")

    os.makedirs(save_dir)

    classes_num = data.get('num_classes') # 类别数量
    model_name = data.get('model') # 网络名称

    # 载入模型
    model = net(number=classes_num, model=model_name)
    model_weight_path = opt.weights
    model.load_state_dict(torch.load(model_weight_path, map_location=opt.device))

    # 载入图片
    transform = ImageTransform()
    images, names, paths = transform.transform_image(opt.source, model_name)

    model.eval() # 评估模式
    # 保存结果
    with open(os.path.join(save_dir, 'predictions.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'True_Class', 'Predicted_Class', 'Probability'])

        with torch.no_grad():
            for image, name, path in zip(images, names, paths):
                print(name)
                output = torch.squeeze(model(image))
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy() # 寻找最大值对应的索引
                predicted_class = class_indict[str(predict_cla)]
                probability = predict[predict_cla].numpy()

                if path != opt.source:
                    true_class = path.split('/')[-1] # 获取path的最后一段作为真实类别标签
                else:
                    true_class = name

                print(name, true_class, predicted_class, probability)

                # 写入结果
                writer.writerow([name, true_class, predicted_class, probability])

                # 更新类别的总预测数和正确预测数
                if predicted_class not in class_totals:
                    class_totals[predicted_class] = 0
                    class_corrects[predicted_class] = 0

                class_totals[predicted_class] += 1
                if predicted_class == true_class:
                    class_corrects[predicted_class] += 1

    # 计算每个类别的平均准确率
    class_accuracies = {}
    for class_name in class_totals:
        class_accuracies[class_name] = class_corrects[class_name] / class_totals[class_name]
        print(class_name, class_accuracies)

    # 计算总体平均准确率
    overall_accuracy = np.mean(list(class_accuracies.values()))
    print('Overall_Accuracy', overall_accuracy)

    # 保存结果
    with open(os.path.join(save_dir, 'result.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Accuracy'])
        for class_name, accuracy in class_accuracies.items():
            writer.writerow([class_name, accuracy])
        writer.writerow(['Overall Accuracy', overall_accuracy])

    # 保存参数
    with open(os.path.join(save_dir, 'cfg.yaml'), 'w') as file:
        selected_params = {}
        selected_params['classes_num'] = classes_num
        selected_params['model_name'] = model_name

        yaml.dump(selected_params, file, sort_keys=False)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)