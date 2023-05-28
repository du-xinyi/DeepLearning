import argparse
import torch
import sys
import os
import yaml
import time

from pathlib import Path
from utils.parameters import opt_yaml, model_parameters
from utils.dataloaders import DataLoaders
from utils.model import net
from utils.optimizer import optimizer
from utils.loss import select_loss
from utils.plot import plot_loss, plot_accuracy, plot_confusion_matrix, plot_f1_scores


# 获取当前文件夹路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

optimizer_list = ['SGD', 'Adam', 'RMSprop'] # 优化器列表
lr_list = ['Fixed', 'Cosine'] # 学习率列表
loss_list = ['CrossEntropyLoss', 'NLLLoss', 'BCEWithLogitsLoss', 'BCELoss'] # 损失函数列表
model_list = [
        'resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152'] # 模型列表

Train_Loss = [] # 训练过程的损失率
Train_Accuracy = [] # 训练过程的准确率
Val_Loss = [] # 验证过程的损失率
Val_Accuracy = [] # 验证过程的准确率

predictions = [] # 预测结果
targets = [] # 真实标签

counter = 1 # 计数器

# 命令行参数
def parse_opt(known=False):
    # 创建解析器对象
    parser = argparse.ArgumentParser()

    # 可选参数
    parser.add_argument('--data_dir', type=str, default=ROOT / 'datasets', help='path to the data directory')
    parser.add_argument('--epochs', type=int, default=30, help='total training epochs')
    parser.add_argument('--k', type=int, default=10, help='number of folds for k-fold cross validation')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training and validation')
    parser.add_argument('--optimizer', type=str, choices=optimizer_list, default='Adam', help='optimizer')
    parser.add_argument('--lr', type=str, choices=lr_list, default='Fixed', help='learning rate')
    parser.add_argument('--loss', type=str, choices=loss_list, default='CrossEntropyLoss', help='loss function')
    parser.add_argument('--num_workers', type=int, default=min([os.cpu_count(), 8]),
                        help='number of worker threads for loading data')
    parser.add_argument('--model', type=str, choices=model_list, default='resnet152',help='choose the network model')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--no_transfer_learning', action='store_false',
                        help='disable transfer learning')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to train the model (default: "cuda" if available, else "cpu")')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project')
    parser.add_argument('--name', default='exp', help='save to project/name')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    # 开始时间
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    # 读取配置文件
    with open(ROOT / 'config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    # 检查数据集路径
    assert os.path.exists(opt.data_dir), "{} path does not exist.".format(opt.data_dir)

    # 检查保存路径
    if not os.path.exists(opt.project):
        os.makedirs(opt.project)

    folder_count = len([name for name in os.listdir(opt.project)
                        if os.path.isdir(os.path.join(opt.project, name))]) # 获取指定文件夹内的文件夹数量
    save_dir = os.path.join(opt.project, opt.name + f"{folder_count+1}")

    os.makedirs(save_dir)

    # 判断训练模式
    if os.path.exists(os.path.join(opt.data_dir, 'train')):
        print("train-val validation")
        type = 'train_val'
        num_classes = len([name for name in os.listdir(os.path.join(opt.data_dir, 'train'))
                       if os.path.isdir(os.path.join(os.path.join(opt.data_dir, 'train'), name))])
        steps = opt.epochs
    else:
        print("k-fold cross validation")
        type = 'k_fold'
        num_classes = len([name for name in os.listdir(opt.data_dir)
                       if os.path.isdir(os.path.join(opt.data_dir, name))])
        steps = opt.k

    # 数据载入
    data_loaders = DataLoaders(opt.data_dir, opt.batch_size, opt.num_workers, type, opt.k, save_dir)
    train_loaders, val_loaders, train_num, val_num, class_list = data_loaders.load_data()

    # 网络载入
    model = net(num_classes, opt.model, opt.device, opt.weights, opt.no_transfer_learning)

    # 优化器和学习率调度器载入
    optimize, schedule =optimizer(opt.optimizer, opt.lr, steps, config, model.parameters())

    # 损失函数载入
    loss_function = select_loss(opt.loss)

    best_acc = 0.0

    # 保存的模型
    weight_dir = os.path.join(save_dir, 'weights')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    best_weight = os.path.join(weight_dir, 'best.pth')
    last_weight = os.path.join(weight_dir, 'last.pth')

    for epoch in range(steps):
        # train
        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0
        step = 0

        if type == 'k_fold':
            train_loader = train_loaders[epoch]
            val_loader = val_loaders[epoch]
        else:
            train_loader = train_loaders
            val_loader = val_loaders

        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimize.zero_grad()
            logits = model(images.to(opt.device))
            loss = loss_function(logits, labels.to(opt.device))
            loss.backward()

            # 更新优化器和学习率
            optimize.step()
            if schedule is not None:
                schedule.step()

            # 计算训练损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels.to(opt.device)).sum().item()
            train_total += labels.size(0)

            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate * 100), a, b, loss), end="")
        print()
        train_loss = running_loss / step
        train_accuracy = train_correct / train_total

        # validate
        model.eval()
        running_loss = 0.0
        acc = 0.0

        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = model(val_images.to(opt.device))
                loss = loss_function(outputs, val_labels.to(opt.device))
                predict_y = torch.max(outputs, dim=1)[1]

                running_loss += loss.item()
                acc += (predict_y == val_labels.to(opt.device)).sum().item()

                # 保存预测结果和真实标签
                predictions.extend(predict_y.tolist())
                targets.extend(val_labels.tolist())
            val_loss = running_loss / step
            val_accurate = acc / val_num
            # 保存模型
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model.state_dict(), best_weight) # 最优模型
            torch.save(model.state_dict(), last_weight) # 最终模型

            print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  val_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, train_loss, train_accuracy, val_loss, val_accurate))
            Train_Loss.append(train_loss)
            Train_Accuracy.append(train_accuracy)
            Val_Loss.append(val_loss)
            Val_Accuracy.append(val_accurate)

            model_parameters(Train_Loss, Train_Accuracy, Val_Loss, Val_Accuracy, save_dir) # 保存训练过程数据

    # 绘制损失率和准确率
    plot_loss(Train_Loss, Val_Loss, save_dir)
    plot_accuracy(Train_Accuracy, Val_Accuracy, save_dir)

    # 绘制混淆矩阵和颜色条
    plot_confusion_matrix(targets, predictions, class_list, save_dir)

    # 绘制F1图
    plot_f1_scores(targets, predictions, class_list, save_dir)

    # 结束时间
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    # 保存参数
    opt_yaml(opt, type, num_classes, steps, config, start_time, end_time, save_dir)

    print('Finished Training')

    return 0


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)