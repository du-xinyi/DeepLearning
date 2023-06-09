import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import random
import csv

from sklearn.metrics import confusion_matrix, \
    precision_recall_curve, average_precision_score, \
    precision_score, recall_score, f1_score


# 绘制loss曲线
def plot_loss(train_loss, val_loss, save_dir):
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title('Loss')
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'loss.jpg'), bbox_inches='tight')
    plt.show()


# 绘制accuracy曲线
def plot_accuracy(train_accuracy, val_accuracy, save_dir):
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title('Accuracy')
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'accuracy.jpg'), bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(class_list, targets, predictions, save_dir):
    num_classes = len(class_list)

    # 生成混淆矩阵
    matrix = confusion_matrix(targets, predictions)

    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(12, 10))
    heatmap = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                          xticklabels=range(num_classes),
                          yticklabels=range(num_classes),
                          cbar=True, ax=ax)
    heatmap.set_xlabel("Predicted")
    heatmap.set_ylabel("True")
    heatmap.set_title("Confusion Matrix")
    plt.xticks(np.arange(num_classes) + 0.5, range(num_classes), rotation=45, ha='right')  # 设置x轴刻度位置和标签
    plt.yticks(np.arange(num_classes) + 0.5, range(num_classes), rotation=0, ha='right')  # 设置y轴刻度位置和标签

    # 调整坐标轴标签的位置，以显示完整的标签
    # plt.subplots_adjust(bottom=0.25, left=0.25, top=0.9)

    # 添加颜色条
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    step_size = max(1, int((max_value - min_value) / 10))  # 步长设置为最大值和最小值之差的十分之一，最小为1
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.arange(min_value, max_value + 1, step_size))
    cbar.set_ticklabels(np.arange(min_value, max_value + 1, step_size))

    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.show()


# PR曲线
def plot_pr_curve(targets, predictions, class_list, save_dir):
    precisions = dict()
    recalls = dict()
    average_precision = dict()

    mean_precision = []
    mean_recall = []
    mean_average_precision = 0

    # 计算每个类别的精确度、召回率和平均精确度
    for i, class_name in enumerate(class_list):
        true_class = np.array(targets) == i
        pred_class = np.array(predictions) == i

        precisions[i], recalls[i], _ = precision_recall_curve(true_class, pred_class)

        # 检查并填充空的精确度和召回率
        if precisions[i].size == 0:
            precisions[i] = np.zeros_like(precisions[i])
        if recalls[i].size == 0:
            recalls[i] = np.zeros_like(recalls[i])

        mean_precision.append(precisions[i])
        mean_recall.append(recalls[i])
        average_precision[i] = average_precision_score(true_class, pred_class)
        mean_average_precision += average_precision[i]

    # 计算平均精确度和平均召回率
    mean_precision = np.mean(mean_precision, axis=0)
    mean_recall = np.mean(mean_recall, axis=0)
    mean_average_precision /= len(class_list)

    # 绘制PR曲线
    plt.figure(figsize=(10, 8))

    if len(class_list) <= 6:
        for class_label, class_name in enumerate(class_list):
            plt.plot(recalls[class_label], precisions[class_label], label='Class {}'.format(class_label))
    else:
        random_classes = np.random.choice(len(class_list), size=6, replace=False)
        for i in random_classes:
            plt.plot(recalls[i], precisions[i], label='Class {}'.format(i))

    plt.plot(mean_recall, mean_precision, 'b-', linewidth=2, label='All classes: {:.2f}'.format(mean_average_precision))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    plt.savefig(os.path.join(save_dir, 'PR_curve.jpg'), bbox_inches='tight')
    plt.show()


def plot_f1(class_list, targets, predictions, save_dir):
    num_classes = len(class_list)

    f1_scores = dict()

    for i in range(num_classes):
        true_class = np.array(targets) == i
        pred_class = np.array(predictions) == i
        f1_scores[i] = f1_score(true_class, pred_class)

    # 柱状图
    plt.bar(np.arange(num_classes) + 0.5, f1_scores)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score for Each Class')
    plt.xticks(np.arange(num_classes) + 0.5, range(num_classes), rotation=45, ha='right')  # 横坐标设置在柱状图中间
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, 'f1_Score.jpg'), bbox_inches='tight')
    plt.show()

    # 保存F1值
    with open(os.path.join(save_dir, 'f1_scores.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['class', 'f1 score'])

        for class_idx, score in f1_scores.items():
            class_name = class_list[class_idx]
            writer.writerow([class_name, score])


def plot_evaluation(train_loss, val_loss, train_accuracy, val_accuracy, targets, predictions, confidences, class_list, save_dir):
    plot_loss(train_loss, val_loss, save_dir)
    plot_accuracy(train_accuracy, val_accuracy, save_dir)
    plot_confusion_matrix(class_list, targets, predictions, save_dir)
    plot_pr_curve(targets, predictions, class_list, save_dir)
    plot_f1(class_list, targets, predictions, save_dir)
