import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

from sklearn.metrics import classification_report


def plot_evaluation(train_loss, val_loss, train_accuracy, val_accuracy, targets, predictions, class_list, save_dir):
    num_classes = len(class_list)

    # 混淆矩阵
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(targets, predictions):
        confusion_matrix[true][pred] += 1

    # 生成分类报告
    report = classification_report(targets, predictions, target_names=class_list, output_dict=True, zero_division=1)

    # 提取各个类别的F1值
    f1_scores = [report[class_name]['f1-score'] for class_name in class_list]

    # 绘制loss曲线
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.jpg'), bbox_inches='tight')
    plt.show()

    # 绘制accuracy曲线
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy.jpg'), bbox_inches='tight')
    plt.show()

    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(12, 10))
    heatmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                          xticklabels=range(num_classes),
                          yticklabels=range(num_classes),
                          cbar=True, ax=ax)
    heatmap.set_xlabel("Predicted")
    heatmap.set_ylabel("True")
    heatmap.set_title("Confusion Matrix")
    plt.xticks(np.arange(num_classes) + 0.5, range(num_classes), rotation=45, ha='right') # 设置x轴刻度位置和标签
    plt.yticks(np.arange(num_classes) + 0.5, range(num_classes), rotation=0, ha='right') # 设置y轴刻度位置和标签

    # 调整坐标轴标签的位置，以显示完整的标签
    # plt.subplots_adjust(bottom=0.25, left=0.25, top=0.9)

    # 添加颜色条
    min_value = np.min(confusion_matrix)
    max_value = np.max(confusion_matrix)
    step_size = max(1, int((max_value - min_value) / 10))  # 步长设置为最大值和最小值之差的十分之一，最小为1
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.arange(min_value, max_value + 1, step_size))
    cbar.set_ticklabels(np.arange(min_value, max_value + 1, step_size))

    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), bbox_inches='tight')
    plt.show()

    # 绘制F1值
    plt.bar(range(num_classes), f1_scores)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score for Each Class')
    plt.xticks(np.arange(num_classes) + 0.5, range(num_classes), rotation=45, ha='right')
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, 'f1_Score.jpg'), bbox_inches='tight')
    plt.show()