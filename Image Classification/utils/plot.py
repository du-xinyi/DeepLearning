import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

from sklearn.metrics import classification_report

def plot_loss(Train_Loss, Val_Loss, save_dir):
    plt.plot(Train_Loss, label='Train Loss')
    plt.plot(Val_Loss, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title('Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.jpg'))
    plt.show()


def plot_accuracy(Train_Accuracy, Val_Accuracy, save_dir):
    plt.plot(Train_Accuracy, label='Train Accuracy')
    plt.plot(Val_Accuracy, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'accuracy.jpg'))
    plt.show()


def plot_confusion_matrix(targets, predictions, class_list, save_dir):
    num_classes = len(class_list)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(targets, predictions):
        confusion_matrix[true][pred] += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                          xticklabels=class_list,
                          yticklabels=class_list,
                          cbar=True, ax=ax)
    heatmap.set_xlabel("Predicted")
    heatmap.set_ylabel("True")
    heatmap.set_title("Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, ha='right')
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.arange(np.min(confusion_matrix), np.max(confusion_matrix) + 1))
    cbar.set_ticklabels(np.arange(np.min(confusion_matrix), np.max(confusion_matrix) + 1))
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.show()


def plot_f1_scores(targets, predictions, class_list):
    report = classification_report(targets, predictions, target_names=class_list, output_dict=True, zero_division=1)

    f1_scores = [report[class_name]['f1-score'] for class_name in class_list]

    plt.bar(class_list, f1_scores)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('F1 Score for Each Class')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()