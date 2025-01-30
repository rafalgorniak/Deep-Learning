import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def draw_confusion_matrix(data_labels, predicted_labels, title, axis_x, axis_y):
    cm = confusion_matrix(data_labels, predicted_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=np.unique(data_labels),
                yticklabels=np.unique(data_labels))
    plt.title(title)
    plt.xlabel(axis_x)
    plt.ylabel(axis_y)
    plt.show()
