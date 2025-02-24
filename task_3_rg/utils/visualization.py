import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_embeddings_classification(embeddings, labels, title="Embeddings visualization"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.show()


def plot_embeddings_regression(embeddings, labels, title="Embeddings visualization"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.show()


def plot_chart_train_phase(train_los, val_los, acc_metr, prec_metr, recc_metr, f1_metr):
    # Assuming these arrays are returned from train_evaluate_save_model
    # Replace these with your actual values
    epochs = list(range(1, len(train_los) + 1))

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)  # Subplot for losses
    plt.plot(epochs, train_los, label='Train Loss', marker='', color='blue')
    plt.plot(epochs, val_los, label='Validation Loss', marker='', color='orange')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy, precision, recall, and F1 score
    plt.subplot(1, 2, 2)  # Subplot for metrics
    plt.plot(epochs, acc_metr, label='Accuracy', marker='', color='green')
    plt.plot(epochs, prec_metr, label='Precision', marker='', color='purple')
    plt.plot(epochs, recc_metr, label='Recall', marker='', color='red')
    plt.plot(epochs, f1_metr, label='F1 Score', marker='', color='brown')
    plt.title('Metrics Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()


def plot_chart_train_phase_regression(train_los, val_los, mae_met):
    epochs = list(range(1, len(train_los) + 1))

    # Plot training and validation loss
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)  # Subplot for losses
    plt.plot(epochs, train_los, label='Train Loss', marker='', color='blue')
    plt.plot(epochs, val_los, label='Validation Loss', marker='', color='orange')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracy, precision, recall, and F1 score
    plt.subplot(1, 2, 2)  # Subplot for metrics
    plt.plot(epochs, mae_met, label='Mean absolute error', marker='', color='green')
    plt.title('Metrics Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()


def visualize_embeddings_1d(embeddings, labels=None, title="1D Embeddings", is_classification=False):
    embeddings = embeddings.squeeze()  # (N,)
    fig, ax = plt.subplots()

    if is_classification:
        classes = np.unique(labels)
        for c in classes:
            idx = (labels == c)
            ax.scatter(embeddings[idx], np.zeros_like(embeddings[idx]),
                       label=f"Class {c}", alpha=0.7)
        ax.legend()
    else:
        scatter = ax.scatter(embeddings, np.zeros_like(embeddings),
                             c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Target value")

    ax.set_title(title)
    ax.set_xlabel("Embedding Value")
    ax.set_yticks([])
    plt.show()


def visualize_embeddings_2d(embeddings, labels=None, title="2D Embeddings", is_classification=False):
    x = embeddings[:, 0]
    y = embeddings[:, 1]

    fig, ax = plt.subplots()
    if is_classification:
        classes = np.unique(labels)
        for c in classes:
            idx = (labels == c)
            ax.scatter(x[idx], y[idx], label=f"Class {c}", alpha=0.7)
        ax.legend()
    else:
        scatter = ax.scatter(x, y, c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, ax=ax, label="Target value")

    ax.set_title(title)
    ax.set_xlabel("Embedding dim 1")
    ax.set_ylabel("Embedding dim 2")
    plt.show()


def visualize_decision_boundary_2d(predictor, embeddings, labels, title="Decision Boundary"):
    x_min, x_max = embeddings[:, 0].min() - 0.1, embeddings[:, 0].max() + 0.1
    y_min, y_max = embeddings[:, 1].min() - 0.1, embeddings[:, 1].max() + 0.1

    x = embeddings[:, 0]
    y = embeddings[:, 1]

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    grid_tensor = torch.tensor(grid_points, dtype=torch.float)
    with torch.no_grad():
        scores = predictor(grid_tensor)
        preds = scores.argmax(dim=1).numpy()

    preds = preds.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, preds, cmap=plt.cm.coolwarm, alpha=0.3)

    classes = np.unique(labels)
    for c in classes:
        idx = (labels == c)
        ax.scatter(x[idx], y[idx],
                   label=f"Class {c}", alpha=0.8)

    ax.set_title(title)
    ax.legend()
    plt.show()


def visualize_regression_surface_2d(predictor, embeddings, targets, title="Regression Surface"):
    x_min, x_max = embeddings[:, 0].min() - 1.0, embeddings[:, 0].max() + 1.0
    y_min, y_max = embeddings[:, 1].min() - 1.0, embeddings[:, 1].max() + 1.0

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    grid_tensor = torch.tensor(grid_points, dtype=torch.float)
    with torch.no_grad():
        preds = predictor(grid_tensor).numpy().reshape(xx.shape)

    fig, ax = plt.subplots()
    contour = ax.contourf(xx, yy, preds, levels=50, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=ax, label="Predicted value")

    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                         c=targets, cmap='coolwarm', edgecolor='k')
    plt.colorbar(scatter, ax=ax, label="True target")

    ax.set_title(title)
    plt.show()
