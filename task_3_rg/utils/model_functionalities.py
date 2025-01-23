import torch
import numpy as np
from torchmetrics import Precision, Recall, F1Score
from torchmetrics.classification import BinaryAccuracy
from sklearn.metrics import mean_absolute_error
from torch_geometric.nn import global_mean_pool


def train_evaluate_save_model(model, criterion, optimizer, train_loader, val_loader, epochs, device, best_model_path,
                              patience):
    # Variables for early stopping and model saving
    best_val_loss = float('inf')  # Initialize with infinity
    patience_counter = 0

    train_los = []
    val_los = []
    acc_metr = []
    prec_metr = []
    recc_metr = []
    f1_metr = []

    accuracy_metric = BinaryAccuracy().to(device)
    precision_metric = Precision(task="binary").to(device)
    recall_metric = Recall(task="binary").to(device)
    f1_metric = F1Score(task="binary").to(device)

    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc, train_precision, train_recall, train_f1 = (
            train(model, train_loader, criterion, optimizer, device,
                accuracy_metric, precision_metric, recall_metric, f1_metric))
        val_loss = evaluate(model, val_loader, criterion, device)

        train_los.append(train_loss)
        val_los.append(val_loss)
        acc_metr.append(train_acc)
        prec_metr.append(train_precision)
        recc_metr.append(train_recall)
        f1_metr.append(train_f1)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check if the validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epochs.")

        # Early stopping condition
        if patience_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

    print("Training complete. Best model saved to", best_model_path)

    return train_los, val_los, acc_metr, prec_metr, recc_metr, f1_metr


def train(model, train_loader, criterion, optimizer, device,
          accuracy_metric, precision_metric, recall_metric, f1_metric):
    model.train()
    total_loss = 0

    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    for data in train_loader:
        data = data.to(device)
        label = data.y.view(-1).long().to(device)

        optimizer.zero_grad()
        out, emb = model(data)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = out.argmax(dim=1)
        accuracy_metric.update(preds, label)
        precision_metric.update(preds, label)
        recall_metric.update(preds, label)
        f1_metric.update(preds, label)

    train_acc = accuracy_metric.compute().item()
    train_precision = precision_metric.compute().item()
    train_recall = recall_metric.compute().item()
    train_f1 = f1_metric.compute().item()

    train_loss = total_loss / len(train_loader)


    return train_loss, train_acc, train_precision, train_recall, train_f1


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            label = data.y.long().view(-1).to(device)

            out, emb = model(data)
            loss = criterion(out, label)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def train_evaluate_save_model_regression(model, criterion, optimizer, train_loader, val_loader, epochs, device, best_model_path,
                              patience):
    # Variables for early stopping and model saving
    best_val_loss = float('inf')  # Initialize with infinity
    patience_counter = 0

    train_los = []
    val_los = []
    mae_met = []

    # Training loop
    for epoch in range(epochs):
        train_loss, mae_metric = train_regression(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_regression(model, val_loader, criterion, device)

        train_los.append(train_loss)
        val_los.append(val_loss)
        mae_met.append(mae_metric)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, MAE metrics: {mae_metric:.3f}")

        # Check if the validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epochs.")

        # Early stopping condition
        if patience_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break

    print("Training complete. Best model saved to", best_model_path)

    return train_los, val_los, mae_met

def train_regression(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []

    for data in train_loader:
        data = data.to(device)
        label = data.y[:, 4].view(-1).to(device)

        optimizer.zero_grad()
        out, emb = model(data)
        loss = criterion(out.view(-1), label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Zbieranie wyników do obliczenia MAE
        all_labels.append(label.cpu().numpy())
        all_preds.append(out.view(-1).detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    mae = mean_absolute_error(all_labels, all_preds)

    train_loss = total_loss / len(train_loader)


    return train_loss, mae


def evaluate_regression(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            label = data.y[:, 4].view(-1).to(device)

            out, emb = model(data)
            loss = criterion(out.view(-1), label)
            total_loss += loss.item()

    return total_loss / len(val_loader)



def get_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Przewidujemy embeddingi i wyniki
            out, emb = model(data)  # emb ma kształt [batch_size, embedding_dim]
            embeddings.append(emb.cpu())

            # Przypisanie etykiet dla każdego grafu
            # Sprawdzamy wymiar `data.y`, by dopasować do grafów
            if len(data.y.size()) == 1:  # Zakładamy, że `data.y` jest wektorem
                graph_labels = data.y  # Nie trzeba nic zmieniać
            else:  # Jeśli `data.y` jest macierzą
                graph_labels = data.y[:, 0]  # Pobieramy pierwszy wymiar (przykład)

            labels.append(graph_labels.cpu())

    # Łączymy wszystkie batchy w jedną macierz embeddingów i jedną listę etykiet
    embeddings = torch.cat(embeddings, dim=0).numpy()  # Rozmiar: [num_graphs, embedding_dim]
    labels = torch.cat(labels, dim=0).numpy()  # Rozmiar: [num_graphs]

    print(f"Embeddings shape: {embeddings.shape}, Labels shape: {labels.shape}")
    return embeddings, labels

