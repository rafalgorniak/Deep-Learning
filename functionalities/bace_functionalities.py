import torch
from torch import optim, nn
from torchmetrics.classification import BinaryAccuracy


def train_bace(model, train_loader, val_loader, class_weights,
               lr=0.0001, max_epochs=100, patience=10, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_val_acc = 0.0
    epochs_no_improve = 0

    accuracy_metric = BinaryAccuracy().to(device)

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        accuracy_metric.reset()

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, emb = model(data)
            labels = data.y.long().view(-1)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs

            preds = out.argmax(dim=1)
            accuracy_metric.update(preds, labels)

        avg_loss = total_loss / len(train_loader.dataset)
        train_acc = accuracy_metric.compute().item()
        accuracy_metric.reset()

        val_loss, val_acc = evaluate_bace(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1} Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    model.load_state_dict(best_model_state)
    return model


def evaluate_bace(model, loader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0

    accuracy_metric = BinaryAccuracy().to(device)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, emb = model(data)
            labels = data.y.long().view(-1)

            loss = criterion(out, labels)
            total_loss += loss.item() * data.num_graphs

            preds = out.argmax(dim=1)
            accuracy_metric.update(preds, labels)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_metric.compute().item()
    accuracy_metric.reset()
    return avg_loss, acc