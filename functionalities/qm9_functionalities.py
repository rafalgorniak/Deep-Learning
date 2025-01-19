import torch
from torch import nn, optim
from torchmetrics import MeanAbsoluteError


def train_qm9(model,
              train_loader,
              val_loader,
              lr=1e-3,
              max_epochs=100,
              patience=10,
              device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        total_graphs = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out, emb = model(data)
            labels = data.y.float().view(-1, 1)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            total_graphs += data.num_graphs

        train_mse = total_loss / total_graphs

        val_mse, val_mae = evaluate_qm9(model, val_loader, device)
        print(f"Epoch {epoch+1} | Train MSE: {train_mse:.4f} | "
              f"Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}")

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model


def evaluate_qm9(model, loader, device='cpu'):
    model.eval()
    mse_criterion = nn.MSELoss()
    mae_metric = MeanAbsoluteError().to(device)

    total_mse = 0.0
    total_graphs = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, emb = model(data)
            labels = data.y.float().view(-1, 1)

            mse_loss = mse_criterion(out, labels)
            total_mse += mse_loss.item() * data.num_graphs
            total_graphs += data.num_graphs

            mae_metric.update(out, labels)

    avg_mse = total_mse / total_graphs
    avg_mae = mae_metric.compute().item()
    mae_metric.reset()
    return avg_mse, avg_mae
