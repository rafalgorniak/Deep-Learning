import torch
import torchmetrics
from torch import nn


def train_and_save_model(model, train_loader, val_loader, criterion, optimizer,
                         num_epochs, model_path, unique_symbols_count, patience=2):
    device = next(model.parameters()).device
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            logits = logits.view(-1, unique_symbols_count)
            y = y.view(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_train_loss:.2f}")

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_logits, _ = model(x_val)
                val_logits = val_logits.view(-1, unique_symbols_count)
                y_val = y_val.view(-1)
                val_loss = criterion(val_logits, y_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {avg_val_loss:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else:
            epochs_no_improve += 1
            print(f"Lack of improvement: {epochs_no_improve} epochs")
            if epochs_no_improve >= patience:
                print("Early stopping")
                break


def generate_text(model, start_text: str, symbol_to_index: dict[str, int], index_to_symbol: dict[int, str],
        sequence_length: int, num_chars: int, temperature: float = 1.0) -> str:
    model.eval()
    device = next(model.parameters()).device

    input_indices = [symbol_to_index[symbol] for symbol in start_text]
    generated_text = start_text

    for _ in range(num_chars):
        if len(input_indices) < sequence_length:
            input_seq = [0] * (sequence_length - len(input_indices)) + input_indices
        else:
            input_seq = input_indices[-sequence_length:]

        input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)

        with torch.no_grad():
            logits, _ = model(input_tensor)
            logits = logits[:, -1, :]
            logits = logits / temperature

            probabilities = torch.softmax(logits, dim=-1).squeeze()

            next_index = torch.multinomial(probabilities, num_samples=1).item()

        next_symbol = index_to_symbol[next_index]
        generated_text += next_symbol
        input_indices.append(next_index)

    return generated_text

def evaluate_model(model, data_loader, unique_symbols_count, device):
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=unique_symbols_count).to(device)
    top3_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=unique_symbols_count, top_k=3).to(device)

    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            logits = logits.view(-1, unique_symbols_count)
            y = y.view(-1)

            loss = criterion(logits, y)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            accuracy.update(preds, y)
            top3_accuracy.update(logits, y)

    avg_loss = total_loss / len(data_loader)
    acc = accuracy.compute()
    top3_acc = top3_accuracy.compute()

    return avg_loss, acc.item(), top3_acc.item()