import torch


def train_evaluate_save_model_transformer(model, criterion, optimizer, train_loader, val_loader, epochs, device, best_model_path, patience):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss = evaluate_model_transformer(model, criterion, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

def evaluate_model_transformer(model, criterion, loader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_model_on_test_transformer(test_loader, model, criterion, symbol_to_index, index_to_symbol, sequence_length, top_k, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    top_k_correct = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=-1)
            total += targets.numel()
            correct += (predicted == targets).sum().item()

            top_k_preds = torch.topk(outputs, k=top_k, dim=-1).indices
            top_k_correct += sum([targets[i] in top_k_preds[i] for i in range(targets.size(0))])

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    top_k_accuracy = top_k_correct / total

    return avg_loss, accuracy, top_k_accuracy

def generate_text_transformer(model, start_phrase, symbol_to_index, index_to_symbol, sequence_length, generate_length, device):
    model.eval()
    generated_indices = [symbol_to_index[char] for char in start_phrase]

    for _ in range(generate_length):
        input_tensor = torch.tensor(generated_indices[-sequence_length:], device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated_indices.append(next_token)

    generated_text = ''.join([index_to_symbol[idx] for idx in generated_indices])