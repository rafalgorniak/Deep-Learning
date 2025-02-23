import torch
import torch.nn.functional as F


def train_evaluate_save_model(model, criterion, optimizer, train_loader, val_loader, epochs, device, best_model_path,
                              patience):
    # Variables for early stopping and model saving
    best_val_loss = float('inf')  # Initialize with infinity
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

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


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)  # outputs: (batch_size, sequence_length, vocab_size)

            # Spłaszcz outputs i targets
            outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * sequence_length, vocab_size)
            targets = targets.view(-1)  # (batch_size * sequence_length)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def generate_text(model, start_phrase, symbol_to_index, index_to_symbol, sequence_length, generate_length, device, top_k=5):
    model.eval()  # Set the model to evaluation mode

    # Convert start_phrase to indices
    input_indices = [symbol_to_index[char] for char in start_phrase.lower()]

    # Pad or truncate to match sequence_length
    if len(input_indices) < sequence_length:
        input_indices = [symbol_to_index[' ']] * (sequence_length - len(input_indices)) + input_indices
    else:
        input_indices = input_indices[-sequence_length:]

    # Convert to a tensor
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)

    generated_text = start_phrase  # Initialize generated text

    for _ in range(generate_length):
        # Forward pass through the model
        with torch.no_grad():
            output = model(input_tensor)

        # Get the prediction for the next character (top-k sampling)
        next_token_logits = output[0, -1, :]  # Take the last timestep's output
        top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k)
        probs = F.softmax(top_k_logits, dim=-1)
        next_token_idx = top_k_indices[torch.multinomial(probs, num_samples=1).item()]

        # Append the predicted character to the generated text
        next_char = index_to_symbol[next_token_idx.item()]
        generated_text += next_char

        # Update the input sequence
        input_indices = input_indices[1:] + [next_token_idx.item()]  # Slide window
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)

    return generated_text


def evaluate_model_on_test(test_loader, model, criterion, symbol_to_index, index_to_symbol, sequence_length, top_k=5,
                           device='cpu'):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_top_k_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)  # outputs: (batch_size, sequence_length, vocab_size)

            # Spłaszcz outputs i targets
            outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * sequence_length, vocab_size)
            targets = targets.view(-1)  # (batch_size * sequence_length)

            # Oblicz stratę
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Obliczenie dokładności
            predicted = torch.argmax(outputs, dim=-1)  # (batch_size * sequence_length)
            total_correct += (predicted == targets).sum().item()

            # Obliczenie top-k dokładności
            top_k_predictions = torch.topk(outputs, k=top_k, dim=-1).indices  # (batch_size * sequence_length, top_k)
            for i in range(targets.size(0)):  # Iteracja po wszystkich tokenach
                if targets[i].item() in top_k_predictions[i].tolist():
                    total_top_k_correct += 1

            total_tokens += targets.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_tokens
    top_k_accuracy = total_top_k_correct / total_tokens

    return avg_loss, accuracy, top_k_accuracy
