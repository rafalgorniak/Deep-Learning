import torch

def train_and_save_model(model, train_loader, criterion, optimizer, num_epochs, model_path, unique_symbols_count):
    device = next(model.parameters()).device

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, hidden = model(x)
            logits = logits.view(-1, unique_symbols_count)
            y = y.view(-1)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.2f}")

        torch.save(model.state_dict(), model_path)
        print(f"Current model saved to {model_path}")


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