import os
import re

import torch.cuda
import torch.optim as optim
from torch import nn

from torch.utils.data import DataLoader
from TextDataset import TextDataset
from project_2.SymbolLSTMModel import SymbolLSTMModel
from project_2.model_functionalities import generate_text, train_and_save_model, evaluate_model

model_path = "symbol_lstm_model.pth"

with open ("pantadeusz.txt", "r", encoding="utf-8") as file:
    unmodified_text: str = file.read()

text: str = re.sub(r'\s+', ' ', unmodified_text)
text: str = text.strip().lower()

unique_symbols: list[str] = sorted(list(set(text)))
unique_symbols_count: int = len(unique_symbols)

symbol_to_index: dict[str, int] = { symbol: i for i, symbol in enumerate(unique_symbols) }
index_to_symbol: dict[int, str] = { i: symbol for i, symbol in enumerate(unique_symbols) }

data_size: int = len(text)
train_size: int = int(0.8 * data_size)
validation_size: int = int(0.1 * data_size)

train_text: str = text[:train_size]
validation_text: str = text[train_size:train_size+validation_size]
test_text: str = text[train_size+validation_size:]

def encode_text(text: str, symbol_to_index: dict[str, int]):
    encoded_text: list = []
    for symbol in text:
        encoded_text.append(symbol_to_index[symbol])

    return encoded_text

train_encoded_text_list: list[str] = encode_text(train_text, symbol_to_index)
validation_encoded_text_list: list[str] = encode_text(validation_text, symbol_to_index)
test_encoded_text_list: list[str] = encode_text(test_text, symbol_to_index)

sequence_length = 30
batch_size = 64

train_dataset = TextDataset(train_encoded_text_list, sequence_length)
validation_dataset = TextDataset(validation_encoded_text_list, sequence_length)
test_dataset = TextDataset(test_encoded_text_list, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

embedding_dimensions = 128
hidden_dimensions = 256
num_layers = 1
learning_rate = 0.001
num_epochs = 50
num_symbols = 200
temperature = 0.5
patience = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SymbolLSTMModel(
    unique_symbols_count=unique_symbols_count,
    embedding_dimensions=embedding_dimensions,
    hidden_dimensions=hidden_dimensions).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
else:
    print("No model found. Training a new model.")
    train_and_save_model(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        model_path=model_path,
        unique_symbols_count=unique_symbols_count,
        patience=patience
    )

test_loss, test_acc, top3_acc = evaluate_model(model, test_loader, unique_symbols_count, device)
print(f"Model Information - test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}, top-3 accuracy: {top3_acc:.4f}")

start_text = "gerwazy"
generated_text = generate_text(
    model=model,
    start_text=start_text,
    symbol_to_index=symbol_to_index,
    index_to_symbol=index_to_symbol,
    sequence_length=sequence_length,
    num_chars=num_symbols,
    temperature=temperature
)

print("Generated Text:\n", generated_text)
