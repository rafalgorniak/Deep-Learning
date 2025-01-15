import os
import torch.cuda
import torch.optim as optim
from torch import nn

from torch.utils.data import DataLoader
from TextDataset import TextDataset
from project_2.SymbolLSTMModel import SymbolLSTMModel
from project_2.SymbolTransformerModel import SymbolTransformerModel
from project_2.model_functionalities import generate_text, train_and_save_model, evaluate_model
from project_2.text_handling import preprocess_text, split_text, encode_text

# Params
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "symbol_lstm_model.pth"
sequence_length = 30
batch_size = 64
embedding_dimensions = 128
hidden_dimensions = 256
num_layers = 1
learning_rate = 0.001
num_epochs = 50
num_symbols = 200
temperature = 0.5
patience = 2

text, unique_symbols, symbol_to_index, index_to_symbol = preprocess_text("pantadeusz.txt")
unique_symbols_count = len(unique_symbols)

train_text, validation_text, test_text = split_text(text)

train_encoded_text_list: list[str] = encode_text(train_text, symbol_to_index)
validation_encoded_text_list: list[str] = encode_text(validation_text, symbol_to_index)
test_encoded_text_list: list[str] = encode_text(test_text, symbol_to_index)

train_dataset = TextDataset(train_encoded_text_list, sequence_length)
validation_dataset = TextDataset(validation_encoded_text_list, sequence_length)
test_dataset = TextDataset(test_encoded_text_list, sequence_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = SymbolLSTMModel(
    unique_symbols_count=unique_symbols_count,
    embedding_dimensions=embedding_dimensions,
    hidden_dimensions=hidden_dimensions).to(device)

#model = SymbolTransformerModel(
#    unique_symbols_count=unique_symbols_count,
#    embedding_dimensions=embedding_dimensions,
#    num_layers=num_layers,
#    num_heads=4,
#    dropout_rate=0.2
#).to(device)

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
