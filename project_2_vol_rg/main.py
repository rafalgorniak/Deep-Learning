import os
import torch.cuda

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from project_2_vol_rg.classes.TextDataSet import TextDataSet
from project_2_vol_rg.models.LSTMModel import LSTMModel
from project_2_vol_rg.models.TransformerModel import TransformerModel
from project_2_vol_rg.utils.data_utils import read_text_file, split_text, decode_text
from project_2_vol_rg.utils.model_functions import train_evaluate_save_model, generate_text, evaluate_model_on_test


def main():
    # constants
    input_text_path = './data/pantadeusz.txt'
    # input_text_path = './data/ksiega_pierwsza.txt'
    model_path = './saved_models/LSTM_model.pth'

    embedding_dim = 128
    hidden_dim = 256
    lstm_layers = 2
    batch_size = 64

    sequence_length = 50
    epoch_number = 20
    learning_rate = 0.001
    patience = 3

    n_head = 8
    num_layers = 4

    text, dictionary, symbol_to_index, index_to_symbol = read_text_file(input_text_path)
    dictionary_size: int = len(dictionary)

    train_text, validation_text, test_text = split_text(text, text_ratio=1)

    train_text_decoded: list[str] = decode_text(train_text, symbol_to_index)
    validation_text_decoded: list[str] = decode_text(validation_text, symbol_to_index)
    test_text_decoded: list[str] = decode_text(test_text, symbol_to_index)

    train_dataset: TextDataSet = TextDataSet(train_text_decoded, sequence_length)
    validation_dataset: TextDataSet = TextDataSet(validation_text_decoded, sequence_length)
    test_dataset: TextDataSet = TextDataSet(test_text_decoded, sequence_length)

    train_loader: DataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader: DataLoader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader: DataLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LSTMModel(
        vocab_size=dictionary_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=lstm_layers,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    else:
        print("No model found. Training a new model...")
        train_evaluate_save_model(model, criterion, optimizer, train_loader=train_loader, val_loader=validation_loader,
                                  epochs=epoch_number, device=device, best_model_path=model_path, patience=patience)

    start_phrase = "Ksiądz robak"
    generate_length = 200

    generated_text = generate_text(
        model=model,
        start_phrase=start_phrase,
        symbol_to_index=symbol_to_index,
        index_to_symbol=index_to_symbol,
        sequence_length=sequence_length,
        generate_length=generate_length,
        device=device
    )

    print("Generated text:", generated_text)

    avg_loss, accuracy, top_k_accuracy = evaluate_model_on_test(
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        symbol_to_index=symbol_to_index,
        index_to_symbol=index_to_symbol,
        sequence_length=sequence_length,
        top_k=5,
        device=device
    )

    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Top-5 Accuracy: {top_k_accuracy:.4f}")


if __name__ == "__main__":
    main()
