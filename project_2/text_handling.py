import re


def preprocess_text(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        unmodified_text = file.read()

    text = re.sub(r'\s+', ' ', unmodified_text).strip().lower()
    unique_symbols = sorted(list(set(text)))

    symbol_to_index = {symbol: i for i, symbol in enumerate(unique_symbols)}
    index_to_symbol = {i: symbol for i, symbol in enumerate(unique_symbols)}

    return text, unique_symbols, symbol_to_index, index_to_symbol


def split_text(text: str, train_ratio=0.8, val_ratio=0.1):
    data_size = len(text)
    train_end = int(train_ratio * data_size)
    val_end = train_end + int(val_ratio * data_size)

    train_text = text[:train_end]
    validation_text = text[train_end:val_end]
    test_text = text[val_end:]

    return train_text, validation_text, test_text


def encode_text(text: str, symbol_to_index: dict[str, int]):
    encoded_text: list = []
    for symbol in text:
        encoded_text.append(symbol_to_index[symbol])

    return encoded_text
