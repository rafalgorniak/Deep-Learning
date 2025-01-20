import re


def read_text_file(path_to_file: str):
    # opening and reading the file
    raw_text = open(path_to_file, 'r', encoding='utf-8').read()
    raw_text = raw_text.lower()

    # getting rid of redundant spaces and special signs
    text = re.sub(r'\s+', ' ', raw_text).strip()

    # getting unique chars and two-way dict
    unique_chars = sorted(set(text))
    char_to_int_dict = dict((c, i) for i, c in enumerate(unique_chars))
    int_to_char_dict = dict((i, c) for i, c in enumerate(unique_chars))

    print("Total Characters:", len(text))
    print("Total Vocab:", len(unique_chars))

    return raw_text, unique_chars, char_to_int_dict, int_to_char_dict


def decode_text(text: str, decoder: dict):
    decoded_text = [decoder[char] for char in text if char in decoder]
    return decoded_text


def encode_text(text: str, encoder: dict):
    encoded_text = [encoder[char] for char in text if char in encoder]
    return encoded_text


def list_to_string(text_list: list):
    string = " ".join(text_list)
    return string


def split_text(text: str, train_rate: float = 0.8, validation_rate: float = 0.1, text_ratio: float = 1.0):
    text_length = len(text)
    train_text_end_index = int(text_length * train_rate * text_ratio)
    validation_text_end_index = int(train_text_end_index + text_length * validation_rate * text_ratio)
    test_text_end_index = int(text_length * text_ratio)

    train_text = text[: train_text_end_index]
    validation_text = text[train_text_end_index: validation_text_end_index]
    test_text = text[validation_text_end_index: test_text_end_index]

    print(f"Characters for train text : {len(train_text)}, validation text: {len(validation_text)}, test text: {len(test_text)})")

    return train_text, validation_text, test_text
