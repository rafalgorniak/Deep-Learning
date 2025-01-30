from collections import Counter


def count_classes(train_data, validation_data, test_data):
    """
    Funkcja do zliczania liczności klas w zbiorze danych.
    """
    counts = Counter()

    for _, labels in train_data:
        counts.update(labels.cpu().numpy())

    print(f"Liczność klas w zbiorze treningowym: {dict(counts)}")

    counts = Counter()

    for _, labels in validation_data:
        counts.update(labels.cpu().numpy())

    print(f"Liczność klas w zbiorze walidacyjnym: {dict(counts)}")

    counts = Counter()

    for _, labels in test_data:
        counts.update(labels.cpu().numpy())

    print(f"Liczność klas w zbiorze testowym: {dict(counts)}")
