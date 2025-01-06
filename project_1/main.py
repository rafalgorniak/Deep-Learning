import torch
from torch import nn, optim

from project_1.classes.CNN import CNN
from project_1.classes.CelebAModel import CelebAModel
from project_1.data.data_loader import load_dataset


def main():
    train_data, validation_data, test_data = load_dataset('CelebA', subset_size=0.05)
    print("data loaded...")

    criterion = nn.BCEWithLogitsLoss()
    cnn_model = CNN()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    model = CelebAModel(
        model=cnn_model,
        criterion=criterion,
        optimizer=optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("model fit ...")
    model.fit(train_data, val_loader=validation_data, num_epochs=50)

    predictions, true_labels = model.predict(test_data)
    print("model predict...")

    predicted_classes = (predictions > 0.5).int()
    print(predicted_classes[:5], true_labels[:5])

    accuracy = model.evaluate(test_data)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
