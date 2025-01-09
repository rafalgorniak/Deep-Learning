import os

import torch
from torch import nn, optim

from project_1.cameraTest.camera import test_with_camera
from project_1.classes.CNN import CNN
from project_1.classes.Model import Model
from project_1.data.data_loader import load_dataset


def main():

    criterion = nn.BCEWithLogitsLoss()
    cnn_model = CNN()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    model = Model(
        model=cnn_model,
        criterion=criterion,
        optimizer=optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    model_path = "./models/gender_model.pth"

    print("Data loaded...")
    train_data, validation_data, test_data = load_dataset('CelebA', subset_size=0.02)

    if os.path.exists(model_path):
        print("Loading existing model...")
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        print("Training model...")
        model.fit(train_data, val_loader=validation_data, num_epochs=20)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Ewaluacja na zbiorze testowym CelebA
    print("Evaluating model on CelebA dataset...")
    accuracy = model.evaluate(test_data)
    print(f"Test Accuracy for CelebA dataset : {accuracy * 100:.2f}%")

    # Ewaluacja na zbiorze testowym WiderFace
    print("Evaluating model on WIDERFace dataset...")
    test_data = load_dataset('WIDERFace')[2]
    accuracy = model.evaluate(test_data)
    print(f"Test Accuracy for WIDERFace dataset : {accuracy * 100:.2f}%")

    # Test wizualny dla camery
    test_with_camera(model)


if __name__ == '__main__':
    main()
