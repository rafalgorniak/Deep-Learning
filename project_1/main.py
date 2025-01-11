import os

import torch
from torch import nn, optim

from project_1.cameraTest.camera import test_with_camera
from project_1.classes.CNN import CNN
from project_1.classes.GenderModel import GenderModel
from project_1.classes.NoseModel import NoseModel
from project_1.classes.ResNet import ResNet
from project_1.data.data_loader import load_dataset


def main():

    # Model 1
    cnn_model = CNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.0001)

    model = GenderModel(
        model=cnn_model,
        criterion=criterion,
        optimizer=optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    model_path = "./models/gender_model.pth"

    print("Data loaded...")
    train_data, validation_data, test_data = load_dataset('CelebA', subset_size=0.05)

    if os.path.exists(model_path):
        print("Loading existing model...")
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        print("Training model...")
        model.fit(train_data, val_loader=validation_data, num_epochs=30)
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

    # Model 2
    resnet_model = ResNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(resnet_model.resnet.fc.parameters(), lr=0.0001)

    model_ = NoseModel(
        model=resnet_model,
        criterion=criterion,
        optimizer=optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu')

    model_path = "./models/nose_model.pth"

    print("Data loaded...")
    train_data, validation_data, test_data = load_dataset('CelebA', subset_size=0.05)

    if os.path.exists(model_path):
        print("Loading existing pre-trained model...")
        model_.load_state_dict(torch.load(model_path))
        model_.eval()
    else:
        print("Training pre-trained model...")
        model_.fit(train_data, val_loader=validation_data, num_epochs=30)
        torch.save(model_.state_dict(), model_path)
        print(f"Pre-trained model saved to {model_path}")

    # Ewaluacja na zbiorze testowym CelebA
    print("Evaluating pre-trained model on CelebA dataset...")
    accuracy = model_.evaluate(test_data)
    print(f"Test Accuracy for CelebA dataset : {accuracy * 100:.2f}%")

    # Ewaluacja na zbiorze testowym WiderFace
    print("Evaluating pre-trained model on WIDERFace dataset...")
    test_data = load_dataset('WIDERFace')[2]
    accuracy = model_.evaluate(test_data)
    print(f"Test Accuracy for WIDERFace dataset : {accuracy * 100:.2f}%")

    # Test wizualny dla camery
    test_with_camera(model, model_)


if __name__ == '__main__':
    main()
