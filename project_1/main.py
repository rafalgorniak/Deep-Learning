import os

import torch
from torch import nn, optim

from project_1.cameraTest.camera import test_with_camera
from project_1.classes.CNN import CNN
from project_1.classes.GenderModel import GenderModel
from project_1.classes.NoseModel import NoseModel
from project_1.classes.ResNet import ResNet
from project_1.data.data_loader import load_dataset
from project_1.utils.confusion_matrix import draw_confusion_matrix
from project_1.utils.count_classes import count_classes


def main(mode: int):

    if mode == 0:
        # Model 1
        cnn_model = CNN()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

        model = GenderModel(
            model=cnn_model,
            criterion=criterion,
            optimizer=optimizer,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        model_path = "./models/gender_model.pth"

        print("Data loaded...")
        train_data, validation_data, test_data = load_dataset('CelebA', subset_size=0.2)

        if os.path.exists(model_path):
            print("Loading existing model...")
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            print("Training model...")
            model.fit(train_data, val_loader=validation_data, num_epochs=50)
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        # Ewaluacja na zbiorze testowym CelebA
        print("Evaluating model on CelebA dataset...")
        accuracy = model.evaluate(test_data)
        print(f"Test Accuracy for CelebA dataset : {accuracy * 100:.2f}%")

        y_pred, y_true = model.predict(test_data)
        y_pred = (y_pred > 0.5).float()

        draw_confusion_matrix(y_true, y_pred, 'Gender detection for CelebA using own CNN',
                              'Predicted Label (0 - male, 1 - female)', 'True Label (0 - male, 1 - female)')

        # Ewaluacja na zbiorze testowym WiderFace
        print("Evaluating model on WIDERFace dataset...")
        test_data = load_dataset('WIDERFace')[2]
        accuracy = model.evaluate(test_data)
        print(f"Test Accuracy for WIDERFace dataset : {accuracy * 100:.2f}%")

        y_pred, y_true = model.predict(test_data)
        y_pred = (y_pred < 0.5).float()

        draw_confusion_matrix(y_true, y_pred, 'Gender detection for widerFace using own CNN',
                              'Predicted Label (0 - male, 1 - female)', 'True Label (0 - male, 1 - female)')

    elif mode == 1:
        # Model 2
        resnet_model = ResNet()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(resnet_model.resnet.fc.parameters(), lr=0.002)

        model = NoseModel(
            model=resnet_model,
            criterion=criterion,
            optimizer=optimizer,
            device='cuda' if torch.cuda.is_available() else 'cpu')

        model_path = "models/nose_model_balanced.pth"

        print("Data loaded...")
        # train_data, validation_data, test_data = load_dataset('CelebA', subset_size=0.05)
        train_data, validation_data, test_data = load_dataset('balanced_celebA', subset_size=0.05)

        count_classes(train_data, validation_data, test_data)

        if os.path.exists(model_path):
            print("Loading existing pre-trained model...")
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            print("Training pre-trained model...")
            model.fit(train_data, val_loader=validation_data, num_epochs=50)
            torch.save(model.state_dict(), model_path)
            print(f"Pre-trained model saved to {model_path}")

        # Ewaluacja na zbiorze testowym CelebA
        print("Evaluating pre-trained model on CelebA dataset...")
        accuracy = model.evaluate(test_data)
        print(f"Test Accuracy for CelebA dataset : {accuracy * 100:.2f}%")

        y_pred, y_true = model.predict(test_data)
        y_pred = (y_pred > 0.5).float()

        draw_confusion_matrix(y_true, y_pred, 'Big Nose detection for CelebA with ResNet',
                              'Predicted Label (0 - small nose, 1 - big nose)', 'True Label (0 - small nose, 1 - big nose)')

        # Ewaluacja na zbiorze testowym WiderFace
        print("Evaluating pre-trained model on WIDERFace dataset...")
        test_data = load_dataset('WIDERFace')[2]
        accuracy = model.evaluate(test_data)
        print(f"Test Accuracy for WIDERFace dataset : {accuracy * 100:.2f}%")

        y_pred, y_true = model.predict(test_data)
        y_pred = (y_pred > 0.5).float()

        draw_confusion_matrix(y_true, y_pred, 'Big Nose detection for widerFace with ResNet',
                              'Predicted Label (0 - small nose, 1 - big nose)', 'True Label (0 - small nose, 1 - big nose)')

    elif mode == 2:
        # Path only for laboratory improvement
        # Model 1
        cnn_model = CNN()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.0001)

        model_gender = GenderModel(
            model=cnn_model,
            criterion=criterion,
            optimizer=optimizer,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        model_path_gender = "./models/gender_model.pth"

        print("Loading existing model...")
        model_gender.load_state_dict(torch.load(model_path_gender))
        model_gender.eval()

        # Ewaluacja na zbiorze testowym WiderFace
        print("Evaluating model on WIDERFace dataset...")
        test_data = load_dataset('WIDERFace')[2]
        accuracy = model_gender.evaluate(test_data)
        print(f"Test Accuracy for WIDERFace dataset : {accuracy * 100:.2f}%")

        y_pred, y_true = model_gender.predict(test_data)
        y_pred = (y_pred > 0.5).float()

        draw_confusion_matrix(y_true, y_pred, 'Gender detection for widerFace using own CNN',
                              'Predicted Label (0 - male, 1 - female)', 'True Label (0 - male, 1 - female)')

        # Model 2
        resnet_model = ResNet()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(resnet_model.resnet.fc.parameters(), lr=0.0001)

        model_nose = NoseModel(
            model=resnet_model,
            criterion=criterion,
            optimizer=optimizer,
            device='cuda' if torch.cuda.is_available() else 'cpu')

        model_path_nose = "./models/nose_model.pth"

        print("Loading existing pre-trained model...")
        model_nose.load_state_dict(torch.load(model_path_nose))
        model_nose.eval()

        # Ewaluacja na zbiorze testowym WiderFace
        print("Evaluating pre-trained model on WIDERFace dataset...")
        test_data = load_dataset('WIDERFace')[2]
        accuracy = model_nose.evaluate(test_data)
        print(f"Test Accuracy for WIDERFace dataset : {accuracy * 100:.2f}%")

        y_pred, y_true = model_nose.predict(test_data)
        y_pred = (y_pred > 0.5).float()

        draw_confusion_matrix(y_true, y_pred, 'Big Nose detection for widerFace with ResNet',
                              'Predicted Label (0 - small nose, 1 - big nose)',
                              'True Label (0 - small nose, 1 - big nose)')

    # Test wizualny dla camery
    # test_with_camera(model, model_)


if __name__ == '__main__':
    main(1)
