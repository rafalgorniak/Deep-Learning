import torch
from torch import nn


class NoseModel(nn.Module):
    def __init__(self, model, criterion, optimizer, device='cpu'):
        super(NoseModel, self).__init__()
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def fit(self, train_loader, val_loader=None, num_epochs=10):
        self.model.train()

        patience = 3
        min_delta = 0.002
        patience_counter = 0
        best_val_accuracy = float('-inf')

        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # Forward
                outputs = self.model(images)

                labels = labels.unsqueeze(1).float()
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

            if val_loader:
                val_accuracy = self.evaluate(val_loader)
                print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

                if val_accuracy > best_val_accuracy + min_delta:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break

    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward
                outputs = self.model(images)
                predictions.append(torch.sigmoid(outputs).cpu())
                true_labels.append(labels.cpu())

        # Zwracamy predykcje i prawdziwe etykiety jako tensory
        return torch.cat(predictions), torch.cat(true_labels)

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1).float()  # Ensure shape compatibility

                # Forward pass
                outputs = self.model(images)
                predictions = (torch.sigmoid(outputs) > 0.5).float()  # Convert to 0/1

                # Accuracy calculation
                correct += (predictions == labels).sum().item()
                total += labels.size(0)  # Count total samples

        return correct / total  # Return accuracy as a fraction

    def probability(self, images):
        self.model.eval()

        with torch.no_grad():
            # Ensure images are on the correct device
            images = images.to(self.device)

            # Forward pass
            outputs = self.model(images)
            predictions = torch.sigmoid(outputs)

        return predictions
