from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # Load the pretrained ResNet model with updated syntax
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Modify the final fully connected layer for 2 classes
        num_features = resnet.fc.in_features
        resnet.fc = nn.Linear(num_features, 1)

        # Freeze all layers except the last fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False

        for param in resnet.fc.parameters():
            param.requires_grad = True

        # Save the modified ResNet model as a submodule
        self.resnet = resnet

    def forward(self, x):
        return self.resnet(x)  # Forward pass through the underlying resnet