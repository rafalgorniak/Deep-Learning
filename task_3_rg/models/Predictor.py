import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_mlp):
        super(Predictor, self).__init__()

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        self.use_mlp = use_mlp
        if self.use_mlp:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        if self.use_mlp:
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
        else:
            x = self.fc(x)

        return x
