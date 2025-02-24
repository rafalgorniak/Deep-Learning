import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers):
        super(Embedder, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, embedding_dim))
        else:
            self.convs[0] = GCNConv(input_dim, embedding_dim)

    def forward(self, x, edge_index, batch):
        x = x.float()

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = self.relu(x)
                x = self.dropout(x)

        x = global_mean_pool(x, batch)

        return x
