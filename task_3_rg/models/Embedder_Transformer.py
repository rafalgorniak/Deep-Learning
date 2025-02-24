import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, global_mean_pool


class Embedder_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers, transformer_heads=1):
        super(Embedder_Transformer, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # Upewnij się, że embedding_dim jest podzielne przez transformer_heads
        assert embedding_dim % transformer_heads == 0, "embedding_dim must be divisible by transformer_heads"

        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(input_dim, hidden_dim, heads=transformer_heads))

        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim, heads=transformer_heads))

        if num_layers > 1:
            self.convs.append(TransformerConv(hidden_dim, embedding_dim, heads=transformer_heads))
        else:
            self.convs[0] = TransformerConv(input_dim, embedding_dim, heads=transformer_heads)

    def forward(self, x, edge_index, batch):
        x = x.float()

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = self.relu(x)
                x = self.dropout(x)

        x = global_mean_pool(x, batch)

        return x