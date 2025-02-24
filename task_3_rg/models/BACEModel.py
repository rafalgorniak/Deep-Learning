import torch.nn as nn

from task_3_rg.models.Embedder import Embedder
from task_3_rg.models.Embedder_Transformer import Embedder_Transformer
from task_3_rg.models.Predictor import Predictor


class BACEModel(nn.Module):
    def __init__(self, in_channels = 9, hidden_dim = 64, embedding_dim = 2, output_dim = 2, num_layers = 3, use_mlp = False):
        super(BACEModel, self).__init__()

        self.embedder = Embedder(in_channels, hidden_dim, embedding_dim, num_layers)
        self.embedder = Embedder_Transformer(in_channels, hidden_dim, embedding_dim, num_layers, transformer_heads=1)

        self.predictor = Predictor(embedding_dim, hidden_dim, output_dim, use_mlp)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        emb = self.embedder(x, edge_index, batch)
        out = self.predictor(emb)

        return out, emb
