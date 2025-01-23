import torch.nn as nn

from task_3_rg.models.Embedder_regression import Embedder_regression
from task_3_rg.models.Predictor import Predictor


class QM9Model(nn.Module):
    def __init__(self, in_channels = 11, hidden_dim = 64, embedding_dim = 2, output_dim = 1, num_layers = 3, use_mlp = False):
        super(QM9Model, self).__init__()

        self.embedder = Embedder_regression(in_channels, hidden_dim, embedding_dim, num_layers)
        self.predictor = Predictor(embedding_dim, hidden_dim, output_dim, use_mlp)

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        emb = self.embedder(x, edge_index, batch, edge_attr=edge_attr)
        out = self.predictor(emb)

        return out, emb
