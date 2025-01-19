from torch import nn

from models.gnn_encoder import GNNEncoder
from models.predictors import LinearPredictor, MLP


class GNNModel(nn.Module):
    def __init__(self,
                 num_features,
                 embedding_dim=2,
                 gnn_layer_type='GCNConv',
                 num_gnn_layers=2,
                 predictor_type='linear',
                 hidden_dim=64,
                 output_dim=1):

        super(GNNModel, self).__init__()

        self.encoder = GNNEncoder(num_features,
                                  embedding_dim=embedding_dim,
                                  gnn_layer_type=gnn_layer_type,
                                  num_gnn_layers=num_gnn_layers)

        if predictor_type == 'linear':
            self.predictor = LinearPredictor(embedding_dim, output_dim)
        else:
            self.predictor = MLP(embedding_dim, hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

        emb = self.encoder(x, edge_index, batch, edge_attr=edge_attr)
        out = self.predictor(emb)

        return out, emb