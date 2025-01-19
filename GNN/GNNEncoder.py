import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, TransformerConv, GCNConv, SAGEConv


class GNNEncoder(nn.Module):
    def __init__(self,
                 num_features,
                 embedding_dim=2,
                 gnn_layer_type='GCNConv',
                 num_gnn_layers=2):
        super(GNNEncoder, self).__init__()
        self.convs = nn.ModuleList()

        GNNLayer = {
            'GCNConv': GCNConv,
            'SAGEConv': SAGEConv,
            'TransformerConv': TransformerConv
        }[gnn_layer_type]

        hidden_dim = 64
        self.convs.append(GNNLayer(num_features, hidden_dim))
        for _ in range(num_gnn_layers - 2):
            self.convs.append(GNNLayer(hidden_dim, hidden_dim))

        if num_gnn_layers > 1:
            self.convs.append(GNNLayer(hidden_dim, embedding_dim))
        else:
            self.convs[0] = GNNLayer(num_features, embedding_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = x.float()

        if edge_attr is not None:
            edge_attr = edge_attr.float()

        for i, conv in enumerate(self.convs):
            if isinstance(conv, TransformerConv):
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                x = F.relu(x)

        x = global_mean_pool(x, batch)
        return x