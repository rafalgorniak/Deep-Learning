import torch
import torch.nn as nn

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from functionalities.bace_functionalities import train_bace, evaluate_bace
from functionalities.shared_functionalities import train_val_test_split, compute_class_weights
from models.gnn_model import GNNModel
from utils.embeddings_visualisation import visualize_embeddings_2d, \
    visualize_embeddings_1d, visualize_decision_boundary_2d

batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

bace_dataset = MoleculeNet(root='data', name='BACE')

bace_train_dataset, bace_val_dataset, bace_test_dataset = train_val_test_split(bace_dataset)

bace_train_loader = DataLoader(bace_train_dataset, batch_size=batch_size, shuffle=True)
bace_val_loader = DataLoader(bace_val_dataset, batch_size=batch_size, shuffle=False)
bace_test_loader = DataLoader(bace_test_dataset, batch_size=batch_size, shuffle=False)

class_weights = compute_class_weights(bace_dataset)

num_node_features = bace_dataset.num_node_features

model_bace = GNNModel(num_features=num_node_features,
                      embedding_dim=2,
                      gnn_layer_type='GCNConv',
                      num_gnn_layers=2,
                      predictor_type='linear',
                      hidden_dim=32,
                      output_dim=2)

trained_model_bace = train_bace(model_bace,
                                bace_train_loader,
                                bace_val_loader,
                                class_weights=class_weights,
                                lr=1e-3,
                                max_epochs=50,
                                patience=10,
                                device=device)

test_loss, test_acc, test_precision_val, test_recall_val, test_f1_val = evaluate_bace(trained_model_bace, bace_test_loader,
                                    nn.CrossEntropyLoss(weight=class_weights.to(device)),
                                    device=device)

print(f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2f}")

all_emb_bace = []
all_labels_bace = []
trained_model_bace.eval()
with torch.no_grad():
    for data in bace_test_loader:
        data = data.to(device)
        out, emb = trained_model_bace(data)
        all_emb_bace.append(emb.cpu())
        all_labels_bace.append(data.y.cpu().view(-1))

all_emb_bace = torch.cat(all_emb_bace, dim=0).numpy()
all_labels_bace = torch.cat(all_labels_bace, dim=0).numpy()

if model_bace.encoder.convs[-1].out_channels == 2:
    visualize_embeddings_2d(all_emb_bace,
                            all_labels_bace,
                            title="BACE 2D Embeddings (Test)",
                            is_classification=True)

    predictor_bace = lambda x: trained_model_bace.predictor(x.to(device))
    visualize_decision_boundary_2d(predictor_bace,
                                   all_emb_bace,
                                   all_labels_bace,
                                   title="BACE Decision Boundary (Test)")

elif model_bace.encoder.convs[-1].out_channels == 1:
    visualize_embeddings_1d(all_emb_bace,
                            all_labels_bace,
                            title="BACE 1D Embeddings (Test)",
                            is_classification=True)