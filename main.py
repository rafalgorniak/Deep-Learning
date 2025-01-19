import torch
import torch.nn as nn
import numpy as np
import random

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from GNN.GNNModel import GNNModel
from functionalities.bace_functionalities import train_bace, evaluate_bace
from functionalities.shared_functionalities import train_val_test_split, compute_class_weights

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
                                max_epochs=100,
                                patience=10,
                                device=device)

test_loss, test_acc = evaluate_bace(trained_model_bace, bace_test_loader,
                                    nn.CrossEntropyLoss(weight=class_weights.to(device)),
                                    device=device)

print(f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2f}")