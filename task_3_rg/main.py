import torch
import os
import torch.cuda
from torch import nn, optim

from torch_geometric.loader import DataLoader

from task_3_rg.data.data_loader import get_bace_datasets, get_qm9_datasets
from task_3_rg.models.BACEModel import BACEModel
from task_3_rg.models.QM9Model import QM9Model
from task_3_rg.utils.model_functionalities import train_evaluate_save_model, get_embeddings, \
    train_evaluate_save_model_regression
from task_3_rg.utils.visualization import plot_chart_train_phase, \
    plot_chart_train_phase_regression, \
     visualize_embeddings_2d, visualize_decision_boundary_2d


def main(mode: int):
    bace_model_path = './saved_models/BACE_model.pth'
    bace_model_path = './saved_models/BACE_model_Transformer.pth'
    qm9_model_path = './saved_models/QM9_model.pth'

    num_epochs = 5
    patience = 5
    learning_rate = 0.001

    if mode == 1:
        train_dataset, val_dataset, test_dataset, class_weights = get_bace_datasets()

        # Assuming train_dataset, val_dataset, and test_dataset are already loaded
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        hidden_dim = 32
        embedding_dim = 2
        output_dim = 2
        num_layers = 2
        use_mlp = True

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = BACEModel(hidden_dim=hidden_dim, embedding_dim=embedding_dim, num_layers=num_layers, output_dim=output_dim,
                          use_mlp=use_mlp)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))


        train_los, val_los, acc_metr, prec_metr, recc_metr, f1_metr = train_evaluate_save_model(model, criterion, optimizer, train_loader=train_loader,
                                  val_loader=val_loader,
                                  epochs=num_epochs, device=device, best_model_path=bace_model_path,
                                  patience=patience)

        plot_chart_train_phase(train_los, val_los, acc_metr, prec_metr, recc_metr, f1_metr)

        # Uzyskujemy embeddingi z walidacyjnego zbioru
        embeddings, labels = get_embeddings(model, test_loader, device)

        #visualize_embeddings_1d(embeddings, labels, title="BACE 2D Embedding Visualisation")
        visualize_embeddings_2d(embeddings, labels, title="BACE 2D Embedding Visualisation", is_classification=True)
        predictor_bace = lambda x: model.predictor(x.to(device))
        visualize_decision_boundary_2d( predictor_bace, embeddings, labels, title="BACE 2D Embedding Visualisation with Decision Boundary")

    else:
        train_dataset, val_dataset, test_dataset = get_qm9_datasets()

        # Assuming train_dataset, val_dataset, and test_dataset are already loaded
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        hidden_dim = 32
        embedding_dim = 2
        output_dim = 1
        num_layers = 4
        use_mlp = False

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = QM9Model(hidden_dim=hidden_dim, embedding_dim=embedding_dim, num_layers=num_layers,
                          output_dim=output_dim,
                          use_mlp=use_mlp)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_los, val_los, mae_met = train_evaluate_save_model_regression(model, criterion,optimizer,
                                                                                train_loader=train_loader,
                                                                                val_loader=val_loader,
                                                                                epochs=num_epochs,
                                                                                device=device,
                                                                                best_model_path=qm9_model_path,
                                                                                patience=patience)

        plot_chart_train_phase_regression(train_los, val_los, mae_met)

        # Uzyskujemy embeddingi z walidacyjnego zbioru
        embeddings, labels = get_embeddings(model, test_loader, device)

        # Wizualizacja embeddingów
        visualize_embeddings_2d(embeddings, labels, title="MQ9 2D Embedding Visualisation")


if __name__ == '__main__':
    main(1)
    # main(2)