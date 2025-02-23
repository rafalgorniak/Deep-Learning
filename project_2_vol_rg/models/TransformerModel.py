import torch
from torch import nn


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, sequence_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                   dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # Embedding + Positional Encoding
        embedded = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]

        # Przekazanie przez TransformerDecoder
        transformer_out = self.transformer(
            tgt=embedded.transpose(0, 1),
            memory=embedded.transpose(0, 1)
        ).transpose(0, 1)

        # Dropout i warstwa liniowa
        transformer_out = self.dropout(transformer_out)
        output = self.fc(transformer_out)  # Zwróć całą sekwencję

        return output