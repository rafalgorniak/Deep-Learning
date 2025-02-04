import math

import torch
from torch import nn


def generate_square_subsequent_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SymbolTransformerDecoderModel(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dimensions: int, num_layers: int,
                 num_heads: int = 4, dropout_rate: float = 0.1):
        super(SymbolTransformerDecoderModel, self).__init__()
        self.embedding_dim = embedding_dimensions
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimensions)
        self.pos_encoder = PositionalEncoding(embedding_dimensions, dropout=dropout_rate)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dimensions,
            nhead=num_heads,
            dropout=dropout_rate
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dimensions, vocabulary_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size, sequence_length = x.size()

        x_embedding = self.embedding(x) * math.sqrt(self.embedding_dim)
        x_embedding = self.pos_encoder(x_embedding)
        x_embedding = x_embedding.transpose(0, 1)
        target_mask = generate_square_subsequent_mask(sequence_length).to(device)

        decoder_output = self.transformer_decoder(
            tgt=x_embedding,
            memory=x_embedding,
            tgt_mask=target_mask,
            memory_mask=target_mask
        )

        decoder_output = decoder_output.transpose(0, 1)
        logits = self.fc_out(decoder_output)
        return logits