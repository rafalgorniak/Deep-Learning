import math

import torch
from torch import nn

from project_2.PositionalEncoding import PositionalEncoding


class SymbolTransformerModel(nn.Module):
    def __init__(self,
                unique_symbols_count: int,
                embedding_dimensions: int,
                num_layers,
                num_heads: int = 4,
                dropout_rate: float = 0.1):
        super(SymbolTransformerModel, self).__init__()
        self.embedding_dimensions = embedding_dimensions
        self.embedding = nn.Embedding(unique_symbols_count, embedding_dimensions)
        self.pos_encoder = PositionalEncoding(embedding_dimensions, dropout=dropout_rate)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dimensions,
            nhead=num_heads,
            dropout=dropout_rate
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embedding_dimensions, unique_symbols_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size, seq_len = x.size()

        embedded = self.embedding(x) * math.sqrt(self.embedding_dimensions)
        embedded = self.pos_encoder(embedded)

        tgt = embedded.transpose(0, 1)
        tgt_mask = generate_square_subsequent_mask(seq_len).to(device)

        memory = torch.zeros(1, batch_size, self.embedding_dimensions, device=device)

        decoder_output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        decoder_output = decoder_output.transpose(0, 1)

        logits = self.fc_out(decoder_output)
        return logits


def generate_square_subsequent_mask(val: int) -> torch.Tensor:
    mask = (torch.triu(torch.ones(val, val)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask