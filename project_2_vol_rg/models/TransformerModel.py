import torch
from torch import nn

from project_2_vol_rg.classes.PositionalEncoding import PositionalEncoding


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, sequence_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=sequence_length)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

    def forward(self, x):
        """
        Forward pass of the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, vocab_size).
        """
        # Embedding and positional encoding
        embedded = self.embedding(x) * (self.embedding_dim ** 0.5)
        encoded = self.positional_encoding(embedded)

        # Print the sequence length
        seq_len = x.size(1)
        print(f"Sequence Length: {seq_len}")  # Ensure this is 800 or the expected length

        # Create causal mask for autoregressive behavior
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        # Debug: print mask shape to verify it
        print(f"Shape of tgt_mask before expand: {tgt_mask.shape}")

        # Expand mask to match the shape (seq_len, batch_size, seq_len)
        tgt_mask = tgt_mask.unsqueeze(1).expand(-1, x.size(0), -1)  # Shape becomes (seq_len, batch_size, seq_len)

        # Debug: print mask shape after expanding
        print(f"Shape of tgt_mask after expand: {tgt_mask.shape}")

        # Transformer decoder
        decoded = self.transformer_decoder(tgt=encoded, memory=encoded, tgt_mask=tgt_mask)

        # Output layer
        output = self.fc_out(decoded)
        return output

    @staticmethod
    def generate_square_subsequent_mask(size):
        """
        Generate a square mask for the sequence to mask out subsequent positions.

        Args:
            size (int): Length of the sequence.

        Returns:
            torch.Tensor: A mask tensor of shape (size, size).
        """
        mask = torch.triu(torch.ones(size, size), diagonal=1)  # Upper triangular matrix
        mask = mask.masked_fill(mask == 1, float('-inf'))  # Masked values are -inf (for attention)
        return mask