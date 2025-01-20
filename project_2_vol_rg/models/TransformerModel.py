import torch.nn as nn
import torch


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, max_seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(max_seq_length, embedding_dim))
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tgt, memory=None, tgt_mask=None, tgt_key_padding_mask=None):
        # Embed the target sequence
        tgt_embedded = self.embedding(tgt) + self.positional_encoding[:tgt.size(1), :].unsqueeze(0)

        # Use a dummy memory tensor if none is provided
        if memory is None:
            memory = torch.zeros(tgt_embedded.size(0), tgt_embedded.size(1), tgt_embedded.size(2), device=tgt.device)

        # Pass through the TransformerDecoder
        decoded = self.transformer_decoder(
            tgt=tgt_embedded,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # Map to vocab space
        output = self.fc(decoded)
        return output