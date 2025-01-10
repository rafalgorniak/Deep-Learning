from torch import nn


class SymbolLSTMModel(nn.Module):
    def __init__(self, unique_symbols_count: int, embedding_dimensions: int, hidden_dimensions: int, num_layers: int=1):
        super(SymbolLSTMModel, self).__init__()
        self.embedding = nn.Embedding(unique_symbols_count, embedding_dimensions)
        self.lstm = nn.LSTM(input_size=embedding_dimensions,
                            hidden_size=hidden_dimensions,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dimensions, unique_symbols_count)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        
        return logits, hidden