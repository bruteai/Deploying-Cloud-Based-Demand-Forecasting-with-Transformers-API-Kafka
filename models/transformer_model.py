import torch
import torch.nn as nn

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.transformer(x)
        return self.fc(x[:, -1, :])
