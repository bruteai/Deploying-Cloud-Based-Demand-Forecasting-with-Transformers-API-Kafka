import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based model for time series forecasting.
    Args:
        input_size:    Number of input features per time step.
        d_model:       Dimension of the embedding (hidden size in the transformer).
        nhead:         Number of attention heads.
        num_encoder_layers:  Number of encoder layers in the transformer.
        dim_feedforward:     Dimension of the feedforward network in the transformer.
        dropout:       Dropout probability for regularization.
        output_size:   Dimension of the output (e.g., 1 for univariate forecasting).
        max_len:       Maximum sequence length for positional encoding.
    """
    def __init__(self, 
                 input_size=1,
                 d_model=64,
                 nhead=4,
                 num_encoder_layers=2,
                 dim_feedforward=128,
                 dropout=0.1,
                 output_size=1,
                 max_len=500):
        super(TimeSeriesTransformer, self).__init__()

        self.d_model = d_model

        # 1) Input embedding layer
        self.input_fc = nn.Linear(input_size, d_model)

        # 2) Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )

        # 4) Final projection layer
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        """
        Forward pass for the transformer model.
        Args:
            x: Tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Output predictions of shape (batch_size, sequence_length, output_size) 
            or (batch_size, output_size) based on usage.
        """
        # x -> (batch_size, seq_len, input_size)

        # Project inputs to d_model
        x = self.input_fc(x)  # (batch_size, seq_len, d_model)

        # Scale the embedding by sqrt(d_model) for better stability
        x = x * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.positional_encoding(x)  # (batch_size, seq_len, d_model)

        # Pass through transformer encoder
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)

        # Typically, we want the last time step's output for forecasting
        # or you can gather the entire sequence for a multi-step forecast
        # For single-step forecast at the final time step:
        out = encoded[:, -1, :]  # (batch_size, d_model)

        # Project to the desired output size
        out = self.fc_out(out)   # (batch_size, output_size)

        return out

class PositionalEncoding(nn.Module):
    """
    Computes positional encoding for input sequences in a Transformer.
    """
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        # Even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer so it’s not a learnable parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Add positional encoding
        x = x + self.pe[:seq_len, :]

        return self.dropout(x)
