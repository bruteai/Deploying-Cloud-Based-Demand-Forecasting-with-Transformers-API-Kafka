import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM-based model for time series forecasting.
    Args:
        input_size:    Number of input features per time step.
        hidden_size:   Number of hidden units in the LSTM.
        num_layers:    Number of stacked LSTM layers.
        output_size:   Dimension of the output (e.g., 1 for univariate forecasting).
        dropout:       Dropout probability for regularization.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)

        # Define a fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass for the LSTM model.
        Args:
            x: Tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Output predictions of shape (batch_size, sequence_length, output_size)
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)

        # Get the final time step output for each sequence
        # If you want predictions at each time step, you can remove the [-1] index
        # and feed all steps into your fc layer.
        out = self.fc(out[:, -1, :])     # (batch_size, output_size)

        # Expand dimensions if you want to keep sequence-length dimension
        # out = out.unsqueeze(1)         # (batch_size, 1, output_size)

        return out
