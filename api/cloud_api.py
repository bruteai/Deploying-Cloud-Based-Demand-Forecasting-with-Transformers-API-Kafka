from fastapi import FastAPI, Query
import torch
import numpy as np

# Import your models
from models.lstm_model import LSTMModel
from models.transformer_model import TimeSeriesTransformer

app = FastAPI()

# ------------------------------------------------------------------------
# 1) Instantiate both models
# ------------------------------------------------------------------------
lstm_model = LSTMModel(
    input_size=1, 
    hidden_size=64, 
    num_layers=2, 
    output_size=1, 
    dropout=0.2
)

transformer_model = TimeSeriesTransformer(
    input_size=1, 
    d_model=64, 
    nhead=4, 
    num_encoder_layers=3, 
    dim_feedforward=128, 
    dropout=0.1, 
    output_size=1, 
    max_len=500
)

# ------------------------------------------------------------------------
# 2) (Optional) Load Pretrained Weights
# ------------------------------------------------------------------------
# If you have saved model weights, you can load them here:
# lstm_model.load_state_dict(torch.load("path_to_lstm_weights.pt"))
# transformer_model.load_state_dict(torch.load("path_to_transformer_weights.pt"))

# Set both models to eval mode to disable dropout, etc.
lstm_model.eval()
transformer_model.eval()

@app.get("/predict/")
async def predict_demand(
    data: list[float] = Query(...),
    model_type: str = "lstm"
):
    """
    Predict demand using either the LSTM or Transformer model.
    
    :param data: A list of float values representing the time series.
    :param model_type: "lstm" or "transformer"
    :return: JSON response with the forecast array.
    """
    # Ensure model_type is lowercase
    model_type = model_type.lower()

    # Select the appropriate model
    if model_type == "transformer":
        chosen_model = transformer_model
    else:
        chosen_model = lstm_model  # default fallback is LSTM

    # Reshape input to (batch_size=1, sequence_length, input_size=1)
    input_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    # Perform inference with no gradient tracking
    with torch.no_grad():
        prediction = chosen_model(input_data)
    
    # If the output shape is [batch_size, output_size], squeeze to get a 1D array
    prediction_list = prediction.squeeze().numpy().tolist()

    return {
        "model_type": model_type,
        "forecast": prediction_list
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
