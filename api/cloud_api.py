from fastapi import FastAPI, Query
import torch
import numpy as np
from models.transformer_model import TransformerTimeSeries

app = FastAPI()

# Load pre-trained model (mocked)
model = TransformerTimeSeries(input_dim=1, num_heads=4, num_layers=3)

@app.get("/predict/")
async def predict_demand(data: list = Query(...)):
    input_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    model.eval()
    prediction = model(input_data).detach().numpy().tolist()
    return {"forecast": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
