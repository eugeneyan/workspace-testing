from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import xgboost as xgb
import torch

app = FastAPI()

class PredictRequest(BaseModel):
    model_type: str
    data: list

class PredictResponse(BaseModel):
    prediction: list

def load_model(model_type: str):
    if model_type == 'sklearn':
        return joblib.load('model/sklearn_model.pkl')
    elif model_type == 'xgboost':
        return xgb.Booster(model_file='model/xgboost_model.json')
    elif model_type == 'pytorch':
        model = torch.load('model/pytorch_model.pth')
        model.eval()
        return model
    else:
        raise ValueError("Unsupported model type")

@app.post("/predict/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    model = load_model(request.model_type)
    if request.model_type in ['sklearn', 'xgboost']:
        prediction = model.predict(request.data)
    elif request.model_type == 'pytorch':
        with torch.no_grad():
            prediction = model(torch.tensor(request.data)).tolist()
    else:
        raise ValueError("Unsupported model type")
    return PredictResponse(prediction=prediction)

@app.get("/ping/")
async def ping():
    return {"message": "pong"}

