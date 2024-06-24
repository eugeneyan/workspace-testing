import joblib
import xgboost as xgb
import torch

def load_model(model_type: str, model_path: str):
    """
    Load a model based on its type and file path.
    Supports sklearn, xgboost, and PyTorch models.
    """
    if model_type == 'sklearn':
        return joblib.load(model_path)
    elif model_type == 'xgboost':
        return xgb.Booster(model_file=model_path)
    elif model_type == 'pytorch':
        model = torch.load(model_path)
        model.eval()
        return model
    else:
        raise ValueError("Unsupported model type")
