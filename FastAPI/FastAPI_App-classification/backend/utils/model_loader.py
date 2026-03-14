import torch
from models.vit import ViTObjectDetector
from config.settings import MODEL_PATH
from utils.classes import CALTECH_101_CLASSES
import os

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(CALTECH_101_CLASSES)
    
    model = ViTObjectDetector(num_classes=num_classes)
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    return model, device
