from fastapi import APIRouter, File, UploadFile, HTTPException
from schemas.prediction import PredictionResponse, BoundingBox
from utils.model_loader import load_model
from utils.classes import CALTECH_101_CLASSES
from config.settings import IMAGE_SIZE, UPLOADS_DIR, BASE_DIR
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import io
import os
import uuid
import logging
import time
import aiofiles
import asyncio

router = APIRouter()

# Setup Predictor Logger
pred_logger = logging.getLogger("predictor")
pred_logger.setLevel(logging.INFO)
log_path = os.path.join(BASE_DIR, "logs", "predictor.log")
file_handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)
if not pred_logger.handlers:
    pred_logger.addHandler(file_handler)

# Load model globally on startup (lazy load handled here for simplicity)
try:
    model, device = load_model()
except Exception as e:
    model = None
    print(f"Warning: Could not load model. {e}")

# Transformation matching the training process
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model implies an error or is not loaded properly.")
    
    try:
        start_time = time.time()
        contents = await file.read()
        
        image_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1]
        if not ext: 
            ext = ".jpg"
        save_path = os.path.join(UPLOADS_DIR, f"{image_id}{ext}")
        async with aiofiles.open(save_path, "wb") as f:
            await f.write(contents)
            
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        width, height = image.size
        
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        def run_inference():
            with torch.no_grad():
                return model(input_tensor)
                
        pred_boxes, pred_logits = await asyncio.to_thread(run_inference)
            
        box = pred_boxes.cpu().squeeze().numpy()
        probs = F.softmax(pred_logits, dim=1)
        conf, class_idx = torch.max(probs, dim=1)
        
        label_name = CALTECH_101_CLASSES[class_idx.item()]
        confidence_score = conf.item()
        
        x1, y1, x2, y2 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
        
        process_time = time.time() - start_time
        pred_logger.info(f"File: {file.filename} | Saved As: {os.path.basename(save_path)} | Class: {label_name} | Confidence: {confidence_score:.4f} | BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] | Time: {process_time:.4f}s")
        
        return PredictionResponse(
            class_name=label_name,
            confidence=confidence_score,
            bounding_box=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
