import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from fastapi import FastAPI
from middleware import setup_cors, LogMiddleware
from routes.prediction import router as prediction_router

app = FastAPI(
    title="ViT Object Detection API", 
    description="API for Caltech-101 Image Prediction using ViT"
)

app.add_middleware(LogMiddleware)
setup_cors(app)

@app.get("/")
async def root():
    return {"message": "Server ViT Object Detection API is running!"}

app.include_router(prediction_router, prefix="/api", tags=["Prediction"])
