from pydantic import BaseModel

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    bounding_box: BoundingBox
