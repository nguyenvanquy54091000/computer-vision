from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request
import time
import logging
import os
from config.settings import BASE_DIR

log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, "app.log"),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Method: {request.method} | Path: {request.url.path} | Status: {response.status_code} | Time: {process_time:.4f}s")
        return response
