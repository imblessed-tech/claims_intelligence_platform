import uuid
import time
import json
import logging
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        #Generate a unique request ID
        request_id = str(uuid.uuid4())[:8]
        
        #Add the request ID to the request metadata
        request.state.request_id = request_id
        
        logger.info(f"[{request_id}] Request {request.method} {request.url}")
        
        # Process request
        start_time = time.time()
        try:
            response = await call_next(request)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            log_data = {
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "duration_ms": round(duration_ms, 2),
                "request_id": request_id
            }
            logger.error(json.dumps(log_data))
            raise e
        duration_ms = (time.time() - start_time) * 1000
        
        # Add response details to response metadata
        response.headers["X-Request-ID"] = request_id
        
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
            "request_id": request_id
        }
        logger.info(json.dumps(log_data))
        
        return response