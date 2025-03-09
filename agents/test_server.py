#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add current directory to path
sys.path.append('.')

try:
    from src.main import app
    import uvicorn
    
    # Override FastAPI's exception handler to print full tracebacks
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.exceptions import RequestValidationError
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "traceback": traceback.format_exc()},
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error: {str(exc)}")
        return JSONResponse(
            status_code=422,
            content={"detail": str(exc)},
        )
    
    # Run the server
    if __name__ == "__main__":
        logger.info("Starting server with debug error handling")
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
        
except Exception as e:
    logger.error(f"Error starting server: {str(e)}")
    logger.error(traceback.format_exc()) 