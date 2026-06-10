#!/usr/bin/env python3
"""
Passport Photo Validator API v2.0 for Bangladesh
FastAPI-based async REST API for validating passport photos.
"""

import os
import asyncio
import tempfile
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import aiofiles

from validator import PassportPhotoValidatorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Passport Photo Validator API v2.0",
    description="Bangladesh passport photo validation service. Upload an image to validate.",
    version="2.0.0"
)

# CORS - allow all origins for testing (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ValidationResponse(BaseModel):
    success: bool
    filename: str
    version: str
    status: str
    results: Dict[str, Any]


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Passport Photo Validator API v2.0",
        "version": "2.0.0",
        "description": "Bangladesh passport photo validation service",
        "endpoints": {
            "POST /validate": "Upload an image file to validate",
            "GET /health": "Health check endpoint"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/test")
async def test_ui():
    """Serve the interactive test UI (index.html)"""
    html_path = "index.html"
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(html_path)


@app.post("/validate", response_model=ValidationResponse)
async def validate_photo(file: UploadFile = File(...)):
    """
    Upload a passport photo for validation.

    - **file**: Image file (jpg, jpeg, png, bmp, tiff, webp)

    Returns detailed validation results in JSON format.
    """
    # Validate file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="passport_val_")
    temp_path = os.path.join(temp_dir, f"upload{file_ext}")

    try:
        # Async file write using aiofiles
        async with aiofiles.open(temp_path, "wb") as buffer:
            # Read file in chunks to avoid loading large files into memory
            chunk_size = 1024 * 1024  # 1MB chunks
            while chunk := await file.read(chunk_size):
                await buffer.write(chunk)

        # Run CPU-bound validation in thread pool to avoid blocking the event loop
        validator = PassportPhotoValidatorV2(temp_path)
        is_valid = await asyncio.to_thread(validator.validate)

        return ValidationResponse(
            success=bool(is_valid),
            filename=file.filename,
            version="v2.0",
            status="PASSED" if is_valid else "FAILED",
            results=validator.validation_results
        )

    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")

    finally:
        # Cleanup temp files (async-friendly cleanup)
        await asyncio.to_thread(shutil.rmtree, temp_dir, ignore_errors=True)
        await file.close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
