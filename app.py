"""
Ramadan Timetable Converter - Main Application
Production-grade web app for converting university timetables to Ramadan timing.
"""

import os
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import structlog

from vision.preprocessor import ImagePreprocessor
from ocr.extractor import ScheduleExtractor
from ocr.ai_vision import AIVisionExtractor
from conversion.ramadan_converter import RamadanConverter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR = BASE_DIR / "templates"

MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20")) * 1024 * 1024

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# App Init
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Ramadan Timetable Converter",
    description="Upload timetable images/PDFs → get Ramadan-adjusted schedules + calendar export",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Singletons
preprocessor = ImagePreprocessor()
extractor = ScheduleExtractor()
ai_extractor = AIVisionExtractor()
converter = RamadanConverter()

# In-memory store for sessions (production would use Redis/DB)
sessions: dict = {}

# ---------------------------------------------------------------------------
# Routes - Pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Landing page with upload form."""
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------------
# Routes - API
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_timetable(file: UploadFile = File(...)):
    """
    Accept an image or PDF upload, run the full pipeline:
    1. Preprocess image
    2. OCR + schedule extraction
    3. Ramadan conversion
    Return the extracted + converted schedule with confidence scores.
    """
    # Validate file type
    allowed = {
        "image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp",
        "application/pdf",
    }
    if file.content_type not in allowed:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}. Allowed: {', '.join(allowed)}")

    # Read and save
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"File too large. Max {MAX_UPLOAD_SIZE // (1024*1024)} MB.")

    ext = Path(file.filename or "upload").suffix or ".png"
    file_id = uuid.uuid4().hex[:12]
    file_path = UPLOAD_DIR / f"{file_id}{ext}"
    file_path.write_bytes(content)

    logger.info("file_uploaded", file_id=file_id, filename=file.filename, size=len(content))

    try:
        raw_schedule = None

        # 1. Try AI Vision (Gemini) first — much better for photos
        if ai_extractor.is_available():
            logger.info("trying_ai_extraction", file_id=file_id)
            raw_schedule = ai_extractor.extract_from_file(str(file_path))
            if raw_schedule and raw_schedule.get("classes"):
                logger.info("ai_extraction_done", file_id=file_id,
                            classes_found=len(raw_schedule["classes"]))

        # 2. Fallback to Tesseract OCR if AI didn't work
        if not raw_schedule or not raw_schedule.get("classes"):
            logger.info("falling_back_to_tesseract", file_id=file_id)
            processed_images = preprocessor.process(str(file_path))
            logger.info("preprocessing_done", file_id=file_id, pages=len(processed_images))
            raw_schedule = extractor.extract(processed_images)
            logger.info("extraction_done", file_id=file_id,
                        classes_found=len(raw_schedule.get("classes", [])))

        # 3. Convert to Ramadan timing
        converted = converter.convert(raw_schedule)
        logger.info("conversion_done", file_id=file_id)

        # Store in session
        session_id = uuid.uuid4().hex[:16]
        sessions[session_id] = {
            "file_id": file_id,
            "original": raw_schedule,
            "converted": converted,
            "created_at": datetime.utcnow().isoformat(),
        }

        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "original_schedule": raw_schedule,
            "ramadan_schedule": converted,
        })

    except Exception as e:
        logger.error("pipeline_error", file_id=file_id, error=str(e))
        raise HTTPException(500, f"Processing error: {str(e)}")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", os.getenv("APP_PORT", "8000"))),
        reload=os.getenv("DEBUG", "false").lower() == "true",
    )
