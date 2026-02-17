"""
Ramadan Timetable Converter - Main Application
Production-grade web app for converting university timetables to Ramadan timing.
"""

import os
import uuid
import json
import csv
import io
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import structlog

from vision.preprocessor import ImagePreprocessor
from ocr.extractor import ScheduleExtractor
from ocr.ai_vision import AIVisionExtractor
from conversion.ramadan_converter import RamadanConverter
from calendar_sync.google_cal import GoogleCalendarSync

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
    description="Upload timetable images/PDFs → get Ramadan-adjusted schedules + Google Calendar sync",
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
calendar_sync = GoogleCalendarSync()

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


@app.post("/api/update-schedule")
async def update_schedule(request: Request):
    """Allow user to manually edit extracted schedule before export."""
    body = await request.json()
    session_id = body.get("session_id")
    updated_classes = body.get("classes", [])

    if session_id not in sessions:
        raise HTTPException(404, "Session not found.")

    # Re-convert with user edits
    raw = {"classes": updated_classes}
    converted = converter.convert(raw)

    sessions[session_id]["original"] = raw
    sessions[session_id]["converted"] = converted

    return JSONResponse({
        "success": True,
        "ramadan_schedule": converted,
    })


@app.get("/api/export/json")
async def export_json(session_id: str = Query(...)):
    """Export Ramadan schedule as JSON."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found.")

    data = sessions[session_id]["converted"]
    return JSONResponse(data, headers={
        "Content-Disposition": f'attachment; filename="ramadan_schedule_{session_id}.json"'
    })


@app.get("/api/export/csv")
async def export_csv(session_id: str = Query(...)):
    """Export Ramadan schedule as CSV."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found.")

    data = sessions[session_id]["converted"]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Day", "Course", "Original Start", "Original End",
                     "Ramadan Start", "Ramadan End", "Duration (min)", "Confidence"])

    for cls in data.get("classes", []):
        writer.writerow([
            cls.get("day", ""),
            cls.get("course", ""),
            cls.get("original_start", ""),
            cls.get("original_end", ""),
            cls.get("ramadan_start", ""),
            cls.get("ramadan_end", ""),
            cls.get("duration_minutes", ""),
            cls.get("confidence", ""),
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="ramadan_schedule_{session_id}.csv"'},
    )


@app.get("/api/export/ics")
async def export_ics(session_id: str = Query(...)):
    """Export Ramadan schedule as .ics calendar file (works with Google Calendar, Outlook, Apple)."""
    if session_id not in sessions:
        raise HTTPException(404, "Session not found.")

    data = sessions[session_id]["converted"]
    tz = os.getenv("DEFAULT_TIMEZONE", "Asia/Dubai")
    ramadan_end = "20260318"  # End of Ramadan 2026 in UAE (Wed 18 March)

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Ramadan Timetable Converter//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-TIMEZONE:{tz}",
    ]

    day_offsets = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
    }

    today = datetime.now()
    current_weekday = today.weekday()
    uid_counter = 0

    for cls in data.get("classes", []):
        day_name = cls.get("day", "")
        course = cls.get("course", "Class")
        ram_start = cls.get("ramadan_start", "")
        ram_end = cls.get("ramadan_end", "")
        orig_start = cls.get("original_start", "")
        orig_end = cls.get("original_end", "")

        if not day_name or not ram_start or not ram_end:
            continue

        day_offset = day_offsets.get(day_name)
        if day_offset is None:
            continue

        days_ahead = day_offset - current_weekday
        if days_ahead < 0:
            days_ahead += 7
        next_date = today + timedelta(days=days_ahead)
        date_str = next_date.strftime("%Y%m%d")

        start_ics = ram_start.replace(":", "") + "00"
        end_ics = ram_end.replace(":", "") + "00"
        uid_counter += 1
        uid = f"ramadan-{session_id}-{uid_counter}@timetable"
        now_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

        lines.extend([
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{now_stamp}",
            f"DTSTART;TZID={tz}:{date_str}T{start_ics}",
            f"DTEND;TZID={tz}:{date_str}T{end_ics}",
            f"RRULE:FREQ=WEEKLY;UNTIL={ramadan_end}T235959Z",
            f"SUMMARY:[Ramadan] {course}",
            f"DESCRIPTION:Original: {orig_start}-{orig_end} | Ramadan: {ram_start}-{ram_end}",
            "STATUS:CONFIRMED",
            "BEGIN:VALARM",
            "TRIGGER:-PT15M",
            "ACTION:DISPLAY",
            f"DESCRIPTION:Reminder: {course} in 15 minutes",
            "END:VALARM",
            "END:VEVENT",
        ])

    lines.append("END:VCALENDAR")
    ics_content = "\r\n".join(lines)

    return StreamingResponse(
        io.BytesIO(ics_content.encode("utf-8")),
        media_type="text/calendar",
        headers={"Content-Disposition": f'attachment; filename="ramadan_schedule_{session_id}.ics"'},
    )


@app.get("/api/calendar/auth")
async def calendar_auth(session_id: str = Query(...)):
    """Start Google Calendar OAuth flow."""
    auth_url = calendar_sync.get_auth_url(session_id)
    return JSONResponse({"auth_url": auth_url})


@app.get("/api/calendar/callback")
async def calendar_callback(code: str = Query(...), state: str = Query("")):
    """Handle Google OAuth callback."""
    try:
        session_id = state
        calendar_sync.handle_callback(code, session_id)
        return RedirectResponse(url=f"/?session_id={session_id}&calendar=connected")
    except Exception as e:
        logger.error("calendar_auth_error", error=str(e))
        raise HTTPException(500, f"Calendar auth failed: {str(e)}")


@app.post("/api/calendar/sync")
async def sync_calendar(request: Request):
    """Sync Ramadan schedule to Google Calendar."""
    body = await request.json()
    session_id = body.get("session_id")
    timezone = body.get("timezone", os.getenv("DEFAULT_TIMEZONE", "Asia/Dubai"))

    if session_id not in sessions:
        raise HTTPException(404, "Session not found.")

    data = sessions[session_id]["converted"]

    try:
        result = calendar_sync.sync_events(session_id, data, timezone)
        return JSONResponse({"success": True, "events_created": result})
    except Exception as e:
        logger.error("calendar_sync_error", error=str(e))
        raise HTTPException(500, f"Calendar sync failed: {str(e)}")


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
