"""
AI Vision Extraction Module
Uses Google Gemini (free API) as the PRIMARY extraction engine.
Gemini can actually "understand" timetable structure from messy photos,
screenshots, colored blocks, etc. — far superior to Tesseract grid detection.

Falls back to Tesseract-based extraction if Gemini is unavailable.
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import structlog

logger = structlog.get_logger()

# Try importing google.genai (new SDK)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

EXTRACTION_PROMPT = """You are analyzing a university/college class timetable image.

Extract ALL classes/courses visible in this timetable. For each class, identify:
1. **day** — The day of the week (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday)
2. **course** — The course/subject name (e.g., "Mathematics 101", "Physics Lab", "CS201")
3. **start_time** — Start time in 24-hour HH:MM format (e.g., "08:00", "14:30")
4. **end_time** — End time in 24-hour HH:MM format
5. **room** — Room/location if visible (otherwise empty string)

IMPORTANT RULES:
- Extract EVERY class block you can see, even if partially visible
- Use 24-hour format for ALL times
- If a colored block spans multiple time slots, that is ONE class with the full duration
- If the same course appears on multiple days, list each as a SEPARATE entry
- For time: morning classes like 8, 9, 10, 11 AM are 08:00, 09:00, 10:00, 11:00; afternoon 1, 2, 3 PM are 13:00, 14:00, 15:00
- If you can read text inside colored blocks, that is the course name
- Include section numbers, course codes if visible
- If you cannot determine the end time, estimate based on typical 1-hour slots

Return ONLY a valid JSON object in this exact format (no markdown, no explanation):
{
    "classes": [
        {
            "day": "Monday",
            "course": "Course Name Here",
            "start_time": "08:00",
            "end_time": "09:00",
            "room": ""
        }
    ]
}

If you cannot read the timetable at all, return: {"classes": [], "error": "Could not read timetable"}
"""


class AIVisionExtractor:
    """
    Uses Google Gemini (free) to extract schedule from timetable images.
    This is orders of magnitude better than Tesseract for:
    - Phone photos with glare/blur/skew
    - Colored block timetables
    - Non-standard layouts
    - Handwritten or stylized text
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._configured = False
        self._client = None

        if GEMINI_AVAILABLE and self.api_key:
            try:
                self._client = genai.Client(api_key=self.api_key)
                self._configured = True
                logger.info("gemini_configured", model=self.model_name)
            except Exception as e:
                logger.warning("gemini_config_failed", error=str(e))
        else:
            if not GEMINI_AVAILABLE:
                logger.info("gemini_sdk_not_installed",
                            msg="pip install google-genai")
            elif not self.api_key:
                logger.info("gemini_no_api_key",
                            msg="Set GEMINI_API_KEY in .env (free from aistudio.google.com)")

    def is_available(self) -> bool:
        return self._configured

    def extract_from_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Send an image file to Gemini and get structured schedule back.
        Returns the schedule dict or None if extraction fails.
        """
        if not self._configured:
            return None

        try:
            path = Path(file_path)
            ext = path.suffix.lower()

            # For PDFs, convert pages to images first
            if ext == ".pdf":
                return self._extract_from_pdf(file_path)

            # Read image bytes
            image_data = path.read_bytes()

            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".gif": "image/gif",
            }
            mime_type = mime_map.get(ext, "image/jpeg")

            return self._call_gemini(image_data, mime_type)

        except Exception as e:
            logger.error("gemini_extraction_error", error=str(e))
            return None

    def extract_from_bytes(
        self, image_bytes: bytes, mime_type: str = "image/png"
    ) -> Optional[Dict[str, Any]]:
        """Extract from raw image bytes."""
        if not self._configured:
            return None
        try:
            return self._call_gemini(image_bytes, mime_type)
        except Exception as e:
            logger.error("gemini_extraction_error", error=str(e))
            return None

    def _call_gemini(
        self, image_data: bytes, mime_type: str
    ) -> Optional[Dict[str, Any]]:
        """Call Gemini API with retry logic and model fallback."""
        from google.genai import types

        # Models to try in order (separate quotas per model)
        models_to_try = [self.model_name]
        fallback_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]
        for m in fallback_models:
            if m not in models_to_try:
                models_to_try.append(m)

        contents = [
            types.Content(
                parts=[
                    types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    types.Part.from_text(text=EXTRACTION_PROMPT),
                ]
            )
        ]

        last_error = None
        for model_name in models_to_try:
            for attempt in range(3):  # Up to 3 retries per model
                try:
                    logger.info("gemini_attempt", model=model_name,
                                attempt=attempt + 1)
                    response = self._client.models.generate_content(
                        model=model_name,
                        contents=contents,
                    )

                    if not response or not response.text:
                        logger.warning("gemini_empty_response", model=model_name)
                        break  # Try next model

                    raw_text = response.text.strip()
                    logger.info("gemini_response_ok", model=model_name,
                                length=len(raw_text))

                    # Parse JSON from response
                    schedule = self._parse_json_response(raw_text)
                    if schedule and schedule.get("classes"):
                        classes = []
                        for cls in schedule["classes"]:
                            normalized = self._normalize_class(cls)
                            if normalized:
                                classes.append(normalized)

                        if classes:
                            return {
                                "classes": classes,
                                "total_classes": len(classes),
                                "average_confidence": 0.85,
                                "conflicts": self._detect_conflicts(classes),
                                "extraction_method": f"gemini_ai ({model_name})",
                                "warnings": [],
                            }

                    logger.warning("gemini_no_classes_parsed", model=model_name)
                    break  # Got a response but couldn't parse; try next model

                except Exception as e:
                    last_error = str(e)
                    if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
                        wait = 3 * (attempt + 1)  # 3s, 6s, 9s
                        logger.warning("gemini_rate_limited", model=model_name,
                                       attempt=attempt + 1, wait=wait)
                        time.sleep(wait)
                        continue  # Retry same model
                    else:
                        logger.error("gemini_error", model=model_name,
                                     error=last_error)
                        break  # Non-rate-limit error, try next model

        logger.error("gemini_all_models_failed", last_error=last_error)
        return None

    def _extract_from_pdf(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Handle PDF by converting pages to images first."""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=5)

            all_classes = []
            for i, pil_img in enumerate(images):
                import io
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                result = self._call_gemini(img_bytes, "image/png")
                if result and result.get("classes"):
                    all_classes.extend(result["classes"])

            if all_classes:
                return {
                    "classes": all_classes,
                    "total_classes": len(all_classes),
                    "average_confidence": 0.85,
                    "conflicts": self._detect_conflicts(all_classes),
                    "extraction_method": "gemini_ai",
                    "warnings": [],
                }
        except Exception as e:
            logger.error("pdf_gemini_error", error=str(e))

        return None

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Parse JSON from Gemini response, handling markdown code blocks."""
        # Remove markdown code fences if present
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("gemini_json_parse_failed", preview=text[:200])
        return None

    def _normalize_class(self, cls: Dict) -> Optional[Dict]:
        """Normalize and validate a class entry from Gemini."""
        day = cls.get("day", "").strip()
        course = cls.get("course", "").strip()
        start = cls.get("start_time", "").strip()
        end = cls.get("end_time", "").strip()
        room = cls.get("room", "").strip()

        # Validate day
        valid_days = {
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        }
        if day not in valid_days:
            for d in valid_days:
                if d.lower().startswith(day.lower()[:3]):
                    day = d
                    break
            else:
                return None

        if not course:
            return None

        # Validate/fix times
        start = self._fix_time(start)
        end = self._fix_time(end)

        if not start:
            return None
        if not end:
            end = self._add_minutes(start, 60)

        # Calculate duration
        duration = self._duration_minutes(start, end)
        if duration <= 0 or duration > 360:
            duration = 60
            end = self._add_minutes(start, 60)

        return {
            "day": day,
            "course": course,
            "start_time": start,
            "end_time": end,
            "duration_minutes": duration,
            "room": room,
            "confidence": 0.85,
        }

    def _fix_time(self, t: str) -> Optional[str]:
        """Fix a time string to HH:MM format."""
        if not t:
            return None
        t = t.strip()
        if re.match(r'^\d{2}:\d{2}$', t):
            h, m = int(t[:2]), int(t[3:5])
            if 0 <= h <= 23 and 0 <= m <= 59:
                return t
        m = re.match(r'^(\d{1,2}):(\d{2})$', t)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            if 0 <= h <= 23 and 0 <= mi <= 59:
                return f"{h:02d}:{mi:02d}"
        m = re.match(r'^(\d{1,2})$', t)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return f"{h:02d}:00"
        return None

    def _add_minutes(self, time_str: str, minutes: int) -> str:
        parts = time_str.split(':')
        h, m = int(parts[0]), int(parts[1])
        total = h * 60 + m + minutes
        return f"{(total // 60) % 24:02d}:{total % 60:02d}"

    def _duration_minutes(self, start: str, end: str) -> int:
        try:
            s = int(start[:2]) * 60 + int(start[3:5])
            e = int(end[:2]) * 60 + int(end[3:5])
            d = e - s
            return d if d > 0 else d + 1440
        except (ValueError, IndexError):
            return 60

    def _detect_conflicts(self, classes: List[Dict]) -> List[Dict]:
        conflicts = []
        by_day: Dict[str, List] = {}
        for cls in classes:
            by_day.setdefault(cls.get("day", ""), []).append(cls)
        for day, day_cls in by_day.items():
            sorted_cls = sorted(day_cls, key=lambda c: c.get("start_time", ""))
            for i in range(len(sorted_cls) - 1):
                if sorted_cls[i].get("end_time", "") > sorted_cls[i+1].get("start_time", ""):
                    conflicts.append({
                        "day": day,
                        "class_a": sorted_cls[i].get("course", ""),
                        "class_b": sorted_cls[i+1].get("course", ""),
                    })
        return conflicts
