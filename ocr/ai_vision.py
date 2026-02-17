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

# Import valid slot tables for post-extraction snapping
from conversion.ramadan_converter import MT_LOOKUP, FRI_LOOKUP, FRI_DAYS

# Try importing google.genai (new SDK)
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

EXTRACTION_PROMPT = """You are analyzing a university class timetable image (grid format).

Extract ALL classes/courses. For each class identify:
1. **day** — Day of week (Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday)
2. **course** — Course name exactly as written
3. **start_time** — 24h HH:MM
4. **end_time** — 24h HH:MM
5. **room** — Room if visible (else empty string)

═══════════════════════════════════════════════════
  CRITICAL: HOW TO READ TIMES FROM THE GRID
═══════════════════════════════════════════════════

The timetable has a VERTICAL time axis with labeled hour lines (8am, 9am, 10am, …).
Each colored block represents one class.

RULE 1 — EDGE ALIGNMENT:
• If a block's TOP edge is exactly ON a labeled hour line → that hour :00
• If a block's TOP edge is HALFWAY between two hour lines → use :30
• Same logic for the BOTTOM edge (= end time)

RULE 2 — HALF-ROW vs FULL-ROW:
• Look at the DISTANCE from the nearest hour line to the block edge.
  - Edge ON the line = :00   (e.g., top at the "9am" line → 09:00)
  - Edge HALFWAY down = :30  (e.g., top halfway between "9am" and "10am" → 09:30)
  - NEVER use :15 or :45 — university classes always start/end at :00 or :30.

RULE 3 — DURATION CHECK:
After reading start and end, verify the duration is one of:
  60 min (1 grid row)   — e.g., 09:00→10:00, 09:30→10:30
  90 min (1.5 grid rows) — e.g., 08:00→09:30, 09:30→11:00
  120 min (2 grid rows)  — e.g., 08:30→10:30, 14:30→16:30
If your reading gives a non-standard duration (e.g., 70 min, 100 min, 50 min),
re-examine the block edges — you likely misread a :00 as :30 or vice versa.

RULE 4 — AM/PM conversion:
  8am=08:00  9am=09:00  10am=10:00  11am=11:00
  12pm=12:00  1pm=13:00  2pm=14:00  3pm=15:00
  4pm=16:00  5pm=17:00  6pm=18:00  7pm=19:00  8pm=20:00

RULE 5 — COMMON VALID SLOTS (Mon-Thu):
  1-hr (:30 start): 08:30-09:30, 09:30-10:30, 10:30-11:30, 11:30-12:30, 12:30-13:30, 13:30-14:30, 14:30-15:30, 15:30-16:30, 16:30-17:30, 17:30-18:30, 18:30-19:30
  1-hr (:00 start): 09:00-10:00, 10:00-11:00, 11:00-12:00, 12:00-13:00, 13:00-14:00, 14:00-15:00, 15:00-16:00, 16:00-17:00, 17:00-18:00, 18:00-19:00, 19:00-20:00
  1.5-hr: 08:00-09:30, 09:30-11:00, 11:00-12:30, 13:30-15:00, 15:00-16:30, 16:30-18:00, 18:00-19:30, 19:30-21:00
  2-hr: 08:30-10:30, 10:30-12:30, 12:30-14:30, 14:30-16:30, 16:30-18:30, 18:30-20:30
Your extracted (start, end) should match one of these slots. If it doesn't, re-examine.

═══════════════════════════════════════════════════

OTHER RULES:
- Extract EVERY colored block, even if partially visible
- Same course on multiple days = SEPARATE entries
- Two blocks of same color on same day with a gap = TWO separate classes
- Include section numbers / course codes if visible

Return ONLY valid JSON (no markdown, no explanation):
{
    "classes": [
        {
            "day": "Monday",
            "course": "Course Name",
            "start_time": "09:30",
            "end_time": "11:00",
            "room": ""
        }
    ]
}

If unreadable: {"classes": [], "error": "Could not read timetable"}
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

        # ── Snap to nearest valid university slot ──
        start, end = self._snap_to_valid_slot(start, end, day)
        duration = self._duration_minutes(start, end)

        return {
            "day": day,
            "course": course,
            "start_time": start,
            "end_time": end,
            "duration_minutes": duration,
            "room": room,
            "confidence": 0.85,
        }

    def _snap_to_valid_slot(
        self, start: str, end: str, day: str
    ) -> tuple:
        """
        Snap (start, end) to the nearest valid university time slot.
        This corrects small Gemini misreads like 09:00→09:30.
        """
        lookup = FRI_LOOKUP if day in FRI_DAYS else MT_LOOKUP

        # Check exact match first
        if (start, end) in lookup:
            return (start, end)

        s_min = self._time_to_min(start)
        e_min = self._time_to_min(end)
        if s_min is None or e_min is None:
            return (start, end)

        best_slot = None
        best_dist = float('inf')

        for (ks, ke) in lookup.keys():
            ks_min = self._time_to_min(ks)
            ke_min = self._time_to_min(ke)
            if ks_min is None or ke_min is None:
                continue
            dist = abs(s_min - ks_min) + abs(e_min - ke_min)
            if dist < best_dist:
                best_dist = dist
                best_slot = (ks, ke)

        # Allow snap if total deviation is ≤ 35 minutes
        if best_slot and best_dist <= 35:
            logger.info(
                "slot_snap",
                original=f"{start}-{end}",
                snapped=f"{best_slot[0]}-{best_slot[1]}",
                deviation=best_dist,
                day=day,
            )
            return best_slot

        return (start, end)

    @staticmethod
    def _time_to_min(t: str) -> Optional[int]:
        """Convert HH:MM to total minutes."""
        try:
            parts = t.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except (ValueError, IndexError):
            return None

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
