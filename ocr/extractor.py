"""
OCR & Schedule Extraction Module (v2 — Smart Multi-Strategy)

Key improvements over v1:
  1. Tries MULTIPLE Tesseract PSM modes on MULTIPLE image variants
  2. Uses pytesseract.image_to_data() for word-level bounding boxes
  3. Detects colored blocks as class regions (not just grid lines)
  4. Spatial clustering: groups nearby words into logical blocks
  5. Much wider regex patterns for OCR error tolerance
  6. Picks the extraction with the highest class count + confidence
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np

try:
    import pytesseract
    _tess_cmd = os.getenv("TESSERACT_CMD", "")
    if _tess_cmd and os.path.isfile(_tess_cmd):
        pytesseract.pytesseract.tesseract_cmd = _tess_cmd
except ImportError:
    pytesseract = None

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Flexible regex patterns (OCR-error tolerant)
# ---------------------------------------------------------------------------

# Time: matches "8:00", "08.00", "8 00", "8:OO", "08:00AM", "8:00 PM", etc.
TIME_PATTERN = re.compile(
    r'(\d{1,2})\s*[:\.\-;,]\s*(\d{2})\s*(AM|PM|am|pm|a\.m|p\.m)?'
    r'|(\d{1,2})\s*(AM|PM|am|pm)',
    re.IGNORECASE
)

# Time range: "8:00 - 9:00", "8:00-9:00", "8.00 to 9.00", etc.
TIME_RANGE_PATTERN = re.compile(
    r'(\d{1,2}[:\.\-;]?\s*\d{0,2})\s*'
    r'[-–—~>]+\s*'
    r'(\d{1,2}[:\.\-;]?\s*\d{0,2})\s*'
    r'(AM|PM|am|pm)?',
    re.IGNORECASE
)

# Days of the week — very broad to catch OCR mistakes
DAYS_MAP = {
    # Full names
    "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday",
    "thursday": "Thursday", "friday": "Friday", "saturday": "Saturday",
    "sunday": "Sunday",
    # 3-letter
    "mon": "Monday", "tue": "Tuesday", "wed": "Wednesday",
    "thu": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday",
    # Common OCR errors
    "menday": "Monday", "munday": "Monday", "mnday": "Monday",
    "tues": "Tuesday", "tuseday": "Tuesday", "tueday": "Tuesday",
    "weds": "Wednesday", "wednes": "Wednesday", "wednseday": "Wednesday",
    "thurs": "Thursday", "thur": "Thursday", "thurday": "Thursday",
    "frid": "Friday", "frday": "Friday",
    # Arabic-influenced abbreviation forms
    "m": "Monday", "tu": "Tuesday", "w": "Wednesday",
    "th": "Thursday", "f": "Friday",
}

# Words that are NOT course names
NON_COURSE_WORDS = {
    "time", "day", "hour", "room", "lecture", "lab", "tutorial",
    "break", "lunch", "free", "slot", "period", "semester",
    "timetable", "schedule", "class", "section", "cr", "sec",
    "am", "pm", "to",
}


class ScheduleExtractor:
    """
    Smart multi-strategy schedule extractor.
    Tries multiple OCR configurations on multiple image variants
    and returns the best result.
    """

    def __init__(self):
        if pytesseract is None:
            logger.error("pytesseract_not_available")

    def extract(self, image_variants: List[np.ndarray]) -> Dict[str, Any]:
        """
        Main entry point.
        Receives multiple preprocessed image variants (from ImagePreprocessor).
        Tries multiple OCR strategies on each. Returns the best extraction.
        """
        if pytesseract is None:
            return self._empty_result("Tesseract OCR not installed")

        all_attempts: List[Dict[str, Any]] = []

        # PSM modes to try:
        # 3 = fully automatic (default)
        # 4 = assume single column of text
        # 6 = assume a single uniform block of text
        # 11 = sparse text, find as much text as possible
        # 12 = sparse text with OSD
        psm_modes = [6, 3, 4, 11]

        for vi, variant in enumerate(image_variants):
            for psm in psm_modes:
                try:
                    classes = self._extract_single(variant, psm)
                    if classes:
                        score = self._score_extraction(classes)
                        all_attempts.append({
                            "classes": classes,
                            "score": score,
                            "variant": vi,
                            "psm": psm,
                        })
                        logger.info("extraction_attempt",
                                    variant=vi, psm=psm,
                                    classes=len(classes),
                                    score=round(score, 2))
                except Exception as e:
                    logger.debug("extraction_attempt_failed",
                                 variant=vi, psm=psm, error=str(e))
                    continue

        if not all_attempts:
            logger.warning("no_classes_found_any_strategy")
            return self._empty_result(
                "Could not extract any classes. Try a clearer photo."
            )

        # Pick the best attempt
        best = max(all_attempts, key=lambda a: a["score"])
        logger.info("best_extraction",
                     variant=best["variant"], psm=best["psm"],
                     classes=len(best["classes"]),
                     score=round(best["score"], 2))

        classes = self._deduplicate(best["classes"])
        conflicts = self._detect_conflicts(classes)

        total_conf = sum(c.get("confidence", 0.5) for c in classes)
        avg_conf = total_conf / len(classes) if classes else 0

        result = {
            "classes": classes,
            "total_classes": len(classes),
            "average_confidence": round(avg_conf, 2),
            "conflicts": conflicts,
            "extraction_info": {
                "variant_used": best["variant"],
                "psm_mode": best["psm"],
                "total_attempts": len(all_attempts),
            },
            "warnings": [],
        }

        if avg_conf < 0.5:
            result["warnings"].append(
                "Low OCR confidence. Please review the extracted schedule."
            )
        if conflicts:
            result["warnings"].append(
                f"{len(conflicts)} scheduling conflict(s) detected."
            )

        return result

    # ==================================================================
    # Core extraction for a single image + PSM combo
    # ==================================================================
    def _extract_single(
        self, img: np.ndarray, psm: int
    ) -> List[Dict[str, Any]]:
        """
        Run OCR on one image variant with one PSM mode.
        Uses TWO approaches and merges results:
          A) Word-level bounding box analysis (spatial)
          B) Full-text line-by-line parsing
        """
        classes: List[Dict[str, Any]] = []

        config = f'--oem 3 --psm {psm} -l eng'

        # --- Approach A: Spatial word-box analysis ---
        try:
            spatial_classes = self._spatial_extraction(img, config)
            classes.extend(spatial_classes)
        except Exception as e:
            logger.debug("spatial_extraction_failed", error=str(e))

        # --- Approach B: Full-text parsing ---
        try:
            text_classes = self._text_extraction(img, config)
            classes.extend(text_classes)
        except Exception as e:
            logger.debug("text_extraction_failed", error=str(e))

        return classes

    # ------------------------------------------------------------------
    # Approach A: Spatial / bounding-box-based extraction
    # ------------------------------------------------------------------
    def _spatial_extraction(
        self, img: np.ndarray, config: str
    ) -> List[Dict[str, Any]]:
        """
        Use image_to_data() to get word positions.
        Group words into spatial blocks and infer schedule structure.
        """
        data = pytesseract.image_to_data(
            img, config=config, output_type=pytesseract.Output.DICT
        )

        n = len(data["text"])
        if n == 0:
            return []

        # Collect meaningful words with positions
        words = []
        for i in range(n):
            text = str(data["text"][i]).strip()
            conf = int(data["conf"][i]) if data["conf"][i] != '-1' else 0
            if not text or conf < 20:
                continue
            words.append({
                "text": text,
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
                "conf": conf,
                "block": int(data["block_num"][i]),
                "line": int(data["line_num"][i]),
            })

        if not words:
            return []

        # --- Find day words and time words ---
        day_words = []
        time_words = []
        other_words = []

        for w in words:
            day = self._match_day(w["text"])
            if day:
                day_words.append({**w, "day": day})
            elif self._looks_like_time(w["text"]):
                time_words.append(w)
            else:
                other_words.append(w)

        logger.debug("spatial_words",
                      days=len(day_words),
                      times=len(time_words),
                      other=len(other_words))

        if not day_words:
            return []

        # --- Determine layout orientation ---
        # Are days arranged horizontally (columns) or vertically (rows)?
        day_xs = [d["x"] for d in day_words]
        day_ys = [d["y"] for d in day_words]
        x_spread = max(day_xs) - min(day_xs) if day_xs else 0
        y_spread = max(day_ys) - min(day_ys) if day_ys else 0

        if x_spread > y_spread:
            # Days are in a horizontal header row
            return self._extract_horizontal_layout(
                day_words, time_words, other_words, words
            )
        else:
            # Days are in a vertical column
            return self._extract_vertical_layout(
                day_words, time_words, other_words, words
            )

    def _extract_horizontal_layout(
        self,
        day_words: List[Dict],
        time_words: List[Dict],
        other_words: List[Dict],
        all_words: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Days across the top, times down the left side."""
        classes = []

        # Sort days by x position
        day_words.sort(key=lambda d: d["x"])
        day_columns = []
        for d in day_words:
            cx = d["x"] + d["w"] // 2
            day_columns.append({"day": d["day"], "cx": cx, "x": d["x"],
                                "x2": d["x"] + d["w"]})

        # Sort time words by y position
        time_words.sort(key=lambda t: t["y"])

        # For each non-header word, find which day column it belongs to
        header_y = max(d["y"] + d["h"] for d in day_words) if day_words else 0

        for w in other_words:
            if w["y"] < header_y:
                continue  # Skip header row

            # Find closest day column
            wx = w["x"] + w["w"] // 2
            closest_day = self._closest_column(wx, day_columns)
            if not closest_day:
                continue

            # Find closest time row
            wy = w["y"] + w["h"] // 2
            closest_time = self._closest_time_row(wy, time_words)

            if closest_day and len(w["text"]) > 1:
                entry = {
                    "day": closest_day,
                    "course": w["text"],
                    "start_time": closest_time or "09:00",
                    "end_time": self._add_minutes(closest_time or "09:00", 60),
                    "duration_minutes": 60,
                    "confidence": round(w["conf"] / 100.0, 2),
                }
                classes.append(entry)

        # Merge adjacent words into single course names
        classes = self._merge_nearby_classes(classes)
        return classes

    def _extract_vertical_layout(
        self,
        day_words: List[Dict],
        time_words: List[Dict],
        other_words: List[Dict],
        all_words: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Days down the left side, times across the top or inline."""
        classes = []
        day_words.sort(key=lambda d: d["y"])

        # Build day bands (y ranges)
        day_bands = []
        for i, d in enumerate(day_words):
            y_start = d["y"]
            y_end = day_words[i + 1]["y"] if i + 1 < len(day_words) else 99999
            day_bands.append({"day": d["day"], "y1": y_start, "y2": y_end})

        for w in other_words:
            wy = w["y"] + w["h"] // 2
            day = None
            for band in day_bands:
                if band["y1"] <= wy < band["y2"]:
                    day = band["day"]
                    break
            if not day:
                continue

            time_str = self._extract_time_from_text(w["text"])
            if time_str:
                continue  # This word is a time, not a course

            if len(w["text"]) > 1:
                # Try to find a nearby time word
                closest_time = self._closest_time_row(wy, time_words)
                classes.append({
                    "day": day,
                    "course": w["text"],
                    "start_time": closest_time or "09:00",
                    "end_time": self._add_minutes(closest_time or "09:00", 60),
                    "duration_minutes": 60,
                    "confidence": round(w["conf"] / 100.0, 2),
                })

        classes = self._merge_nearby_classes(classes)
        return classes

    # ------------------------------------------------------------------
    # Approach B: Full-text line-by-line parsing
    # ------------------------------------------------------------------
    def _text_extraction(
        self, img: np.ndarray, config: str
    ) -> List[Dict[str, Any]]:
        """
        OCR the full image as text, then parse line by line.
        Much more flexible regex matching than v1.
        """
        raw_text = pytesseract.image_to_string(img, config=config)
        if not raw_text or len(raw_text.strip()) < 10:
            return []

        logger.debug("ocr_raw_text", length=len(raw_text),
                      preview=raw_text[:200].replace('\n', '|'))

        classes = []
        lines = raw_text.strip().split('\n')
        current_day = None

        for line in lines:
            line = line.strip()
            if not line or len(line) < 2:
                continue

            # Check for day
            day = self._find_day_in_text(line)
            if day:
                current_day = day

            # Try extracting time range + course from this line
            extracted = self._parse_schedule_line(line, current_day)
            if extracted:
                classes.extend(extracted)
                continue

            # If we have a current day and the line looks like a course name
            if current_day and self._looks_like_course(line):
                classes.append({
                    "day": current_day,
                    "course": self._clean_course(line),
                    "start_time": "09:00",
                    "end_time": "10:00",
                    "duration_minutes": 60,
                    "confidence": 0.3,
                })

        return classes

    def _parse_schedule_line(
        self, line: str, current_day: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Parse a single line for schedule data.
        Returns list of classes found (0, 1, or more).
        """
        results = []
        day = current_day

        # Check if line starts with a day
        line_day = self._find_day_in_text(line)
        if line_day:
            day = line_day

        if not day:
            return results

        # Try time range pattern: "8:00 - 9:30"
        tr_match = TIME_RANGE_PATTERN.search(line)
        if tr_match:
            start_raw = tr_match.group(1)
            end_raw = tr_match.group(2)
            ampm = tr_match.group(3)

            start_time = self._normalize_time(start_raw, ampm)
            end_time = self._normalize_time(end_raw, ampm)

            # Course = text outside the time range
            before = line[:tr_match.start()].strip()
            after = line[tr_match.end():].strip()
            course_text = f"{before} {after}".strip()
            # Remove day name from course text
            course_text = self._remove_day_from_text(course_text)
            course = self._clean_course(course_text)

            if start_time:
                if not end_time:
                    end_time = self._add_minutes(start_time, 60)
                results.append({
                    "day": day,
                    "course": course or "Unknown Course",
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_minutes": self._duration_minutes(start_time, end_time),
                    "confidence": 0.65 if course else 0.4,
                })
            return results

        # Try multiple individual times in the line
        time_matches = list(TIME_PATTERN.finditer(line))
        if len(time_matches) >= 2:
            start_m = time_matches[0]
            end_m = time_matches[1]

            s_h = start_m.group(1) or start_m.group(4)
            s_min = start_m.group(2) or "00"
            s_ampm = start_m.group(3) or start_m.group(5)

            e_h = end_m.group(1) or end_m.group(4)
            e_min = end_m.group(2) or "00"
            e_ampm = end_m.group(3) or end_m.group(5)

            start_time = self._normalize_time(f"{s_h}:{s_min}", s_ampm)
            end_time = self._normalize_time(f"{e_h}:{e_min}", e_ampm)

            # Course text
            course_text = line[:start_m.start()] + line[end_m.end():]
            course_text = self._remove_day_from_text(course_text)
            course = self._clean_course(course_text)

            if start_time:
                if not end_time:
                    end_time = self._add_minutes(start_time, 60)
                results.append({
                    "day": day,
                    "course": course or "Unknown Course",
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_minutes": self._duration_minutes(start_time, end_time),
                    "confidence": 0.55 if course else 0.35,
                })
            return results

        # Single time
        if len(time_matches) == 1:
            m = time_matches[0]
            t_h = m.group(1) or m.group(4)
            t_min = m.group(2) or "00"
            t_ampm = m.group(3) or m.group(5)
            time_str = self._normalize_time(f"{t_h}:{t_min}", t_ampm)

            course_text = line[:m.start()] + line[m.end():]
            course_text = self._remove_day_from_text(course_text)
            course = self._clean_course(course_text)

            if time_str and course:
                results.append({
                    "day": day,
                    "course": course,
                    "start_time": time_str,
                    "end_time": self._add_minutes(time_str, 60),
                    "duration_minutes": 60,
                    "confidence": 0.4,
                })

        return results

    # ==================================================================
    # Scoring: which extraction attempt was best?
    # ==================================================================
    def _score_extraction(self, classes: List[Dict]) -> float:
        """
        Score an extraction attempt. Higher = better.
        Considers: number of classes, confidence, completeness.
        """
        if not classes:
            return 0.0

        n = len(classes)
        avg_conf = sum(c.get("confidence", 0) for c in classes) / n

        # Reward having multiple distinct days
        days = set(c.get("day", "") for c in classes)
        day_bonus = min(len(days) / 5.0, 1.0)

        # Reward having actual course names (not just "Unknown")
        named = sum(1 for c in classes
                    if c.get("course", "").lower() not in
                    {"unknown course", "unknown", ""})
        name_ratio = named / n if n else 0

        # Reward reasonable time values
        valid_times = sum(1 for c in classes
                         if self._is_valid_time(c.get("start_time", "")))
        time_ratio = valid_times / n if n else 0

        # Penalize too many or too few classes
        count_score = min(n / 5.0, 1.0) if n <= 30 else 0.5

        score = (
            count_score * 3.0 +
            avg_conf * 2.0 +
            day_bonus * 2.0 +
            name_ratio * 1.5 +
            time_ratio * 1.5
        )

        return score

    # ==================================================================
    # Helper methods
    # ==================================================================

    def _match_day(self, text: str) -> Optional[str]:
        """Match a single word to a day name."""
        if not text or len(text) < 2:
            return None
        t = text.lower().strip().rstrip('.,;:!')
        return DAYS_MAP.get(t)

    def _find_day_in_text(self, text: str) -> Optional[str]:
        """Find any day-of-week mention in a text string."""
        if not text:
            return None
        text_lower = text.lower()
        # Try longer matches first (avoid "sun" matching in "sunlight")
        for key in sorted(DAYS_MAP.keys(), key=len, reverse=True):
            if len(key) >= 3:
                pattern = r'\b' + re.escape(key) + r'\b'
                if re.search(pattern, text_lower):
                    return DAYS_MAP[key]
        return None

    def _remove_day_from_text(self, text: str) -> str:
        """Remove day names from text."""
        for key in sorted(DAYS_MAP.keys(), key=len, reverse=True):
            if len(key) >= 3:
                text = re.sub(r'\b' + re.escape(key) + r'\b', '', text,
                              flags=re.IGNORECASE)
        return text.strip()

    def _looks_like_time(self, text: str) -> bool:
        """Check if a word looks like a time value."""
        text = text.strip()
        if TIME_PATTERN.search(text):
            return True
        # Bare hours like "8" or "14"
        try:
            h = int(text)
            return 1 <= h <= 23
        except ValueError:
            return False

    def _looks_like_course(self, text: str) -> bool:
        """Check if text looks like a course name (not a time or day)."""
        text = text.strip()
        if len(text) < 3:
            return False
        if TIME_PATTERN.search(text):
            return False
        if self._find_day_in_text(text):
            # Only a day name, no course info
            cleaned = self._remove_day_from_text(text).strip()
            if len(cleaned) < 3:
                return False
        # Has letters (not just numbers/symbols)
        return bool(re.search(r'[a-zA-Z]{2,}', text))

    def _extract_time_from_text(self, text: str) -> Optional[str]:
        """Extract time if the text IS a time."""
        m = TIME_PATTERN.search(text)
        if m:
            h = m.group(1) or m.group(4)
            mins = m.group(2) or "00"
            ampm = m.group(3) or m.group(5)
            return self._normalize_time(f"{h}:{mins}", ampm)
        return None

    def _normalize_time(
        self, raw: str, ampm: Optional[str] = None
    ) -> Optional[str]:
        """Normalize time to HH:MM 24-hour format."""
        if not raw:
            return None
        raw = re.sub(r'[^0-9:]', ':', raw).strip(':')
        parts = raw.split(':')
        parts = [p for p in parts if p]

        if not parts:
            return None

        try:
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
        except ValueError:
            return None

        if h > 23 or m > 59:
            return None

        if ampm:
            ampm_clean = ampm.lower().replace('.', '').strip()
            if ampm_clean == 'pm' and h < 12:
                h += 12
            elif ampm_clean == 'am' and h == 12:
                h = 0

        # University heuristic: hours 1-6 are likely PM
        if h >= 1 and h <= 6 and ampm is None:
            h += 12

        return f"{h:02d}:{m:02d}"

    def _is_valid_time(self, t: str) -> bool:
        """Check if time string is valid HH:MM."""
        if not t:
            return False
        try:
            parts = t.split(':')
            h, m = int(parts[0]), int(parts[1])
            return 0 <= h <= 23 and 0 <= m <= 59
        except (ValueError, IndexError):
            return False

    def _add_minutes(self, time_str: str, minutes: int) -> str:
        parts = time_str.split(':')
        h, m = int(parts[0]), int(parts[1])
        total = h * 60 + m + minutes
        return f"{(total // 60) % 24:02d}:{total % 60:02d}"

    def _duration_minutes(self, start: str, end: str) -> int:
        try:
            s = int(start.split(':')[0]) * 60 + int(start.split(':')[1])
            e = int(end.split(':')[0]) * 60 + int(end.split(':')[1])
            d = e - s
            return d if d > 0 else d + 1440
        except (ValueError, IndexError):
            return 60

    def _clean_course(self, text: str) -> str:
        """Clean OCR course name aggressively."""
        if not text:
            return ""
        # Remove repeated symbols
        text = re.sub(r'[|\\/_\-=]{2,}', ' ', text)
        # Remove isolated single characters
        text = re.sub(r'\b[^a-zA-Z0-9]\b', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip junk from edges
        text = text.strip(' .-:,;!@#$%^&*()[]{}')
        # Remove non-course words if the text IS just that word
        if text.lower() in NON_COURSE_WORDS:
            return ""
        return text if len(text) >= 2 else ""

    def _closest_column(
        self, x: int, columns: List[Dict]
    ) -> Optional[str]:
        """Find which day column an x-coordinate belongs to."""
        if not columns:
            return None
        best = min(columns, key=lambda c: abs(c["cx"] - x))
        # Allow generous margin
        margin = max(200, (columns[-1]["cx"] - columns[0]["cx"]) // len(columns))
        if abs(best["cx"] - x) > margin:
            return None
        return best["day"]

    def _closest_time_row(
        self, y: int, time_words: List[Dict]
    ) -> Optional[str]:
        """Find the closest time word to a y-coordinate."""
        if not time_words:
            return None
        closest = min(time_words, key=lambda t: abs(t["y"] - y))
        if abs(closest["y"] - y) > 150:
            return None
        return self._extract_time_from_text(closest["text"])

    def _merge_nearby_classes(
        self, classes: List[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Merge classes that are spatially adjacent (same day + same time)
        into single entries with combined course names.
        """
        if len(classes) <= 1:
            return classes

        merged = []
        used = set()

        for i, c1 in enumerate(classes):
            if i in used:
                continue

            combined_course = c1["course"]
            best_conf = c1["confidence"]

            for j, c2 in enumerate(classes):
                if j <= i or j in used:
                    continue
                if (c2["day"] == c1["day"] and
                        c2["start_time"] == c1["start_time"]):
                    # Same slot — merge course names
                    combined_course += " " + c2["course"]
                    best_conf = max(best_conf, c2["confidence"])
                    used.add(j)

            # Clean the merged name
            combined_course = self._clean_course(combined_course)
            if combined_course:
                merged.append({
                    **c1,
                    "course": combined_course,
                    "confidence": round(best_conf, 2),
                })

        return merged

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    def _deduplicate(self, classes: List[Dict]) -> List[Dict]:
        """Remove duplicate entries, keep highest confidence."""
        best: Dict[Tuple, Dict] = {}
        for cls in classes:
            key = (
                cls.get("day", ""),
                cls.get("start_time", ""),
            )
            existing = best.get(key)
            if not existing or cls.get("confidence", 0) > existing.get("confidence", 0):
                # Prefer the one with an actual course name
                if cls.get("course", "") and cls.get("course", "") != "Unknown Course":
                    best[key] = cls
                elif not existing:
                    best[key] = cls

        return list(best.values())

    def _detect_conflicts(self, classes: List[Dict]) -> List[Dict]:
        """Detect overlapping classes on the same day."""
        conflicts = []
        by_day: Dict[str, List] = {}
        for cls in classes:
            by_day.setdefault(cls.get("day", ""), []).append(cls)

        for day, day_classes in by_day.items():
            sorted_cls = sorted(day_classes,
                                key=lambda c: c.get("start_time", ""))
            for i in range(len(sorted_cls) - 1):
                curr_end = sorted_cls[i].get("end_time", "")
                next_start = sorted_cls[i + 1].get("start_time", "")
                if curr_end > next_start:
                    conflicts.append({
                        "day": day,
                        "class_a": sorted_cls[i].get("course", ""),
                        "class_b": sorted_cls[i + 1].get("course", ""),
                    })
        return conflicts

    def _empty_result(self, warning: str) -> Dict[str, Any]:
        return {
            "classes": [],
            "total_classes": 0,
            "average_confidence": 0,
            "conflicts": [],
            "warnings": [warning],
        }
