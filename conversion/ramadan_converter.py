"""
Ramadan Conversion Engine
Uses the EXACT official university Ramadan timing tables.
Separate lookup for Monday-Thursday vs Friday.
NO arithmetic — pure lookup.
"""

from typing import Dict, Any, List, Optional, Tuple

import structlog

logger = structlog.get_logger()


# ===========================================================================
# OFFICIAL RAMADAN TIMING — MONDAY to THURSDAY (also Sunday)
# ===========================================================================

MT_LOOKUP: Dict[Tuple[str, str], Tuple[str, str]] = {
    # ===== 1-hour slots (:30 boundaries) =====
    ("08:30", "09:30"): ("08:30", "09:10"),
    ("09:30", "10:30"): ("09:10", "09:50"),
    ("10:30", "11:30"): ("09:50", "10:30"),
    ("11:30", "12:30"): ("10:30", "11:10"),
    ("12:30", "13:30"): ("11:10", "11:50"),
    ("13:30", "14:30"): ("11:50", "12:30"),
    ("14:30", "15:30"): ("12:30", "13:10"),
    ("15:30", "16:30"): ("13:10", "13:50"),
    ("16:30", "17:30"): ("13:50", "14:30"),
    ("17:30", "18:30"): ("14:30", "15:10"),
    ("18:30", "19:30"): ("15:10", "15:50"),
    ("19:30", "20:30"): ("15:50", "16:30"),
    ("20:30", "21:30"): ("16:30", "17:10"),

    # ===== 1-hour slots (:00 boundaries) =====
    ("09:00", "10:00"): ("08:50", "09:30"),
    ("10:00", "11:00"): ("09:30", "10:10"),
    ("11:00", "12:00"): ("10:10", "10:50"),
    ("12:00", "13:00"): ("10:50", "11:30"),
    ("13:00", "14:00"): ("11:30", "12:10"),
    ("14:00", "15:00"): ("12:10", "12:50"),
    ("15:00", "16:00"): ("12:50", "13:30"),
    ("16:00", "17:00"): ("13:30", "14:10"),
    ("17:00", "18:00"): ("14:10", "14:50"),
    ("18:00", "19:00"): ("14:50", "15:30"),
    ("19:00", "20:00"): ("15:30", "16:10"),
    ("20:00", "21:00"): ("16:10", "16:50"),
    ("21:00", "22:00"): ("16:50", "17:30"),

    # ===== 1.5-hour (90 min) slots =====
    ("08:00", "09:30"): ("08:00", "09:00"),
    ("09:30", "11:00"): ("09:10", "10:10"),
    ("11:00", "12:30"): ("10:10", "11:10"),
    ("13:30", "15:00"): ("11:50", "12:50"),
    ("15:00", "16:30"): ("12:50", "13:50"),
    ("16:30", "18:00"): ("13:50", "14:50"),
    ("18:00", "19:30"): ("14:50", "15:50"),
    ("19:30", "21:00"): ("15:50", "16:50"),
    ("21:00", "22:30"): ("16:50", "17:50"),

    # ===== 2-hour slots (:30 boundaries) =====
    ("08:30", "10:30"): ("08:30", "09:50"),
    ("10:30", "12:30"): ("09:50", "11:10"),
    ("12:30", "14:30"): ("11:10", "12:30"),
    ("14:30", "16:30"): ("12:30", "13:50"),
    ("16:30", "18:30"): ("13:50", "15:10"),
    ("18:30", "20:30"): ("15:10", "16:30"),
    ("20:30", "22:30"): ("16:30", "17:50"),
}


# ===========================================================================
# OFFICIAL RAMADAN TIMING — FRIDAY
# ===========================================================================

FRI_LOOKUP: Dict[Tuple[str, str], Tuple[str, str]] = {
    # ===== Morning block =====
    ("08:00", "09:00"): ("08:00", "08:40"),
    ("08:00", "10:00"): ("08:00", "09:20"),
    ("08:00", "12:00"): ("08:00", "10:40"),
    ("08:30", "12:00"): ("08:30", "10:50"),
    ("09:00", "12:00"): ("08:40", "10:40"),
    ("10:00", "12:00"): ("09:20", "10:40"),

    # ===== Afternoon block =====
    ("14:00", "15:00"): ("10:40", "11:20"),
    ("14:00", "16:00"): ("10:40", "12:00"),
    ("14:00", "17:00"): ("14:00", "16:00"),
    ("14:00", "18:00"): ("14:00", "16:40"),
    ("14:00", "19:00"): ("14:00", "17:20"),
    ("14:30", "16:30"): ("14:00", "15:20"),
    ("15:00", "16:00"): ("14:00", "14:40"),
    ("15:00", "17:00"): ("14:00", "15:20"),
    ("15:00", "18:00"): ("14:00", "16:00"),
    ("15:00", "19:00"): ("14:00", "16:40"),
    ("16:00", "18:00"): ("14:40", "16:00"),
    ("16:00", "19:00"): ("14:40", "16:40"),
    ("16:30", "18:30"): ("15:20", "16:40"),
    ("17:30", "20:30"): ("15:30", "17:30"),
    ("18:00", "19:00"): ("16:40", "17:20"),
    ("18:00", "20:00"): ("16:00", "17:20"),
    ("18:00", "21:00"): ("16:00", "18:00"),
    ("18:30", "20:30"): ("16:00", "17:20"),
    ("19:00", "20:00"): ("16:40", "17:20"),
}

# Days that use the Monday-Thursday table
MT_DAYS = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Saturday"}
FRI_DAYS = {"Friday"}


class RamadanConverter:
    """
    Converts a regular timetable to Ramadan timing using
    the official university lookup tables.
    Separate tables for Mon-Thu and Friday.
    """

    def __init__(self):
        self.mt_lookup = dict(MT_LOOKUP)
        self.fri_lookup = dict(FRI_LOOKUP)
        self._build_fuzzy_index()

    def _build_fuzzy_index(self):
        """Pre-compute time-to-minutes mapping for fuzzy matching."""
        self._mt_index = []
        for (s, e), (rs, re_) in self.mt_lookup.items():
            s_min = self._to_minutes(s)
            e_min = self._to_minutes(e)
            self._mt_index.append((s_min, e_min, s, e, rs, re_))

        self._fri_index = []
        for (s, e), (rs, re_) in self.fri_lookup.items():
            s_min = self._to_minutes(s)
            e_min = self._to_minutes(e)
            self._fri_index.append((s_min, e_min, s, e, rs, re_))

    def convert(self, schedule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert an entire schedule dict.
        Each class gets ramadan_start, ramadan_end, and mapping_type fields.
        """
        converted_classes = []
        stats = {
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "unmatched": 0,
        }

        for cls in schedule.get("classes", []):
            converted = self._convert_class(cls)
            converted_classes.append(converted)

            mt = converted.get("mapping_type", "unmatched")
            if mt == "exact":
                stats["exact_matches"] += 1
            elif mt == "fuzzy":
                stats["fuzzy_matches"] += 1
            else:
                stats["unmatched"] += 1

        result = {
            "classes": converted_classes,
            "total_classes": len(converted_classes),
            "conversion_stats": stats,
        }

        if stats["unmatched"] > 0:
            result["warnings"] = [
                f"{stats['unmatched']} class(es) could not be mapped. "
                "Please verify manually."
            ]

        return result

    def _convert_class(self, cls: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single class entry using day-appropriate table."""
        original_start = cls.get("start_time", "")
        original_end = cls.get("end_time", "")
        day = cls.get("day", "")

        # Normalize
        original_start = self._normalize(original_start)
        original_end = self._normalize(original_end)

        # Pick the right lookup table based on day
        is_friday = day in FRI_DAYS
        lookup = self.fri_lookup if is_friday else self.mt_lookup
        index = self._fri_index if is_friday else self._mt_index

        # 1. Exact lookup
        key = (original_start, original_end)
        if key in lookup:
            ram_start, ram_end = lookup[key]
            return {
                **cls,
                "original_start": original_start,
                "original_end": original_end,
                "ramadan_start": ram_start,
                "ramadan_end": ram_end,
                "duration_minutes": self._duration(ram_start, ram_end),
                "mapping_type": "exact",
            }

        # 2. Fuzzy match (nearest start+end within tolerance)
        fuzzy = self._fuzzy_match(
            original_start, original_end, index, tolerance=20
        )
        if fuzzy:
            ram_start, ram_end, matched_key = fuzzy
            logger.info(
                "fuzzy_match",
                original=f"{original_start}-{original_end}",
                matched=f"{matched_key[0]}-{matched_key[1]}",
                day=day,
            )
            return {
                **cls,
                "original_start": original_start,
                "original_end": original_end,
                "ramadan_start": ram_start,
                "ramadan_end": ram_end,
                "duration_minutes": self._duration(ram_start, ram_end),
                "mapping_type": "fuzzy",
                "fuzzy_matched_from": f"{matched_key[0]}-{matched_key[1]}",
            }

        # 3. Unmatched — return original with flag
        logger.warning(
            "no_ramadan_mapping",
            start=original_start,
            end=original_end,
            day=day,
        )
        return {
            **cls,
            "original_start": original_start,
            "original_end": original_end,
            "ramadan_start": original_start,
            "ramadan_end": original_end,
            "mapping_type": "unmatched",
            "warning": "No Ramadan mapping found. Using original times.",
        }

    def _fuzzy_match(
        self, start: str, end: str, index: list, tolerance: int = 20
    ) -> Optional[Tuple[str, str, Tuple[str, str]]]:
        """
        Find the closest matching entry in the given lookup index
        within a tolerance (minutes).
        """
        s_min = self._to_minutes(start)
        e_min = self._to_minutes(end)

        if s_min is None or e_min is None:
            return None

        best = None
        best_dist = float('inf')

        for idx_s, idx_e, key_s, key_e, ram_s, ram_e in index:
            dist = abs(s_min - idx_s) + abs(e_min - idx_e)
            if dist < best_dist and dist <= tolerance * 2:
                best_dist = dist
                best = (ram_s, ram_e, (key_s, key_e))

        return best

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(time_str: str) -> str:
        """Normalize time to HH:MM."""
        time_str = time_str.strip().replace('.', ':')
        parts = time_str.split(':')
        if len(parts) == 2:
            try:
                h, m = int(parts[0]), int(parts[1])
                return f"{h:02d}:{m:02d}"
            except ValueError:
                pass
        return time_str

    @staticmethod
    def _to_minutes(time_str: str) -> Optional[int]:
        """Convert HH:MM to total minutes."""
        try:
            parts = time_str.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _duration(start: str, end: str) -> int:
        """Duration in minutes."""
        try:
            s = int(start.split(':')[0]) * 60 + int(start.split(':')[1])
            e = int(end.split(':')[0]) * 60 + int(end.split(':')[1])
            d = e - s
            return d if d > 0 else d + 1440
        except (ValueError, IndexError):
            return 0

    def get_table(self, day: str = "Monday") -> Dict:
        """Return the lookup table for a given day."""
        lookup = self.fri_lookup if day in FRI_DAYS else self.mt_lookup
        return {
            f"{k[0]}-{k[1]}": {"ramadan_start": v[0], "ramadan_end": v[1]}
            for k, v in sorted(lookup.items())
        }
