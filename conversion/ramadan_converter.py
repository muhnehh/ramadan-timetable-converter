"""
Ramadan Conversion Engine
Lookup-based official Ramadan timing conversion.
NO arithmetic shifting — uses an exact mapping table.
"""

import re
from typing import Dict, Any, List, Optional, Tuple

import structlog

logger = structlog.get_logger()


# ===========================================================================
# OFFICIAL RAMADAN TIMING LOOKUP TABLE
# Maps original class times → Ramadan class times
# Based on UAE university Ramadan schedule standards
#
# This covers the most common patterns:
# - 1-hour lectures
# - 1.5-hour (90-minute) lectures
# - 2-hour lectures
# - 3-hour labs/lectures
#
# Format: (original_start, original_end) → (ramadan_start, ramadan_end)
# All times in 24-hour HH:MM format.
# ===========================================================================

RAMADAN_LOOKUP: Dict[Tuple[str, str], Tuple[str, str]] = {
    # ============ Morning Block ============
    # 1-hour slots
    ("08:00", "09:00"): ("09:00", "09:45"),
    ("09:00", "10:00"): ("09:50", "10:35"),
    ("10:00", "11:00"): ("10:40", "11:25"),
    ("11:00", "12:00"): ("11:30", "12:15"),
    ("12:00", "13:00"): ("12:20", "13:05"),
    ("13:00", "14:00"): ("13:10", "13:55"),
    ("14:00", "15:00"): ("14:00", "14:45"),
    ("15:00", "16:00"): ("14:50", "15:35"),
    ("16:00", "17:00"): ("15:40", "16:25"),
    ("17:00", "18:00"): ("16:30", "17:15"),
    ("18:00", "19:00"): ("17:20", "18:05"),
    ("19:00", "20:00"): ("18:10", "18:55"),

    # ============ 90-minute (1.5 hr) slots ============
    ("08:00", "09:30"): ("09:00", "10:10"),
    ("09:00", "10:30"): ("09:50", "11:00"),
    ("09:30", "11:00"): ("09:50", "11:00"),
    ("10:00", "11:30"): ("10:40", "11:50"),
    ("10:30", "12:00"): ("10:40", "11:50"),
    ("11:00", "12:30"): ("11:30", "12:40"),
    ("11:30", "13:00"): ("11:30", "12:40"),
    ("12:00", "13:30"): ("12:20", "13:30"),
    ("12:30", "14:00"): ("12:20", "13:30"),
    ("13:00", "14:30"): ("13:10", "14:20"),
    ("13:30", "15:00"): ("13:10", "14:20"),
    ("14:00", "15:30"): ("14:00", "15:10"),
    ("14:30", "16:00"): ("14:00", "15:10"),
    ("15:00", "16:30"): ("14:50", "16:00"),
    ("15:30", "17:00"): ("14:50", "16:00"),
    ("16:00", "17:30"): ("15:40", "16:50"),
    ("16:30", "18:00"): ("15:40", "16:50"),
    ("17:00", "18:30"): ("16:30", "17:40"),

    # ============ 2-hour slots ============
    ("08:00", "10:00"): ("09:00", "10:35"),
    ("09:00", "11:00"): ("09:50", "11:25"),
    ("10:00", "12:00"): ("10:40", "12:15"),
    ("11:00", "13:00"): ("11:30", "13:05"),
    ("12:00", "14:00"): ("12:20", "13:55"),
    ("13:00", "15:00"): ("13:10", "14:45"),
    ("14:00", "16:00"): ("14:00", "15:35"),
    ("15:00", "17:00"): ("14:50", "16:25"),
    ("16:00", "18:00"): ("15:40", "17:15"),
    ("17:00", "19:00"): ("16:30", "18:05"),

    # ============ 3-hour slots (labs) ============
    ("08:00", "11:00"): ("09:00", "11:25"),
    ("09:00", "12:00"): ("09:50", "12:15"),
    ("10:00", "13:00"): ("10:40", "13:05"),
    ("11:00", "14:00"): ("11:30", "13:55"),
    ("12:00", "15:00"): ("12:20", "14:45"),
    ("13:00", "16:00"): ("13:10", "15:35"),
    ("14:00", "17:00"): ("14:00", "16:25"),
    ("15:00", "18:00"): ("14:50", "17:15"),
    ("16:00", "19:00"): ("15:40", "18:05"),

    # ============ Half-hour aligned slots ============
    ("08:30", "09:30"): ("09:00", "09:45"),
    ("09:30", "10:30"): ("09:50", "10:35"),
    ("10:30", "11:30"): ("10:40", "11:25"),
    ("11:30", "12:30"): ("11:30", "12:15"),
    ("12:30", "13:30"): ("12:20", "13:05"),
    ("13:30", "14:30"): ("13:10", "13:55"),
    ("14:30", "15:30"): ("14:00", "14:45"),
    ("15:30", "16:30"): ("14:50", "15:35"),
    ("16:30", "17:30"): ("15:40", "16:25"),
    ("17:30", "18:30"): ("16:30", "17:15"),

    # ============ 50-minute lecture variants ============
    ("08:00", "08:50"): ("09:00", "09:45"),
    ("09:00", "09:50"): ("09:50", "10:35"),
    ("10:00", "10:50"): ("10:40", "11:25"),
    ("11:00", "11:50"): ("11:30", "12:15"),
    ("12:00", "12:50"): ("12:20", "13:05"),
    ("13:00", "13:50"): ("13:10", "13:55"),
    ("14:00", "14:50"): ("14:00", "14:45"),
    ("15:00", "15:50"): ("14:50", "15:35"),
    ("16:00", "16:50"): ("15:40", "16:25"),
    ("17:00", "17:50"): ("16:30", "17:15"),
}


class RamadanConverter:
    """
    Converts a regular timetable to Ramadan timing using
    a strict lookup table. No arithmetic guessing.
    """

    def __init__(self, custom_table: Optional[Dict] = None):
        """
        Args:
            custom_table: Optional override mapping. Same format as RAMADAN_LOOKUP.
        """
        self.lookup = dict(RAMADAN_LOOKUP)
        if custom_table:
            self.lookup.update(custom_table)

        # Build fuzzy index for nearest-match fallback
        self._build_fuzzy_index()

    def _build_fuzzy_index(self):
        """Pre-compute time-to-minutes mapping for fuzzy matching."""
        self._index = []
        for (s, e), (rs, re_) in self.lookup.items():
            s_min = self._to_minutes(s)
            e_min = self._to_minutes(e)
            self._index.append((s_min, e_min, s, e, rs, re_))

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
        """Convert a single class entry."""
        original_start = cls.get("start_time", "")
        original_end = cls.get("end_time", "")

        # Normalize
        original_start = self._normalize(original_start)
        original_end = self._normalize(original_end)

        # 1. Exact lookup
        key = (original_start, original_end)
        if key in self.lookup:
            ram_start, ram_end = self.lookup[key]
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
        fuzzy = self._fuzzy_match(original_start, original_end, tolerance=20)
        if fuzzy:
            ram_start, ram_end, matched_key = fuzzy
            logger.info(
                "fuzzy_match",
                original=f"{original_start}-{original_end}",
                matched=f"{matched_key[0]}-{matched_key[1]}",
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
        self, start: str, end: str, tolerance: int = 20
    ) -> Optional[Tuple[str, str, Tuple[str, str]]]:
        """
        Find the closest matching entry in the lookup table
        within a tolerance (minutes).
        """
        s_min = self._to_minutes(start)
        e_min = self._to_minutes(end)

        if s_min is None or e_min is None:
            return None

        best = None
        best_dist = float('inf')

        for idx_s, idx_e, key_s, key_e, ram_s, ram_e in self._index:
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

    # ------------------------------------------------------------------
    # Admin: update table
    # ------------------------------------------------------------------
    def add_mapping(
        self,
        original_start: str,
        original_end: str,
        ramadan_start: str,
        ramadan_end: str,
    ):
        """Add or override a mapping entry."""
        key = (self._normalize(original_start), self._normalize(original_end))
        val = (self._normalize(ramadan_start), self._normalize(ramadan_end))
        self.lookup[key] = val
        self._build_fuzzy_index()
        logger.info("mapping_added", key=key, val=val)

    def get_table(self) -> Dict:
        """Return the full lookup table as serializable dict."""
        return {
            f"{k[0]}-{k[1]}": {"ramadan_start": v[0], "ramadan_end": v[1]}
            for k, v in sorted(self.lookup.items())
        }
