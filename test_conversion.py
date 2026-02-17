"""Quick test of the Ramadan conversion engine."""
from conversion.ramadan_converter import RamadanConverter

c = RamadanConverter()

test = {"classes": [
    {"day": "Monday", "course": "Math 101", "start_time": "08:00", "end_time": "09:00", "duration_minutes": 60, "confidence": 0.9},
    {"day": "Tuesday", "course": "Physics Lab", "start_time": "14:00", "end_time": "17:00", "duration_minutes": 180, "confidence": 0.85},
    {"day": "Wednesday", "course": "English", "start_time": "10:00", "end_time": "11:30", "duration_minutes": 90, "confidence": 0.75},
    {"day": "Thursday", "course": "CS 201", "start_time": "09:05", "end_time": "10:05", "duration_minutes": 60, "confidence": 0.6},
]}

result = c.convert(test)

for cls in result["classes"]:
    day = cls["day"]
    course = cls["course"]
    orig = cls["original_start"] + "-" + cls["original_end"]
    ram = cls["ramadan_start"] + "-" + cls["ramadan_end"]
    mt = cls["mapping_type"]
    print(f"  {day:12} {course:15} {orig:13} -> {ram:13}  [{mt}]")

print(f"\nStats: {result['conversion_stats']}")
