"""Quick test: Gemini extraction on uploaded timetable."""
from dotenv import load_dotenv
load_dotenv()

from ocr.ai_vision import AIVisionExtractor
import json, glob, os

ext = AIVisionExtractor()
print("Available:", ext.is_available())

# Find latest upload
files = sorted(glob.glob("uploads/*"), key=os.path.getmtime, reverse=True)
if not files:
    print("No uploads found")
    exit()

path = files[0]
print(f"Testing: {path}")

result = ext.extract_from_file(path)
if result:
    print(f"Method: {result.get('extraction_method')}")
    print(f"Total classes: {result.get('total_classes')}")
    print()
    for c in result.get("classes", []):
        day = c["day"]
        course = c["course"]
        t = f"{c['start_time']}-{c['end_time']}"
        dur = c["duration_minutes"]
        room = c.get("room", "")
        print(f"  {day:12s} {course:45s} {t:13s} {dur:3d}min  {room}")
else:
    print("FAILED - returned None")
