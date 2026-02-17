# &#9770; Ramadan Timetable Converter

A production-grade web application that converts university timetable images/PDFs into Ramadan-adjusted schedules with Google Calendar sync.

## Features

- **Smart Upload**: Drag & drop timetable photos, screenshots, or PDFs
- **AI-Powered OCR**: Handles blurry photos, skewed angles, glare, colored blocks
- **Ramadan Conversion**: Official lookup-based time mapping (no arithmetic guessing)
- **Interactive Editing**: Review and edit extracted schedule before exporting
- **Export Options**: JSON, CSV, Google Calendar sync
- **Confidence Scoring**: Every extracted entry rated for accuracy
- **Conflict Detection**: Overlapping class detection

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone and configure
cp .env.example .env
# Edit .env with your settings

# Build and run
docker-compose up -d

# App is live at http://localhost:8000
```

### Option 2: Local Python

**Prerequisites:**
- Python 3.10+
- Tesseract OCR installed ([download](https://github.com/tesseract-ocr/tesseract))
- Poppler (for PDF support) — [Windows](https://github.com/oschwartz10612/poppler-windows/releases)

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env — set TESSERACT_CMD if not on PATH

# Run
python app.py
```

App runs at **http://localhost:8000**

---

## Configuration (.env)

| Variable | Description | Default |
|---|---|---|
| `APP_HOST` | Server host | `0.0.0.0` |
| `APP_PORT` | Server port | `8000` |
| `DEBUG` | Enable auto-reload | `true` |
| `TESSERACT_CMD` | Path to tesseract binary | `tesseract` |
| `DEFAULT_TIMEZONE` | Calendar timezone | `Asia/Dubai` |
| `MAX_UPLOAD_SIZE_MB` | Max file upload size | `20` |
| `GOOGLE_CLIENT_ID` | OAuth client ID | — |
| `GOOGLE_CLIENT_SECRET` | OAuth client secret | — |
| `GOOGLE_REDIRECT_URI` | OAuth callback URL | `http://localhost:8000/api/calendar/callback` |

---

## Google Calendar Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable **Google Calendar API**
4. Create **OAuth 2.0 Credentials** (Web application type)
5. Add redirect URI: `http://localhost:8000/api/calendar/callback`
6. Copy Client ID and Secret to `.env`

---

## Architecture

```
ramadan_timing/
├── app.py                          # FastAPI main application
├── vision/
│   └── preprocessor.py             # Image preprocessing (OpenCV)
├── ocr/
│   └── extractor.py                # OCR + schedule extraction (Tesseract)
├── conversion/
│   └── ramadan_converter.py        # Lookup-based Ramadan conversion
├── calendar_sync/
│   └── google_cal.py               # Google Calendar OAuth + sync
├── templates/
│   └── index.html                  # Frontend (single-page app)
├── static/                         # Static assets
├── uploads/                        # Uploaded files (auto-created)
├── Dockerfile                      # Container image
├── docker-compose.yml              # Container orchestration
├── requirements.txt                # Python dependencies
└── .env.example                    # Configuration template
```

### Pipeline Flow

```
Upload → Preprocess (OpenCV) → OCR (Tesseract) → Schedule Extraction
                                                        ↓
                                               Ramadan Conversion (Lookup)
                                                        ↓
                                              Display / Edit / Export
                                                        ↓
                                              Google Calendar Sync
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/api/upload` | Upload timetable image/PDF |
| `POST` | `/api/update-schedule` | Update extracted schedule |
| `GET` | `/api/export/json?session_id=X` | Download JSON |
| `GET` | `/api/export/csv?session_id=X` | Download CSV |
| `GET` | `/api/calendar/auth?session_id=X` | Start OAuth flow |
| `GET` | `/api/calendar/callback` | OAuth callback |
| `POST` | `/api/calendar/sync` | Create calendar events |
| `GET` | `/api/health` | Health check |

---

## Cloud Deployment

### Railway / Render / Fly.io

All support Docker-based deployment:

```bash
# Railway
railway up

# Render
# Connect GitHub repo → Auto-deploy from Dockerfile

# Fly.io
fly launch
fly deploy
```

### AWS / GCP / Azure

```bash
# Build image
docker build -t ramadan-timetable .

# Tag for registry
docker tag ramadan-timetable:latest YOUR_REGISTRY/ramadan-timetable:latest

# Push
docker push YOUR_REGISTRY/ramadan-timetable:latest

# Deploy to ECS / Cloud Run / Azure Container Instances
```

---

## Ramadan Timing Table

The conversion uses an **exact lookup table** — no arithmetic shifting.

Example mappings:

| Original | Ramadan | Duration Type |
|---|---|---|
| 08:00 - 09:00 | 09:00 - 09:45 | 1 hour |
| 09:00 - 10:30 | 09:50 - 11:00 | 90 min |
| 10:00 - 12:00 | 10:40 - 12:15 | 2 hour |
| 14:00 - 17:00 | 14:00 - 16:25 | 3 hour lab |

Full table: see `conversion/ramadan_converter.py`

To add custom mappings:
```python
converter = RamadanConverter()
converter.add_mapping("07:30", "08:30", "08:30", "09:15")
```

---

## Maintenance

- **Logs**: JSON-formatted via structlog, Docker captures via json-file driver
- **Health**: `/api/health` endpoint for monitoring
- **Scaling**: Increase `--workers` in Dockerfile CMD or scale Docker replicas
- **Updates**: `docker-compose pull && docker-compose up -d`

---

## Troubleshooting

| Issue | Solution |
|---|---|
| OCR returns garbage | Ensure Tesseract is installed and `TESSERACT_CMD` is correct |
| PDF upload fails | Install Poppler: `apt install poppler-utils` or download Windows binaries |
| Low confidence scores | Upload clearer images; use flash, avoid shadows |
| Calendar sync fails | Verify Google API credentials in `.env` |
| Docker build fails | Ensure Docker Desktop is running |

---

## License

MIT
