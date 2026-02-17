"""
Google Calendar Sync Module
Handles OAuth authentication and event creation for Ramadan schedules.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import structlog

logger = structlog.get_logger()

# Google API imports (optional — graceful degradation if not configured)
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    logger.warning("google_api_not_available", msg="Install google-api-python-client for calendar sync")


# OAuth scopes
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

# Day name → next weekday offset
DAY_OFFSETS = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
}

# In-memory credential store (production would use encrypted DB)
_credentials_store: Dict[str, Any] = {}


class GoogleCalendarSync:
    """
    Manages Google Calendar OAuth flow and event creation
    for Ramadan-adjusted timetables.
    """

    def __init__(self):
        self.client_id = os.getenv("GOOGLE_CLIENT_ID", "")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
        self.redirect_uri = os.getenv(
            "GOOGLE_REDIRECT_URI",
            "http://localhost:8000/api/calendar/callback"
        )
        self._configured = bool(
            self.client_id
            and self.client_secret
            and self.client_id != "your-client-id-here"
        )

    def is_configured(self) -> bool:
        """Check if Google Calendar API is properly configured."""
        return self._configured and GOOGLE_API_AVAILABLE

    def get_auth_url(self, session_id: str) -> str:
        """
        Generate Google OAuth authorization URL.
        Session ID is passed as state parameter for callback correlation.
        """
        if not self.is_configured():
            raise RuntimeError(
                "Google Calendar API not configured. "
                "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env"
            )

        client_config = {
            "web": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.redirect_uri],
            }
        }

        flow = Flow.from_client_config(client_config, scopes=SCOPES)
        flow.redirect_uri = self.redirect_uri

        auth_url, _ = flow.authorization_url(
            access_type="offline",
            include_granted_scopes="true",
            state=session_id,
            prompt="consent",
        )

        return auth_url

    def handle_callback(self, code: str, session_id: str):
        """
        Exchange OAuth authorization code for credentials.
        Store credentials for the session.
        """
        if not self.is_configured():
            raise RuntimeError("Google Calendar API not configured.")

        client_config = {
            "web": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [self.redirect_uri],
            }
        }

        flow = Flow.from_client_config(client_config, scopes=SCOPES)
        flow.redirect_uri = self.redirect_uri
        flow.fetch_token(code=code)

        _credentials_store[session_id] = flow.credentials
        logger.info("calendar_authenticated", session_id=session_id)

    def sync_events(
        self,
        session_id: str,
        schedule: Dict[str, Any],
        timezone: str = "Asia/Dubai",
        weeks: int = 4,
    ) -> int:
        """
        Create Google Calendar events for the Ramadan schedule.

        Args:
            session_id: Session with stored credentials
            schedule: Converted Ramadan schedule dict
            timezone: IANA timezone string
            weeks: Number of weeks to create repeating events

        Returns:
            Number of events created
        """
        if not self.is_configured():
            raise RuntimeError("Google Calendar API not configured.")

        creds = _credentials_store.get(session_id)
        if not creds:
            raise RuntimeError(
                "Not authenticated. Please authorize via /api/calendar/auth first."
            )

        service = build("calendar", "v3", credentials=creds)
        events_created = 0

        classes = schedule.get("classes", [])

        # Find the next occurrence of each day
        today = datetime.now()
        current_weekday = today.weekday()  # 0=Monday

        for cls in classes:
            day_name = cls.get("day", "")
            course = cls.get("course", cls.get("ramadan_start", "Class"))
            ram_start = cls.get("ramadan_start", "")
            ram_end = cls.get("ramadan_end", "")

            if not day_name or not ram_start or not ram_end:
                continue

            day_offset = DAY_OFFSETS.get(day_name)
            if day_offset is None:
                continue

            # Calculate the next occurrence of this weekday
            days_ahead = day_offset - current_weekday
            if days_ahead < 0:
                days_ahead += 7

            next_date = today + timedelta(days=days_ahead)
            date_str = next_date.strftime("%Y-%m-%d")

            # Build event
            event_body = {
                "summary": f"[Ramadan] {course}",
                "description": (
                    f"Original: {cls.get('original_start', '?')} - "
                    f"{cls.get('original_end', '?')}\n"
                    f"Ramadan: {ram_start} - {ram_end}\n"
                    f"Auto-generated by Ramadan Timetable Converter"
                ),
                "start": {
                    "dateTime": f"{date_str}T{ram_start}:00",
                    "timeZone": timezone,
                },
                "end": {
                    "dateTime": f"{date_str}T{ram_end}:00",
                    "timeZone": timezone,
                },
                "recurrence": [
                    f"RRULE:FREQ=WEEKLY;COUNT={weeks}"
                ],
                "reminders": {
                    "useDefault": False,
                    "overrides": [
                        {"method": "popup", "minutes": 15},
                    ],
                },
                "colorId": "6",  # Banana (yellow-ish for Ramadan)
            }

            try:
                service.events().insert(
                    calendarId="primary",
                    body=event_body,
                ).execute()
                events_created += 1
                logger.info(
                    "calendar_event_created",
                    course=course,
                    day=day_name,
                    time=f"{ram_start}-{ram_end}",
                )
            except Exception as e:
                logger.error(
                    "calendar_event_error",
                    course=course,
                    error=str(e),
                )

        return events_created

    def get_status(self, session_id: str) -> Dict[str, Any]:
        """Check calendar connection status for a session."""
        return {
            "configured": self.is_configured(),
            "authenticated": session_id in _credentials_store,
            "google_api_available": GOOGLE_API_AVAILABLE,
        }
