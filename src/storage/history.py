"""Request history storage module using SQLite."""

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..db.db_config import get_db


class HistoryStorage:
    """SQLite-based storage for request history."""

    def __init__(self):
        """Initialize history storage using shared haid.db."""
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with get_db() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_data TEXT NOT NULL,
                    response_data TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for better query performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_service_timestamp
                ON request_history(service, timestamp DESC)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_request_id
                ON request_history(request_id)
            """)

    def add_request(
        self,
        service: str,
        request_id: str,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
        status: str = "success",
    ) -> None:
        """
        Add a request to history.

        Args:
            service: Service name (crawl, embed, caption)
            request_id: Unique request ID
            request_data: Original request data
            response_data: Response data
            status: Request status (success/error)
        """
        timestamp = datetime.utcnow().isoformat()

        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO request_history
                (service, request_id, timestamp, status, request_data, response_data)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    service,
                    request_id,
                    timestamp,
                    status,
                    json.dumps(request_data),
                    json.dumps(response_data),
                ),
            )

    def get_history(
        self, service: str, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get request history for a service.

        Args:
            service: Service name
            limit: Maximum number of entries to return
            offset: Number of entries to skip

        Returns:
            List of history entries (most recent first)
        """
        with get_db() as conn:
            cursor = conn.execute(
                """
                SELECT service, request_id, timestamp, status, request_data, response_data
                FROM request_history
                WHERE service = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                (service, limit, offset),
            )

            rows = cursor.fetchall()

            entries = []
            for row in rows:
                entries.append(
                    {
                        "timestamp": row["timestamp"],
                        "request_id": row["request_id"],
                        "status": row["status"],
                        "request": json.loads(row["request_data"]),
                        "response": json.loads(row["response_data"]),
                    }
                )

            return entries

    def get_request(self, service: str, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific request by ID.

        Args:
            service: Service name
            request_id: Request ID to find

        Returns:
            Request entry or None if not found
        """
        with get_db() as conn:
            cursor = conn.execute(
                """
                SELECT service, request_id, timestamp, status, request_data, response_data
                FROM request_history
                WHERE service = ? AND request_id = ?
                LIMIT 1
                """,
                (service, request_id),
            )

            row = cursor.fetchone()

            if row is None:
                return None

            return {
                "timestamp": row["timestamp"],
                "request_id": row["request_id"],
                "status": row["status"],
                "request": json.loads(row["request_data"]),
                "response": json.loads(row["response_data"]),
            }

    def clear_history(self, service: str) -> None:
        """
        Clear all history for a service.

        Args:
            service: Service name
        """
        with get_db() as conn:
            conn.execute(
                """
                DELETE FROM request_history
                WHERE service = ?
                """,
                (service,),
            )

    def get_stats(self) -> Dict[str, int]:
        """
        Get overall task statistics.

        Returns:
            Dictionary with running, today, and total counts
        """
        # Get today's date in ISO format (start of day)
        today = datetime.utcnow().date().isoformat()

        with get_db() as conn:
            # Total count
            cursor = conn.execute("SELECT COUNT(*) FROM request_history")
            total = cursor.fetchone()[0]

            # Today's count (using created_at timestamp)
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM request_history
                WHERE DATE(created_at) = ?
                """,
                (today,),
            )
            today_count = cursor.fetchone()[0]

            # For now, running is 0 since we don't track active requests
            # This could be enhanced in the future with a separate tracking mechanism
            return {
                "running": 0,
                "today": today_count,
                "total": total,
            }


# Global instance
history_storage = HistoryStorage()
