"""Request history storage module."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class HistoryStorage:
    """Simple file-based storage for request history."""

    def __init__(self, storage_dir: str = ".history"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def _get_file_path(self, service: str) -> Path:
        """Get file path for a service's history."""
        return self.storage_dir / f"{service}.jsonl"

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
        file_path = self._get_file_path(service)

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "status": status,
            "request": request_data,
            "response": response_data,
        }

        with open(file_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

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
        file_path = self._get_file_path(service)

        if not file_path.exists():
            return []

        entries = []
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        # Reverse to get most recent first
        entries.reverse()

        # Apply pagination
        return entries[offset : offset + limit]

    def get_request(self, service: str, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific request by ID.

        Args:
            service: Service name
            request_id: Request ID to find

        Returns:
            Request entry or None if not found
        """
        file_path = self._get_file_path(service)

        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry["request_id"] == request_id:
                        return entry

        return None

    def clear_history(self, service: str) -> None:
        """
        Clear all history for a service.

        Args:
            service: Service name
        """
        file_path = self._get_file_path(service)
        if file_path.exists():
            file_path.unlink()


# Global instance
history_storage = HistoryStorage()
