"""Append-only audit logger backed by the AuditLogEntry table."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional


class AuditLogger:
    def __init__(self, default_user: str = "app"):
        self._default_user = default_user

    def log(
        self,
        call_id: str,
        action: str,
        user: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        try:
            from src.database.connection import session_scope
            from src.database.models import AuditLogEntry
            with session_scope() as session:
                entry = AuditLogEntry(
                    call_id=call_id,
                    action=action,
                    user=user or self._default_user,
                    timestamp=datetime.now(timezone.utc),
                    details=json.dumps(details) if details else None,
                )
                session.add(entry)
        except Exception:
            # Audit logging must not crash the pipeline
            pass
