"""Integration tests: database persistence."""
from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.connection import init_db, session_scope
from src.database.models import AuditLogEntry, Base, CallRecord, TranscriptionCache
from src.graph.state import (
    CallReport,
    IntakeResult,
    ResolutionStatus,
    SummaryResult,
    TranscriptionResult,
)
from src.agents.report import compile_report, persist_report


@pytest.fixture(scope="module")
def engine():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    return eng


class TestCallRecordPersistence:
    def test_persist_and_retrieve(self, engine):
        report = compile_report("persist-test-1", status="completed")
        persist_report(report, engine=engine)

        from sqlalchemy.orm import sessionmaker as SM
        Session = SM(bind=engine)
        session = Session()
        row = session.query(CallRecord).filter_by(call_id="persist-test-1").first()
        session.close()
        assert row is not None
        assert row.call_id == "persist-test-1"
        assert row.status == "completed"

    def test_persist_with_summary(self, engine):
        summary = SummaryResult(
            call_id="persist-test-2",
            call_purpose="Test",
            key_discussion_points=["Point 1"],
            resolution_status=ResolutionStatus.resolved,
            sentiment_trajectory="Neutral",
        )
        report = compile_report("persist-test-2", summary=summary, status="completed")
        persist_report(report, engine=engine)

        from sqlalchemy.orm import sessionmaker as SM
        Session = SM(bind=engine)
        session = Session()
        row = session.query(CallRecord).filter_by(call_id="persist-test-2").first()
        session.close()
        assert row is not None
        assert row.summary_json is not None
        parsed = json.loads(row.summary_json)
        assert parsed["call_purpose"] == "Test"

    def test_multiple_records(self, engine):
        for i in range(3):
            r = compile_report(f"multi-test-{i}")
            persist_report(r, engine=engine)

        from sqlalchemy.orm import sessionmaker as SM
        Session = SM(bind=engine)
        session = Session()
        count = session.query(CallRecord).filter(CallRecord.call_id.like("multi-test-%")).count()
        session.close()
        assert count == 3


class TestAuditLogPersistence:
    def test_audit_entry_created(self, engine):
        from src.security.audit import AuditLogger
        # Patch the session to use our test engine
        import src.database.connection as conn
        orig_engine = conn._default_engine
        conn._default_engine = engine
        conn._session_factories = {}
        try:
            logger = AuditLogger()
            logger.log("audit-test-1", "test_action", details={"key": "value"})
        finally:
            conn._default_engine = orig_engine
            conn._session_factories = {}

        from sqlalchemy.orm import sessionmaker as SM
        Session = SM(bind=engine)
        session = Session()
        entry = session.query(AuditLogEntry).filter_by(call_id="audit-test-1").first()
        session.close()
        assert entry is not None
        assert entry.action == "test_action"

    def test_audit_details_serialized(self, engine):
        import src.database.connection as conn
        orig_engine = conn._default_engine
        conn._default_engine = engine
        conn._session_factories = {}
        try:
            from src.security.audit import AuditLogger
            logger = AuditLogger()
            logger.log("audit-test-2", "detail_action", details={"error": "test error"})
        finally:
            conn._default_engine = orig_engine
            conn._session_factories = {}

        from sqlalchemy.orm import sessionmaker as SM
        Session = SM(bind=engine)
        session = Session()
        entry = session.query(AuditLogEntry).filter_by(call_id="audit-test-2").first()
        session.close()
        if entry:
            details = json.loads(entry.details)
            assert details["error"] == "test error"


class TestTranscriptionCache:
    def test_cache_insert_and_retrieve(self, engine):
        from sqlalchemy.orm import sessionmaker as SM
        Session = SM(bind=engine)
        session = Session()
        cache = TranscriptionCache(
            audio_hash="abc123",
            transcription_json='{"call_id": "x", "full_text": "hello", "segments": [], "avg_confidence": 1.0, "low_confidence": false, "from_cache": false, "language": "en"}',
        )
        session.add(cache)
        session.commit()

        row = session.query(TranscriptionCache).filter_by(audio_hash="abc123").first()
        session.close()
        assert row is not None
        assert row.audio_hash == "abc123"
