"""Shared pytest fixtures."""
from __future__ import annotations

import io
import struct
import wave
from typing import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.database.connection import init_db, session_scope
from src.database.models import Base


def make_wav_bytes(duration_seconds: float = 5.0, sample_rate: int = 16000, channels: int = 1) -> bytes:
    num_frames = int(sample_rate * duration_seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * num_frames * channels)
    return buf.getvalue()


@pytest.fixture(scope="session")
def wav_bytes_5s():
    return make_wav_bytes(5.0)


@pytest.fixture(scope="session")
def wav_bytes_long():
    """WAV longer than 60 minutes."""
    return make_wav_bytes(3601.0)


@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db_session(db_engine) -> Generator[Session, None, None]:
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()
