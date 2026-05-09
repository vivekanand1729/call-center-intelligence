"""SQLAlchemy connection management with session pooling."""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base

_session_factories: dict[int, sessionmaker] = {}
_default_engine: Engine | None = None


def get_engine(db_path: str | None = None) -> Engine:
    global _default_engine
    if db_path is None:
        from src.utils.config import load_config
        db_path = load_config().db_path

    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})

    # Encryption key via PRAGMA (if sqlcipher is available)
    db_key = os.getenv("DB_ENCRYPTION_KEY", "")
    if db_key:
        @event.listens_for(engine, "connect")
        def set_pragma(dbapi_conn, conn_record):
            dbapi_conn.execute(f"PRAGMA key='{db_key}'")

    if _default_engine is None:
        _default_engine = engine
    return engine


def init_db(engine: Engine | None = None) -> None:
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)


def get_session(engine: Engine | None = None) -> Session:
    if engine is None:
        engine = _default_engine if _default_engine is not None else get_engine()
    eid = id(engine)
    if eid not in _session_factories:
        _session_factories[eid] = sessionmaker(bind=engine)
    return _session_factories[eid]()


@contextmanager
def session_scope(engine: Engine | None = None) -> Generator[Session, None, None]:
    session = get_session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
