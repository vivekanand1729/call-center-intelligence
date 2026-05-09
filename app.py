"""App entrypoint: initialize all components and launch the Streamlit UI."""
from __future__ import annotations

import os
import sys

# Ensure src/ is on the path
sys.path.insert(0, os.path.dirname(__file__))

from src.utils.config import load_config
from src.database.connection import get_engine, init_db
from src.agents.transcription import _get_whisper_model
from src.graph.workflow import compile_workflow
from src.security.audit import AuditLogger


def main():
    cfg = load_config()

    # Configure LangSmith tracing
    if cfg.langchain_tracing_v2 and cfg.langchain_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = cfg.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = cfg.langchain_project

    # Initialize database
    engine = get_engine(cfg.db_path)
    init_db(engine)

    # Pre-load Whisper model singleton
    print(f"Loading Whisper model: {cfg.whisper_model_size}")
    _get_whisper_model(cfg.whisper_model_size)

    # Compile pipeline
    workflow = compile_workflow(config=cfg, db_engine=engine)

    # Log startup
    audit = AuditLogger()

    print("Starting Call Center Intelligence System on http://localhost:8501")
    import streamlit.web.cli as stcli
    sys.argv = [
        "streamlit",
        "run",
        os.path.join(os.path.dirname(__file__), "src", "ui", "streamlit_app.py"),
        "--server.port=8501",
        "--server.address=0.0.0.0",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
