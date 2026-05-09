# Call Center Intelligence

An AI-powered multi-agent pipeline for analyzing and scoring customer service call recordings. Built with LangGraph, it transcribes audio, detects security threats, redacts PII, summarizes conversations, and scores call quality — all through an interactive Streamlit UI.

## Features

- **Audio Transcription** — Local faster-whisper (GPU-accelerated) with SHA-256 caching to avoid re-processing
- **Security Scanning** — 22+ prompt injection patterns detected; pipeline halts on suspicious content
- **PII Redaction** — Automatically redacts SSNs, credit cards, emails, and phone numbers before LLM processing
- **Call Summarization** — Structured LLM output: purpose, discussion points, action items, sentiment trajectory
- **QA Scoring** — 5-dimension quality scoring (professionalism, empathy, problem resolution, compliance, communication clarity)
- **Compliance Escalation** — Critical violations automatically flagged for supervisor review
- **Report Generation** — PDF and JSON exports; all results persisted to SQLite
- **Multi-Provider LLM** — Supports OpenAI, Google Gemini, and Groq
- **Observability** — Optional LangSmith tracing integration

## Architecture

The pipeline is implemented as a LangGraph `StateGraph` with 7 sequential stages and conditional routing:

```
intake → transcription → injection_detection → pii_redaction → summarization+qa → report
                                                                                      ↓
                                                                           (critical flag?) → supervisor_review
```

Each stage is an independent agent node. Failures route to a dedicated error handler with exponential-backoff retries (configurable, default 3).

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph 1.1+, LangChain 1.2+ |
| LLM Providers | OpenAI (gpt-4o), Google Gemini (gemini-2.0-flash), Groq (llama-3.3-70b) |
| Transcription | faster-whisper 1.1+ |
| Web UI | Streamlit 1.40+ |
| Database | SQLite via SQLAlchemy 2.0+ |
| PDF Reports | ReportLab 4.0+ |
| Validation | Pydantic 2.13+ |

## Prerequisites

- Python 3.11+
- FFmpeg (`brew install ffmpeg` on macOS)
- API key for at least one LLM provider

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd call-center-intelligence

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
make install

# Configure environment
cp .env.example .env
# Edit .env — add your API key(s) at minimum
```

## Configuration

### Local development

Copy `.env.example` to `.env` and set the following:

```bash
# Choose your LLM provider
LLM_PROVIDER=openai          # openai | gemini | groq

# API keys (only the one matching LLM_PROVIDER is required)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...

# Whisper model size (tiny = fastest, large-v3 = most accurate)
WHISPER_MODEL_SIZE=tiny

# Pipeline settings
MAX_RETRIES_PER_NODE=3
LLM_TIMEOUT_SECONDS=60

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
```

### Streamlit Cloud deployment

`OPENAI_API_KEY` and `LANGCHAIN_API_KEY` are read from **Streamlit Secrets** in the live deployment — do not hardcode them. Add them in the Streamlit Cloud dashboard under **App settings → Secrets**:

```toml
# .streamlit/secrets.toml (for local reference only — never commit this file)
OPENAI_API_KEY = "sk-..."
LANGCHAIN_API_KEY = "..."
```

## Running

```bash
# Via Makefile
make run

# Directly
PYTHONPATH=. streamlit run src/ui/streamlit_app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Docker

```bash
docker build -t call-center-intelligence .

docker run -p 8501:8501 \
  -e OPENAI_API_KEY=sk-... \
  -e WHISPER_MODEL_SIZE=tiny \
  -v $(pwd)/data:/app/data \
  call-center-intelligence
```

## UI Tabs

| Tab | Purpose |
|-----|---------|
| **Analyze** | Upload an audio file and run the full pipeline; view transcript, summary, and QA scorecard |
| **History** | Search and browse past call analyses |
| **Observability** | LangSmith trace viewer and debug information |

## QA Scoring Dimensions

| Dimension | Weight |
|-----------|--------|
| Problem Resolution | 30% |
| Empathy | 20% |
| Compliance | 20% |
| Professionalism | 15% |
| Communication Clarity | 15% |

Each dimension is scored 1–5. Compliance flags with `severity=critical` trigger automatic escalation.

## Project Structure

```
call-center-intelligence/
├── app.py                    # Entry point: init DB, load Whisper, launch Streamlit
├── src/
│   ├── ui/streamlit_app.py   # Web interface (3 tabs)
│   ├── graph/
│   │   ├── workflow.py       # LangGraph StateGraph definition
│   │   ├── state.py          # PipelineState + 14 Pydantic models
│   │   └── edges.py          # Conditional routing functions
│   ├── agents/               # Pipeline stage implementations
│   │   ├── intake.py
│   │   ├── transcription.py
│   │   ├── summarization.py
│   │   ├── qa_scoring.py
│   │   └── report.py
│   ├── security/
│   │   ├── injection_detector.py
│   │   ├── pii_redactor.py
│   │   └── audit.py
│   ├── database/             # SQLAlchemy models and connection management
│   ├── services/             # Pipeline runner, LangSmith integration
│   └── utils/                # Config, LLM factory, audio utilities
├── tests/
│   ├── unit/                 # Per-component unit tests
│   ├── security/             # Injection and PII detection tests
│   └── integration/          # End-to-end and database tests
├── data/
│   ├── calls.db              # SQLite database (created at runtime)
│   └── samples/              # Sample call recordings
├── Dockerfile
├── Makefile
└── .env.example
```

## Testing

```bash
make test               # Unit + security tests
make test-unit          # Unit tests only
make test-security      # Security tests only (22+ injection payloads)
make test-integration   # End-to-end pipeline + database tests
make test-all           # All tests with coverage report
```

## Development

```bash
make lint               # Run ruff linter
make format             # Auto-format with ruff
make clean              # Remove build artifacts and temp files
```

## Supported Audio Formats

WAV, MP3, FLAC, M4A — maximum duration 60 minutes.

## Database Schema

| Table | Purpose |
|-------|---------|
| `call_records` | One row per analyzed call; stores transcript, summary, QA scores, and report as JSON |
| `audit_log_entries` | Per-call activity log for every pipeline action |
| `transcription_cache` | SHA-256 hash → cached transcription (avoids reprocessing) |