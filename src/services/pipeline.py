"""Pipeline service: converts Gradio/Streamlit audio input to pipeline results."""
from __future__ import annotations

import io
import os
import tempfile
import wave
from dataclasses import dataclass, field
from typing import Any, Optional

from src.graph.state import AudioInput

# Module-level temp file list for rolling cleanup
_temp_files: list[str] = []
_MAX_TEMP_FILES = 50


def _cleanup_old_temps() -> None:
    global _temp_files
    while len(_temp_files) > _MAX_TEMP_FILES:
        old = _temp_files.pop(0)
        try:
            if os.path.exists(old):
                os.unlink(old)
        except OSError:
            pass


def _register_temp(path: str) -> None:
    _temp_files.append(path)
    _cleanup_old_temps()


@dataclass
class PipelineResult:
    status: str
    call_id: str = ""
    transcript: str = ""
    summary_md: str = ""
    qa_md: str = ""
    pdf_bytes: Optional[bytes] = None
    json_str: Optional[str] = None
    error: Optional[str] = None
    pdf_path: Optional[str] = None
    json_path: Optional[str] = None


def _numpy_to_wav_bytes(sample_rate: int, audio_array) -> bytes:
    """Convert numpy audio array (from Streamlit/Gradio) to WAV bytes."""
    import numpy as np
    if audio_array.dtype != np.int16:
        # Normalize float32 to int16
        audio_int16 = (audio_array * 32767).astype(np.int16)
    else:
        audio_int16 = audio_array

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        channels = 1 if audio_int16.ndim == 1 else audio_int16.shape[1]
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


def process_call(
    workflow: Any,
    audio_data: bytes,
    filename: str = "call.wav",
    caller_id: Optional[str] = None,
    department: Optional[str] = None,
) -> PipelineResult:
    from src.agents.report import generate_report_json, generate_report_pdf
    from src.utils.formatters import format_qa, format_summary, format_transcript

    audio_input = AudioInput(
        audio_data=audio_data,
        filename=filename,
        caller_id=caller_id or None,
        department=department or None,
    )

    try:
        result_state = workflow.invoke({"audio_input": audio_input})
    except Exception as e:
        return PipelineResult(status="failed", error=str(e))

    status = result_state.get("status", "failed")
    error = result_state.get("error")
    call_id = ""

    if result_state.get("intake"):
        call_id = result_state["intake"].call_id

    if status in ("failed",) or error:
        return PipelineResult(status=status, call_id=call_id, error=error)

    transcript_text = format_transcript(result_state.get("transcription"))
    summary_md = format_summary(result_state.get("summary"))
    qa_md = format_qa(result_state.get("qa_scores"))

    report = result_state.get("report")
    pdf_bytes = generate_report_pdf(report) if report else None
    json_str = generate_report_json(report) if report else None

    # Write to temp files for download
    pdf_path = json_path = None
    if pdf_bytes:
        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_pdf.write(pdf_bytes)
        tmp_pdf.close()
        pdf_path = tmp_pdf.name
        _register_temp(pdf_path)
    if json_str:
        tmp_json = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
        tmp_json.write(json_str)
        tmp_json.close()
        json_path = tmp_json.name
        _register_temp(json_path)

    return PipelineResult(
        status=status,
        call_id=call_id,
        transcript=transcript_text,
        summary_md=summary_md,
        qa_md=qa_md,
        pdf_bytes=pdf_bytes,
        json_str=json_str,
        pdf_path=pdf_path,
        json_path=json_path,
    )
