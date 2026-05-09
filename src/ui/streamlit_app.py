"""Streamlit web interface: Analyze Call, History, Observability tabs."""
from __future__ import annotations

import io
import os
import wave

import streamlit as st

st.set_page_config(
    page_title="Call Center Intelligence System",
    page_icon="📞",
    layout="wide",
)


def _get_workflow():
    if "workflow" not in st.session_state:
        from src.graph.workflow import compile_workflow
        st.session_state["workflow"] = compile_workflow()
    return st.session_state["workflow"]


def _audio_to_wav_bytes(uploaded_file) -> tuple[bytes, str]:
    """Convert uploaded file to bytes and detect filename."""
    data = uploaded_file.read()
    return data, uploaded_file.name


# ─── Tab 1: Analyze Call ──────────────────────────────────────────────────────

def render_analyze_tab():
    st.header("Analyze Call")
    st.markdown("Upload a call recording (WAV, MP3, FLAC, M4A) to analyze it.")

    col1, col2 = st.columns([2, 1])
    with col1:
        audio_file = st.file_uploader(
            "Upload Audio File",
            type=["wav", "mp3", "flac", "m4a"],
            help="Max 50 MB, max 60 minutes",
        )
    with col2:
        caller_id = st.text_input("Caller ID (optional)")
        department = st.text_input("Department (optional)")

    analyze_btn = st.button("Analyze Call", type="primary", use_container_width=True)

    if analyze_btn:
        if audio_file is None:
            st.error("Please upload an audio file.")
            return

        status_placeholder = st.empty()
        status_placeholder.info(
            "Processing your call... This may take 1-3 minutes. Do NOT refresh the page."
        )

        try:
            from src.services.pipeline import process_call
            audio_bytes, filename = _audio_to_wav_bytes(audio_file)
            workflow = _get_workflow()
            result = process_call(
                workflow,
                audio_data=audio_bytes,
                filename=filename,
                caller_id=caller_id or None,
                department=department or None,
            )
            status_placeholder.empty()

            if result.status in ("failed",) or result.error:
                st.error(f"Analysis failed: {result.error}")
                return

            if result.status == "flagged_for_review":
                st.warning("This call has been **flagged for supervisor review** due to a critical compliance issue.")

            st.success(f"Analysis complete! (Call ID: `{result.call_id}`)")

            st.subheader("Transcript")
            st.text_area("Speaker-labeled transcript", value=result.transcript, height=250)

            col_sum, col_qa = st.columns(2)
            with col_sum:
                st.subheader("Summary")
                st.markdown(result.summary_md)
            with col_qa:
                st.subheader("QA Scorecard")
                st.markdown(result.qa_md)

            st.subheader("Download Reports")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                if result.pdf_bytes:
                    st.download_button(
                        "Download PDF Report",
                        data=result.pdf_bytes,
                        file_name=f"report_{result.call_id[:8]}.pdf",
                        mime="application/pdf",
                    )
                else:
                    st.info("PDF report not available (install reportlab for PDF support)")
            with dl_col2:
                if result.json_str:
                    st.download_button(
                        "Download JSON Report",
                        data=result.json_str,
                        file_name=f"report_{result.call_id[:8]}.json",
                        mime="application/json",
                    )
        except Exception as e:
            status_placeholder.empty()
            st.error(f"Unexpected error: {e}")


# ─── Tab 2: All Call History ─────────────────────────────────────────────────

def render_history_tab():
    st.header("All Call History")

    try:
        import json as _json
        from src.database.connection import session_scope
        from src.database.models import CallRecord
        from sqlalchemy import desc

        with session_scope() as session:
            records = (
                session.query(CallRecord)
                .order_by(desc(CallRecord.processed_at))
                .limit(100)
                .all()
            )

        if not records:
            st.info("No calls analyzed yet. Use the **Analyze Call** tab to get started.")
            return

        # Summary table
        table_data = []
        for r in records:
            qa_score = "-"
            if r.qa_scores_json:
                try:
                    d = _json.loads(r.qa_scores_json)
                    qa_score = f"{d.get('overall_score', 0):.1f}"
                except Exception:
                    pass
            table_data.append({
                "Call ID": r.call_id[:12] + "…",
                "Status": r.status,
                "Filename": r.audio_filename or "-",
                "QA Score": qa_score,
                "Processed At": r.processed_at.strftime("%Y-%m-%d %H:%M") if r.processed_at else "-",
            })

        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)

        # Detail view
        st.subheader("Call Detail")
        call_ids = [r.call_id for r in records]
        selected = st.selectbox("Select Call ID for details", call_ids, format_func=lambda x: x[:20] + "…")
        if selected:
            record = next((r for r in records if r.call_id == selected), None)
            if record:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Status:** {record.status}")
                    st.markdown(f"**Processed:** {record.processed_at}")
                    if record.transcript_text:
                        st.text_area("Transcript", value=record.transcript_text, height=200)
                with col2:
                    if record.summary_json:
                        from src.graph.state import SummaryResult
                        from src.utils.formatters import format_summary
                        try:
                            s = SummaryResult.model_validate_json(record.summary_json)
                            st.markdown(format_summary(s))
                        except Exception:
                            st.json(_json.loads(record.summary_json))
    except Exception as e:
        st.error(f"Error loading history: {e}")


# ─── Tab 3: Observability ─────────────────────────────────────────────────────

def render_observability_tab():
    st.header("Observability")

    if st.button("Refresh Metrics", use_container_width=False):
        st.cache_data.clear()

    try:
        from src.services.observability import get_observability_dashboard
        metrics_md, langsmith_md, audit_rows = get_observability_dashboard()

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(metrics_md)
        with col2:
            st.markdown(langsmith_md)

        st.subheader("Recent Audit Events")
        if audit_rows:
            import pandas as pd
            df = pd.DataFrame(audit_rows, columns=["Timestamp", "Call ID", "Action", "Details"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No audit events recorded yet.")
    except Exception as e:
        st.error(f"Error loading observability data: {e}")


# ─── Main app ─────────────────────────────────────────────────────────────────

def build_app():
    st.title("Call Center Intelligence System")
    st.markdown("*AI-Powered Multi-Agent Analysis Pipeline*")

    tab1, tab2, tab3 = st.tabs(["Analyze Call", "All Call History", "Observability"])
    with tab1:
        render_analyze_tab()
    with tab2:
        render_history_tab()
    with tab3:
        render_observability_tab()


if __name__ == "__main__":
    build_app()
