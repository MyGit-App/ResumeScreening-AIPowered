"""Streamlit interface for evidence-grounded AI resume screening."""

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

APP_DIR = Path(__file__).resolve().parent
PIPELINE_NOTEBOOK = APP_DIR / "ResumeScreening.ipynb"
SAMPLE_RESUME_DIR = APP_DIR / "sample_resumes"
SAMPLE_RESUME_PATHS = sorted(SAMPLE_RESUME_DIR.glob("*.pdf"))
SAMPLE_CANDIDATE_NAMES = [
    path.stem.removeprefix("resume_").replace("_", " ").title()
    for path in SAMPLE_RESUME_PATHS
]
SAMPLE_CANDIDATE_NAMES_BY_FILE = {
    path.name: candidate_name
    for path, candidate_name in zip(
        SAMPLE_RESUME_PATHS,
        SAMPLE_CANDIDATE_NAMES,
        strict=True,
    )
}
SAMPLE_JD_PATH = APP_DIR / "sample_jd.txt"
MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024


@st.cache_resource
def load_pipeline_notebook(
    notebook_path: str,
    notebook_mtime: float,
) -> dict[str, Any]:
    """Load trusted pipeline functions from the companion notebook code cells."""
    del notebook_mtime
    with Path(notebook_path).open(encoding="utf-8") as notebook_file:
        notebook_data = json.load(notebook_file)

    pipeline_namespace: dict[str, Any] = {"__name__": "ResumeScreening"}
    for cell_index, cell in enumerate(notebook_data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if source.strip():
            cell_name = f"{notebook_path}#cell-{cell_index}"
            exec(compile(source, cell_name, "exec"), pipeline_namespace)

    required_symbols = (
        "DEFAULT_OPENROUTER_MODEL",
        "build_screening_clients",
        "evaluate_resume",
    )
    missing_symbols = [
        symbol for symbol in required_symbols if symbol not in pipeline_namespace
    ]
    if missing_symbols:
        raise RuntimeError(
            "The screening notebook is missing required definitions: "
            + ", ".join(missing_symbols)
        )
    return pipeline_namespace


pipeline_module = load_pipeline_notebook(
    str(PIPELINE_NOTEBOOK),
    PIPELINE_NOTEBOOK.stat().st_mtime,
)
DEFAULT_OPENROUTER_MODEL = pipeline_module["DEFAULT_OPENROUTER_MODEL"]
build_screening_clients = pipeline_module["build_screening_clients"]
evaluate_resume = pipeline_module["evaluate_resume"]


load_dotenv(APP_DIR / ".env")

st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="📄",
    layout="wide",
)

SAMPLE_JD = SAMPLE_JD_PATH.read_text(encoding="utf-8")


def render_evidence_list(
    title: str,
    evidence: list[dict[str, str]],
    empty_message: str,
) -> None:
    st.markdown(f"**{title}**")
    if not evidence:
        st.write(empty_message)
        return
    for item in evidence:
        st.markdown(f"- **{item['skill']}:** {item['evidence']}")


def get_default_api_key() -> str:
    environment_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if environment_key:
        return environment_key
    try:
        return str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
    except StreamlitSecretNotFoundError:
        return ""


def get_openrouter_model() -> str:
    environment_model = os.getenv("OPENROUTER_LLM_MODEL", "").strip()
    if environment_model:
        return environment_model
    try:
        configured_model = str(st.secrets.get("OPENROUTER_LLM_MODEL", "")).strip()
    except StreamlitSecretNotFoundError:
        configured_model = ""
    return configured_model or DEFAULT_OPENROUTER_MODEL


def render_evaluation(filename: str, evaluation: dict[str, Any]) -> None:
    with st.container(border=True):
        st.subheader(evaluation["candidate_name"])
        st.caption(filename)
        metric_col, recommendation_col = st.columns(2)
        metric_col.metric("JD match score", f"{evaluation['match_score']}/100")
        recommendation_col.metric("Recommendation", evaluation["recommendation"])
        st.write(evaluation["summary"])
        st.caption(f"Why: {evaluation['recommendation_reason']}")
        strengths_col, gaps_col = st.columns(2)
        with strengths_col:
            render_evidence_list(
                "Strengths",
                evaluation["strengths"],
                "No strengths were identified in the retrieved excerpts.",
            )
        with gaps_col:
            render_evidence_list(
                "Weaknesses",
                evaluation["weaknesses"],
                "No job-related weaknesses were identified in the retrieved excerpts.",
            )
        render_evidence_list(
            "Missing required skills",
            evaluation["missing_skills"],
            "No required skills were identified as missing from the retrieved excerpts.",
        )


def render_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        sample_name = SAMPLE_CANDIDATE_NAMES_BY_FILE.get(result["filename"])
        if sample_name:
            result["evaluation"]["candidate_name"] = sample_name

    ranked = sorted(
        results,
        key=lambda result: result["evaluation"]["match_score"],
        reverse=True,
    )
    best = ranked[0]
    st.success(
        f"Top evidence-based match: **{best['evaluation']['candidate_name']}** "
        f"({best['evaluation']['match_score']}/100). "
        "Use this as decision support, not as an automated hiring decision."
    )

    overview_tab, candidate_tab, report_tab = st.tabs(
        ["Comparison", "Candidate evaluations", "Download report"]
    )
    with overview_tab:
        rows = [
            {
                "Rank": rank,
                "Candidate": result["evaluation"]["candidate_name"],
                "Resume": result["filename"],
                "Match score": result["evaluation"]["match_score"],
                "Recommendation": result["evaluation"]["recommendation"],
                "Summary": result["evaluation"]["summary"],
            }
            for rank, result in enumerate(ranked, start=1)
        ]
        st.dataframe(rows, hide_index=True, width="stretch")
    with candidate_tab:
        for result in ranked:
            render_evaluation(result["filename"], result["evaluation"])
    with report_tab:
        report = [
            {
                "filename": result["filename"],
                **result["evaluation"],
            }
            for result in ranked
        ]
        st.download_button(
            "Download evaluations as JSON",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="resume_screening_results.json",
            mime="application/json",
            width="content",
        )
        st.caption(
            "The report contains only retrieved resume evidence and the supplied "
            "job description's requirements."
        )


st.title("AI Resume Screening Assistant")
st.write(
    "Compare PDF resumes with a job description using LangChain RAG. "
    "Every candidate assessment is generated from retrieved resume excerpts."
)

with st.sidebar:
    st.header("Screening setup")
    api_key = st.text_input(
        "OpenRouter API key",
        value=get_default_api_key(),
        type="password",
        help="Read from OPENROUTER_API_KEY in .env or enter it for this session.",
    )
    include_samples = st.checkbox(
        f"Include sample resumes ({', '.join(SAMPLE_CANDIDATE_NAMES)})",
        help="Adds the listed sample PDFs to the uploaded resumes.",
    )
    st.caption(
        "Resume embeddings are created locally. Resume excerpts and the job "
        "description are sent to OpenRouter for Nemotron evaluation. "
        "Do not upload resumes unless you are authorized to process them."
    )

st.header("1. Add resumes")
uploaded_files = st.file_uploader(
    "Upload one or more PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
)

st.header("2. Add a job description")
if st.button("Load sample data-scientist job description", width="content"):
    st.session_state["job_description"] = SAMPLE_JD

with st.form("screening_form"):
    job_description = st.text_area(
        "Job description",
        key="job_description",
        height=240,
        placeholder="Paste the role requirements and responsibilities here.",
    )
    submitted = st.form_submit_button(
        "Evaluate resumes",
        type="primary",
        width="stretch",
    )

if submitted:
    st.session_state.pop("screening_results", None)
    st.session_state.pop("screening_failures", None)
    resume_inputs: list[tuple[str, bytes]] = [
        (uploaded_file.name, uploaded_file.getvalue())
        for uploaded_file in uploaded_files or []
    ]
    if include_samples:
        resume_inputs.extend(
            (sample_path.name, sample_path.read_bytes())
            for sample_path in SAMPLE_RESUME_PATHS
        )

    if not api_key.strip():
        st.error(
            "Enter an OpenRouter API key in the sidebar or set "
            "OPENROUTER_API_KEY in .env."
        )
    elif not job_description.strip():
        st.error("Enter a job description before evaluating resumes.")
    elif not resume_inputs:
        st.error("Upload at least one PDF or select the sample resumes.")
    else:
        oversized = [
            name
            for name, pdf_bytes in resume_inputs
            if len(pdf_bytes) > MAX_PDF_SIZE_BYTES
        ]
        if oversized:
            st.error(
                "Each PDF must be 10 MB or smaller. Oversized files: "
                + ", ".join(oversized)
            )
        else:
            llm, embeddings = build_screening_clients(
                api_key,
                model=get_openrouter_model(),
            )
            results: list[dict[str, Any]] = []
            failures: list[tuple[str, str]] = []
            progress = st.progress(0.0, text="Preparing resume evaluation...")
            for index, (filename, pdf_bytes) in enumerate(resume_inputs, start=1):
                progress.progress(
                    (index - 1) / len(resume_inputs),
                    text=f"Evaluating {filename} ({index}/{len(resume_inputs)})",
                )
                try:
                    evaluation = evaluate_resume(
                        pdf_bytes,
                        filename,
                        job_description,
                        llm,
                        embeddings,
                    )
                    results.append(
                        {"filename": filename, "evaluation": evaluation}
                    )
                except Exception as error:
                    failures.append((filename, str(error)))
                progress.progress(
                    index / len(resume_inputs),
                    text=f"Processed {filename} ({index}/{len(resume_inputs)})",
                )
            progress.empty()
            st.session_state["screening_results"] = results
            st.session_state["screening_failures"] = failures

if st.session_state.get("screening_failures"):
    st.warning("Some resumes could not be evaluated:")
    for filename, reason in st.session_state["screening_failures"]:
        st.error(f"{filename}: {reason}")

if st.session_state.get("screening_results"):
    st.header("3. Evaluation results")
    render_results(st.session_state["screening_results"])
