# AI Resume Screening Assistant

A Jupyter Notebook pipeline and Streamlit app that compare one or more PDF
resumes with a job description using LangChain retrieval-augmented generation
(RAG). `ResumeScreening.ipynb` contains the pipeline source code;
`app.py` loads those definitions and provides the Streamlit interface.

## Features

- Loads text from PDF resumes with LangChain's `PyPDFLoader`.
- Splits each resume into overlapping chunks and creates an isolated FAISS index
  for that candidate.
- Retrieves JD-relevant resume excerpts, then evaluates them with the model set
  by `OPENROUTER_LLM_MODEL` (defaults to
  `nvidia/nemotron-3-ultra-550b-a55b:free`) through OpenRouter.
- Creates embeddings locally with
  `sentence-transformers/all-MiniLM-L6-v2`, avoiding paid embedding API calls.
- Parses JSON responses and validates their required fields, value types, and
  recommendation labels with standalone functions (no custom classes).
- Displays a candidate-by-candidate comparison with a match score, strengths,
  weaknesses, missing required skills, and a hiring recommendation.
- Exports evaluations as JSON.

Candidate facts in the evaluation are grounded in retrieved resume excerpts.
The job description supplies the requirements to compare against; it is not
treated as evidence about a candidate. The app is decision support only and
should not replace human review.

## Run locally

1. Use Python 3.10 or newer and install the dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set your OpenRouter API key, or enter the key
   in the app's sidebar.

3. Start Streamlit:

   ```powershell
   streamlit run app.py
   ```

Keep `ResumeScreening.ipynb` beside `app.py`; the Streamlit app loads
the pipeline functions from its code cells in order. The notebook organizes
imports, client setup, PDF parsing, prompt construction, JSON validation, and
candidate evaluation into separate cells without custom classes. The app accepts uploaded
PDFs or the three included sample resumes. Select
**Include the 3 sample resumes**, load the sample Data Scientist job description,
and click **Evaluate resumes** to run a three-candidate demonstration.

## Example evaluation scenarios

The bundled resumes support these demonstrations, all against the same job
description:

1. Evaluate a single resume and review its match score, strengths, gaps, and
   recommendation.
2. Compare all three resumes in the ranked comparison table.
3. Inspect each candidate's missing required skills and the resume evidence for
   each assessment.

Each candidate is retrieved and evaluated independently. A candidate's
assessment cannot use another candidate's resume.

## Deploy

For Streamlit Community Cloud, publish this project, select `app.py` as the
entry point, and add `OPENROUTER_API_KEY` and `OPENROUTER_LLM_MODEL` under the
app's **Settings → Secrets**.
Do not commit `.env` or API keys. Resume embeddings are computed locally, but
retrieved resume excerpts and the job description are sent to OpenRouter for
model evaluation; only process resumes when authorized. The free chat model may
still be subject to OpenRouter availability, rate limits, or account policies.
