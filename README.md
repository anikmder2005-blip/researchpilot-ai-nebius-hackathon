# ResearchPilot AI

> **From scattered information to evidence-backed intelligence.** 

ResearchPilot AI is a research agent for students, developers and researchers. Give it a question (and optionally
your own PDF / TXT / Markdown documents) and it plans the research, searches the web, retrieves passages from your
documents, and writes a **cited** answer you can inspect and export as a Markdown report.

Built for the **Nebius × NVIDIA Global AI Hackathon 2026** (track: *Best Apps and Agents*).
Main model: an **NVIDIA Nemotron** model served through **Nebius Token Factory**.

## Problem and solution

Research answers from a plain chatbot are hard to trust: no sources, no visible process. ResearchPilot separates
**evidence** (what tools actually returned) from **interpretation** (what the model wrote), forces citations to
evidence IDs, removes citations that point to nothing, and says so when evidence is missing.

## Features

| Area | What it does |
|---|---|
| Agent | Plan → search/retrieve per sub-question → sufficiency check → optional follow-up round → cited answer. Bounded by max tool calls, max rounds and a wall-clock budget. |
| Web research | Tavily search behind a provider interface. Real titles, URLs, dates; clickable links. |
| Document RAG | PDF/TXT/MD upload, validation, chunking, embeddings (Nebius), FAISS vector search, page-level attribution. Says "no evidence" when nothing is relevant. |
| Memory | SQLite projects, preferences, history and user-saved findings. Inspect, delete, clear all. |
| Reports | Downloadable Markdown report: summary, methodology (what actually ran), findings, evidence table, limitations, conclusion, references. |
| Transparency | Activity timeline shows real events only (no private reasoning). |
| Safety | Web pages and documents are treated as untrusted data (fenced, instruction-ignoring prompt, markdown images stripped). |

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the Mermaid diagram.

```
Streamlit UI (app.py, ui/) → ResearchAgent (core/agent.py)
   ├─ planner / reflection / synthesis → Nebius Token Factory → NVIDIA Nemotron
   ├─ TavilyProvider (tools/web_search.py) → web evidence
   ├─ DocumentSearchTool → FAISS index (rag/) ← Nebius embeddings
   ├─ MemoryRepository (SQLite)
   └─ report_generator → Markdown download
```

## Tech stack

Python 3.11+ · Streamlit · `requests` (OpenAI-compatible REST client, no SDK needed) · FAISS (NumPy fallback) ·
PyMuPDF (pypdf fallback) · SQLite · python-dotenv · pytest/unittest.

## Prerequisites

- Python 3.11 or newer
- A **Nebius Token Factory** API key ([tokenfactory.nebius.com](https://tokenfactory.nebius.com))
- Optional: a **Tavily** API key for web search ([tavily.com](https://tavily.com))

## Setup (Windows PowerShell)

```powershell
cd researchpilot-ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env          # paste your keys
python scripts\verify_setup.py --live
streamlit run app.py
```

macOS/Linux: `source .venv/bin/activate` and `cp .env.example .env`.
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope Process Bypass`.

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `NEBIUS_API_KEY` | yes | Token Factory key (chat **and** embeddings) |
| `NEBIUS_MODEL` | yes (default provided) | Nemotron model ID. Default `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` is taken from public catalog listings - **confirm it with `verify_setup.py --live`** |
| `NEBIUS_BASE_URL` | no | Default `https://api.tokenfactory.nebius.com/v1/` |
| `TAVILY_API_KEY` | for web search | Without it, only uploaded documents can be researched |
| `EMBEDDING_MODEL` | no | Default `Qwen/Qwen3-Embedding-8B` (also unconfirmed until you run the live check) |
| `LLM_PROVIDER` | no | `nebius` (default) or `nvidia_nim` (explicit alternative: set `NVIDIA_API_KEY`, `NVIDIA_NIM_MODEL`) |

Limits (`AGENT_MAX_STEPS`, `AGENT_MAX_ROUNDS`, `AGENT_DEADLINE_S`, `MAX_UPLOAD_MB`, `DOC_MIN_SCORE`, ...) are in `.env.example`.

> Nemotron models on Token Factory are **reasoning models**: they spend output tokens thinking. If answers come back
> empty or cut off, raise `MAX_OUTPUT_TOKENS`.

## Using the app

1. Create a **research project** in the sidebar.
2. (Optional) **Documents** → upload PDFs/TXT/MD → *Index selected documents*. Try `sample_data/rag_vs_finetuning_notes.md`.
3. **Research** → ask a question. Watch the activity timeline; open *Evidence* to see every source.
4. Save useful findings to memory; open **Report** to download the Markdown report.

## Tests

```powershell
python -m pytest -q                      # unit tests, no network or keys needed
$env:RUN_LIVE_TESTS=1; python -m pytest tests/test_integration_live.py -v   # optional real Nebius calls
```

Unit tests also run with the standard library: `python -m unittest discover -s tests -t .`

## Deployment on Nebius

Practical path: run the included `Dockerfile` on a **Nebius AI Cloud Compute VM** (small CPU VM is enough because
inference happens on Token Factory, not on the VM).

```bash
docker build -t researchpilot .
docker run -p 8501:8501 --env-file .env -v rp-data:/data researchpilot
```

Open port 8501 only to people you trust (the app has no login). The Dockerfile has **not** been build-tested by the
package author. Token Factory itself is the Nebius service providing the Nemotron inference.

## Privacy

Questions, web snippets and document passages are sent to Nebius (chat + embeddings); queries are sent to Tavily.
Uploaded files are processed in memory and not saved. Memory (history, findings, preferences) is stored **unencrypted**
in `data/researchpilot.db`. API keys are only read from the environment and never logged or stored.

## Limitations

- Output is AI-drafted and **not verified**; check the cited sources.
- Scanned PDFs (images) are not supported (no OCR). No PDF report export.
- The document index is per session and must be rebuilt after a restart.
- `DOC_MIN_SCORE` (default 0.25) is a heuristic; tune it for your embedding model.
- Model/embedding IDs and the Streamlit UI were not verified against live services by the package author (see
  [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md), "Verification status").
- No authentication or multi-user separation.

## Project layout

```
app.py  config/  core/  tools/  rag/  memory/  ui/  tests/  docs/  scripts/  sample_data/
```

License: MIT.
