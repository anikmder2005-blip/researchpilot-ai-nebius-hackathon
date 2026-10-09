"""ResearchPilot AI - Streamlit research workspace.

Run:  streamlit run app.py
"""
from __future__ import annotations

import logging

import streamlit as st

from config.settings import load_settings
from core.agent import ResearchAgent
from core.llm import LLMClient
from core.report_generator import build_report, report_filename
from core.research import extract_section
from memory.database import connect
from memory.repository import MemoryRepository
from rag.embeddings import EmbeddingError, NebiusEmbedder
from rag.ingestion import IngestionError, extract_pages
from rag.retrieval import DocumentStore
from tools.document_search import DocumentSearchTool
from tools.web_search import TavilyProvider
from ui.components import (
    chip, md_escape, render_result, render_setup_problems, render_status_chips,
)
from ui.styles import CSS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

st.set_page_config(page_title="ResearchPilot AI", page_icon="🧭", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

PAGES = ["Research", "Documents", "History", "Memory", "Report", "Status"]


# ----------------------------------------------------------------------------- state
@st.cache_resource
def get_repo(db_path: str) -> MemoryRepository:
    return MemoryRepository(connect(db_path))


def init_state() -> None:
    ss = st.session_state
    if "settings" not in ss:
        ss.settings = load_settings()
    if "store" not in ss:
        s = ss.settings
        ss.store = DocumentStore(NebiusEmbedder(s), s.chunk_size, s.chunk_overlap)
    ss.setdefault("results", {})      # project_id -> [{"question", "result"}]
    ss.setdefault("project_id", None)


init_state()
settings = st.session_state.settings
store: DocumentStore = st.session_state.store
repo = get_repo(settings.db_path)


def current_project() -> dict | None:
    pid = st.session_state.project_id
    return repo.get_project(pid) if pid else None


# ----------------------------------------------------------------------------- sidebar
def sidebar() -> str:
    with st.sidebar:
        st.markdown('<p class="rp-brand">🧭 ResearchPilot AI</p>', unsafe_allow_html=True)
        st.markdown('<p class="rp-tag">From scattered information to evidence-backed intelligence.</p>', unsafe_allow_html=True)

        projects = repo.list_projects()
        ids = [p["id"] for p in projects]
        if projects:
            if st.session_state.project_id not in ids:
                st.session_state.project_id = ids[0]
            st.session_state.project_id = st.selectbox(
                "Research project",
                ids,
                index=ids.index(st.session_state.project_id),
                format_func=lambda i: next(p["title"] for p in projects if p["id"] == i),
            )
        with st.expander("New research project", expanded=not projects):
            with st.form("new_project", clear_on_submit=True):
                title = st.text_input("Project title", placeholder="e.g. RAG vs fine-tuning for my thesis")
                if st.form_submit_button("Create project", use_container_width=True):
                    try:
                        st.session_state.project_id = repo.create_project(title)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.divider()
        render_status_chips(settings, len(store.documents))
        st.caption("Your questions go to the configured model/search APIs. See Status for privacy details.")
    return page


# ----------------------------------------------------------------------------- pages
def memory_note(pid: int) -> str:
    """User-approved context only: saved preferences and saved findings."""
    lines = [f"- Preference {k}: {v}" for k, v in repo.get_preferences(pid).items()]
    lines += [f"- Saved finding: {f['text']}" for f in repo.list_findings(pid)[:10]]
    return "\n".join(lines)


def run_research(question: str, pid: int) -> None:
    llm = LLMClient(settings)
    search = TavilyProvider(settings.tavily_api_key) if settings.search_ready else None
    doc_tool = DocumentSearchTool(store, settings.doc_top_k, settings.doc_min_score) if settings.embeddings_ready else None
    agent = ResearchAgent(llm, settings, search=search, doc_tool=doc_tool)

    with st.chat_message("user"):
        st.markdown(md_escape(question))
    with st.chat_message("assistant"):
        with st.status("Researching...", expanded=True) as status:
            def on_event(ev):
                icon = {"ok": "✅", "warning": "⚠️", "error": "❌", "skipped": "⏭️"}.get(ev.status, "•")
                st.write(f"{icon} **{ev.name}** {('- ' + ev.detail) if ev.detail else ''}")
            result = agent.run(question, on_event=on_event, context_note=memory_note(pid))
            status.update(label="Done" if result.ok else "Finished with problems", state="complete" if result.ok else "error")
    st.session_state.results.setdefault(pid, []).append({"question": question, "result": result})
    if result.ok:
        repo.add_question(
            pid, question, extract_section(result.answer, "Executive Summary")[:800],
            result.answer, [e.to_dict() for e in result.evidence],
        )
    st.rerun()


def page_research(project: dict | None) -> None:
    st.header("Research")
    if not render_setup_problems(settings):
        return
    if not project:
        st.info("Create a research project in the sidebar to begin.")
        return
    st.caption(f"Project: **{md_escape(project['title'])}**")
    if not settings.search_ready and len(store) == 0:
        st.warning("No research tools are ready: add `TAVILY_API_KEY` for web search and/or upload a document.")

    pid = project["id"]
    items = st.session_state.results.get(pid, [])
    if not items:
        st.markdown(
            "Ask a research question below. The agent will plan, search the web and your documents, "
            "and answer with citations you can inspect.\n\n"
            "*Example:* Compare RAG and fine-tuning for a university project and recommend an approach "
            "based on cost, complexity, and evaluation requirements."
        )
    for i, item in enumerate(items):
        with st.chat_message("user"):
            st.markdown(md_escape(item["question"]))
        with st.chat_message("assistant"):
            res = item["result"]
            render_result(res, key=f"r{pid}-{i}")
            c1, c2 = st.columns([3, 1])
            with c1:
                with st.expander("Save a finding to project memory"):
                    default = extract_section(res.answer, "Conclusion")[:500] if res.ok else ""
                    text = st.text_area("Finding", value=default, key=f"f{pid}-{i}")
                    if st.button("Save finding", key=f"sf{pid}-{i}"):
                        cited = [e.to_dict() for e in res.evidence if e.id in res.cited_ids]
                        try:
                            repo.add_finding(pid, text, cited)
                            st.success("Saved to memory.")
                        except ValueError as exc:
                            st.error(str(exc))
            with c2:
                if not res.ok and st.button("Retry", key=f"retry{pid}-{i}"):
                    st.session_state.retry_question = item["question"]
                    st.rerun()

    question = st.chat_input("Ask a research question...") or st.session_state.pop("retry_question", None)
    if question is not None:
        if not question.strip():
            st.warning("Please enter a research question.")
        else:
            run_research(question.strip(), pid)


def page_documents() -> None:
    st.header("Documents")
    st.caption("Upload PDF, TXT or Markdown files. Their text is chunked and embedded for retrieval; "
               "the files themselves are not saved to disk.")
    if not settings.embeddings_ready:
        st.warning("Embeddings are not configured (needs `NEBIUS_API_KEY` or `EMBEDDING_API_KEY`), so uploads are disabled.")
        return
    files = st.file_uploader("Add documents", type=["pdf", "txt", "md", "markdown"], accept_multiple_files=True)
    if files and st.button("Index selected documents", type="primary"):
        for f in files:
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    name, pages = extract_pages(f.name, f.getvalue(), settings.max_upload_mb)
                    n = store.add_document(name, pages)
                    st.success(f"{name}: {n} passages indexed" + (f" across {len(pages)} pages." if pages[0][0] else "."))
                except (IngestionError, EmbeddingError) as exc:
                    st.error(f"{f.name}: {exc}")
    docs = store.documents
    if not docs:
        st.info("No documents indexed yet.")
        return
    st.subheader("Indexed documents")
    st.caption(f"Vector index backend: {store.backend}")
    for name, n in docs.items():
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{md_escape(name)}** - {n} passages")
        if c2.button("Remove", key=f"rm-{name}"):
            store.remove_document(name)
            st.rerun()


def page_history(project: dict | None) -> None:
    st.header("Research history")
    if not project:
        st.info("Create a project first.")
        return
    questions = repo.list_questions(project["id"])
    if not questions:
        st.info("No saved research yet. Answers are saved automatically after a successful run.")
        return
    for q in questions:
        with st.expander(f"{q['question'][:90]}  ·  {q['created_at']}"):
            st.markdown(q["answer"] or "_No answer saved._")
            if q["sources"]:
                st.markdown("**Sources**")
                for s in q["sources"]:
                    label = md_escape(s.get("title") or s.get("source_name") or s.get("id", ""))
                    st.markdown(f"- [{s['id']}] " + (f"[{label}]({s['url']})" if s.get("url") else f"{label} {s.get('location', '')}"))
            if st.button("Delete this entry", key=f"dq{q['id']}"):
                repo.delete_question(q["id"])
                st.rerun()


def page_memory(project: dict | None) -> None:
    st.header("Memory")
    st.caption("Only what is listed here is stored (in a local SQLite file). No API keys, no full documents.")
    if project:
        pid = project["id"]
        st.subheader("Research preferences")
        prefs = repo.get_preferences(pid)
        for k, v in prefs.items():
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{md_escape(k)}:** {md_escape(v)}")
            if c2.button("Delete", key=f"dp-{k}"):
                repo.delete_preference(pid, k)
                st.rerun()
        with st.form("pref_form", clear_on_submit=True):
            k = st.text_input("Preference name", placeholder="e.g. audience")
            v = st.text_input("Value", placeholder="e.g. undergraduate students")
            if st.form_submit_button("Save preference") and k.strip() and v.strip():
                repo.set_preference(pid, k.strip(), v.strip())
                st.rerun()
        st.subheader("Saved findings")
        findings = repo.list_findings(pid)
        if not findings:
            st.info("No findings saved. Use 'Save a finding' under an answer.")
        for f in findings:
            c1, c2 = st.columns([5, 1])
            c1.markdown(md_escape(f["text"]))
            if c2.button("Delete", key=f"df-{f['id']}"):
                repo.delete_finding(f["id"])
                st.rerun()
        st.divider()
        if st.button("Delete this project and all its data"):
            repo.delete_project(pid)
            st.session_state.results.pop(pid, None)
            st.session_state.project_id = None
            st.rerun()
    st.subheader("Inspect everything stored")
    counts = repo.counts()
    st.caption(f"{counts['projects']} project(s), {counts['questions']} saved answer(s), "
               f"{counts['findings']} finding(s), {counts['preferences']} preference(s).")
    with st.expander("Show raw stored data"):
        st.json(repo.export_all())
    st.subheader("Clear all memory")
    confirm = st.checkbox("I understand this permanently deletes all projects, history and findings.")
    if st.button("Clear all memory", disabled=not confirm):
        repo.clear_all()
        st.session_state.results = {}
        st.session_state.project_id = None
        st.rerun()


def page_report(project: dict | None) -> None:
    st.header("Report")
    if not project:
        st.info("Create a project first.")
        return
    pid = project["id"]
    items = [i for i in st.session_state.results.get(pid, [])]
    if not items:
        st.info("Run a research question first. Reports are built from this session's runs, which include "
                "the plan and activity log.")
        return
    idx = st.selectbox("Question", range(len(items)), index=len(items) - 1,
                       format_func=lambda i: items[i]["question"][:100])
    res = items[idx]["result"]
    if not res.ok:
        st.warning("This run did not produce an answer; the report will say so and list any tool problems.")
    md = build_report(res, title=project["title"], saved_findings=repo.list_findings(pid))
    st.download_button("Download Markdown report", md, file_name=report_filename(res), mime="text/markdown", type="primary")
    st.caption("PDF export is not included. The report is AI-drafted and unverified.")
    st.markdown("---")
    st.markdown(md)


def page_status() -> None:
    st.header("Status")
    st.markdown(
        chip(f"Provider: {settings.provider_label}", "ok" if settings.llm_ready else "err")
        + chip("Tavily search", "ok" if settings.search_ready else "warn")
        + chip("Embeddings", "ok" if settings.embeddings_ready else "warn"),
        unsafe_allow_html=True,
    )
    st.table({
        "Setting": ["Chat model", "Chat endpoint", "Embedding model", "Max tool calls / run", "Max rounds", "Time budget (s)",
                    "Upload limit (MB)", "Vector backend", "Database file"],
        "Value": [settings.llm_model, settings.llm_base_url, settings.embedding_model, settings.agent_max_steps,
                  settings.agent_max_rounds, settings.agent_deadline_s, settings.max_upload_mb, store.backend, settings.db_path],
    })
    for p in settings.validate(need_search=True):
        st.warning(p)
    st.subheader("Privacy")
    st.markdown(
        "- Questions, retrieved web snippets and document passages are sent to the model provider "
        f"({settings.provider_label}); document passages are sent to the embedding endpoint; queries go to Tavily.\n"
        "- Uploaded files are processed in memory and are not written to disk.\n"
        "- Saved history, findings and preferences are stored unencrypted in the local SQLite file shown above.\n"
        "- API keys are read from environment variables and never stored or logged."
    )
    st.caption("To verify the live connection and model IDs run: `python scripts/verify_setup.py --live`")


# ----------------------------------------------------------------------------- main
page = sidebar()
project = current_project()
if page == "Research":
    page_research(project)
elif page == "Documents":
    page_documents()
elif page == "History":
    page_history(project)
elif page == "Memory":
    page_memory(project)
elif page == "Report":
    page_report(project)
else:
    page_status()
