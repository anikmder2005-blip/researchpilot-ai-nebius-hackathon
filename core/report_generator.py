"""Markdown report generation. Deterministic: it assembles what the agent actually produced."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from core.research import Evidence, ResearchResult, extract_section

DISCLAIMER = (
    "> **Notice:** This report was drafted by an AI model from the sources listed below. It has not "
    "been independently verified or scientifically validated. Check the cited sources before relying on it."
)

REQUIRED_SECTIONS = (
    "Executive Summary",
    "Methodology",
    "Key Findings",
    "Evidence and Sources",
    "Limitations and Unresolved Questions",
    "Conclusion",
    "References",
)


def build_methodology(result: ResearchResult) -> str:
    web = [e for e in result.evidence if e.kind == "web"]
    docs = [e for e in result.evidence if e.kind == "doc"]
    lines = [
        f"- Model: `{result.model}` (NVIDIA Nemotron) used to plan the research and write the answer.",
    ]
    if result.plan:
        plan_kind = "model-generated" if result.plan.source == "model" else "deterministic fallback"
        lines.append(f"- Research plan ({plan_kind}) with {len(result.plan.subquestions)} sub-question(s):")
        lines += [f"  {i}. {q}" for i, q in enumerate(result.plan.subquestions, 1)]
    lines.append(f"- Tool calls used: {result.steps_used}. Stop reason: {result.stop_reason or 'n/a'}.")
    lines.append(f"- Evidence collected: {len(web)} web source(s), {len(docs)} document passage(s).")
    lines.append("- Claims are cited with evidence IDs; citations to non-existent evidence are removed automatically.")
    return "\n".join(lines)


def _evidence_table(evidence: list[Evidence]) -> str:
    if not evidence:
        return "_No evidence was collected._"
    rows = ["| ID | Type | Source | Location / Date | Link |", "|---|---|---|---|---|"]
    for e in evidence:
        where = e.location or e.published or "-"
        link = f"[open]({e.url})" if e.url else "-"
        title = (e.title or e.source_name).replace("|", "\\|")[:80]
        rows.append(f"| {e.id} | {'Web' if e.kind == 'web' else 'Document'} | {title} | {where} | {link} |")
    return "\n".join(rows)


def build_report(
    result: ResearchResult,
    title: Optional[str] = None,
    saved_findings: Optional[list[dict]] = None,
    generated_at: Optional[datetime] = None,
) -> str:
    """Assemble the Markdown report. Section content comes from the model answer where present."""
    when = (generated_at or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")
    answer = result.answer or ""
    summary = extract_section(answer, "Executive Summary")
    findings = extract_section(answer, "Key Findings")
    comparison = extract_section(answer, "Comparison")
    conflicts = extract_section(answer, "Conflicts and Uncertainty")
    limits = extract_section(answer, "Limitations")
    conclusion = extract_section(answer, "Conclusion")
    if not (summary or findings or conclusion):
        # The model ignored the template: keep its text rather than lose it.
        findings = answer.strip() or "_No answer was generated._"
        summary = "_The model did not provide a separate executive summary; see Key Findings._"

    parts = [
        f"# {title or result.question[:120]}",
        "",
        f"**Research question:** {result.question}",
        f"**Generated:** {when}  ",
        f"**Model:** {result.model or 'n/a'}",
        "",
        DISCLAIMER,
        "",
        "## Executive Summary",
        summary or "_Not provided._",
        "",
        "## Methodology",
        build_methodology(result),
        "",
        "## Key Findings",
        findings or "_Not provided._",
    ]
    if comparison and comparison.strip().lower().rstrip(".") != "not applicable":
        parts += ["", "### Comparison", comparison]
    if saved_findings:
        parts += ["", "### Findings saved by the researcher"]
        parts += [f"- {f['text']}" for f in saved_findings]
    parts += ["", "## Evidence and Sources", _evidence_table(result.evidence)]
    parts += ["", "## Limitations and Unresolved Questions"]
    if conflicts:
        parts += ["**Conflicts and uncertainty**", conflicts, ""]
    parts += [limits or "_The model did not list limitations. Treat all findings as unverified._"]
    if result.errors:
        parts += ["", "**Tool problems during this run:**"] + [f"- {e}" for e in dict.fromkeys(result.errors)]
    if result.invalid_citations:
        parts += ["", f"- {len(result.invalid_citations)} citation(s) to non-existent evidence were removed."]
    parts += ["", "## Conclusion", conclusion or "_Not provided._", "", "## References"]
    if result.evidence:
        cited = set(result.cited_ids)
        parts += [f"{e.reference}{'' if e.id in cited else '  _(consulted, not cited)_'}" for e in result.evidence]
    else:
        parts.append("_No references: no evidence was collected._")
    return "\n".join(parts).strip() + "\n"


def report_filename(result: ResearchResult) -> str:
    slug = "".join(c.lower() if c.isalnum() else "-" for c in result.question[:50]).strip("-") or "report"
    while "--" in slug:
        slug = slug.replace("--", "-")
    return f"researchpilot-{slug}.md"
