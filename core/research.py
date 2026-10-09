"""Shared data models and helpers: evidence, timeline events, citation checking.

Key rule: evidence (what tools returned) is stored separately from the model's interpretation
(the answer text). Evidence IDs are assigned by code, never by the model.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

CITATION_RE = re.compile(r"\[([WD]\d+)\]")


@dataclass
class Evidence:
    id: str                      # "W1" (web) or "D1" (document), assigned by code
    kind: str                    # "web" | "doc"
    title: str
    text: str                    # snippet / passage returned by the tool
    url: str = ""                # web only
    source_name: str = ""        # domain or filename
    published: str = ""          # ISO date if the provider returned one
    location: str = ""           # e.g. "page 3" for documents
    score: float = 0.0
    query: str = ""              # the sub-question that retrieved it

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def reference(self) -> str:
        if self.kind == "web":
            date = f" ({self.published})" if self.published else ""
            return f"[{self.id}] {self.title} - {self.source_name}{date}. {self.url}"
        loc = f", {self.location}" if self.location else ""
        return f"[{self.id}] {self.source_name}{loc} (uploaded document)"


@dataclass
class TimelineEvent:
    name: str
    status: str                  # "ok" | "warning" | "error" | "skipped"
    detail: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    question: str
    subquestions: list[str]
    use_web: bool = True
    use_docs: bool = False
    rationale: str = ""
    source: str = "model"        # "model" | "fallback" (deterministic plan if model output unusable)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResearchResult:
    question: str
    answer: str = ""
    plan: Optional[Plan] = None
    evidence: list[Evidence] = field(default_factory=list)
    timeline: list[TimelineEvent] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stop_reason: str = ""
    steps_used: int = 0
    model: str = ""
    ok: bool = False             # True only if a model-written answer was produced from evidence

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {e.id: e for e in self.evidence}


def check_citations(answer: str, evidence: list[Evidence]) -> tuple[list[str], list[str]]:
    """Return (valid_cited_ids, invalid_ids). Invalid IDs are ones the model invented."""
    known = {e.id for e in evidence}
    seen: list[str] = []
    for m in CITATION_RE.finditer(answer or ""):
        cid = m.group(1)
        if cid not in seen:
            seen.append(cid)
    return [c for c in seen if c in known], [c for c in seen if c not in known]


def strip_invalid_citations(answer: str, invalid: list[str]) -> str:
    """Replace citations to non-existent evidence with an explicit marker."""
    for cid in invalid:
        answer = answer.replace(f"[{cid}]", "[unverified citation removed]")
    return answer


_IMG_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")


def strip_markdown_images(text: str) -> str:
    """Remove markdown images from model output: untrusted content must not trigger remote fetches."""
    return _IMG_RE.sub(lambda m: f"[image removed: {m.group(1)}]" if m.group(1) else "[image removed]", text or "")


def build_evidence_context(evidence: list[Evidence], max_chars: int) -> str:
    """Render evidence as clearly delimited, untrusted data blocks, capped to max_chars."""
    blocks: list[str] = []
    used = 0
    per_item = max(400, max_chars // max(1, len(evidence)))
    for e in evidence:
        header = f"<evidence id=\"{e.id}\" type=\"{e.kind}\" source=\"{_attr(e.source_name or e.title)}\""
        if e.location:
            header += f" location=\"{_attr(e.location)}\""
        if e.published:
            header += f" published=\"{_attr(e.published)}\""
        header += ">"
        body = e.text.strip()[:per_item]
        block = f"{header}\n{body}\n</evidence>"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def _attr(s: str) -> str:
    return (s or "").replace('"', "'").replace("\n", " ")[:120]


def extract_section(markdown: str, heading: str) -> str:
    """Return the body under '## heading' (case-insensitive), or '' if missing."""
    pattern = re.compile(
        rf"^#{{2,3}}\s*{re.escape(heading)}\s*$(.*?)(?=^#{{2,3}}\s|\Z)", re.IGNORECASE | re.MULTILINE | re.DOTALL
    )
    m = pattern.search(markdown or "")
    return m.group(1).strip() if m else ""
