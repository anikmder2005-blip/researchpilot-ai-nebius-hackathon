"""Research planning: turn a question into a short plan with sub-questions.

The model proposes the plan as JSON. If its output is unusable we fall back to a minimal
deterministic plan and say so (Plan.source == "fallback") - we never pretend the model planned.
"""
from __future__ import annotations

import logging
from typing import Optional

from core.llm import LLMClient, LLMError
from core.research import Plan

log = logging.getLogger("researchpilot.planner")

MAX_SUBQUESTIONS = 4

PLANNER_SYSTEM = (
    "You are a research planner. Given a research question, output ONLY a JSON object with keys: "
    '"subquestions" (2-4 short, specific, searchable questions), '
    '"use_web" (boolean, true if current/public information is needed), '
    '"use_docs" (boolean, true only if uploaded documents are available AND likely relevant), '
    '"rationale" (one sentence). No prose outside the JSON.'
)


def fallback_plan(question: str, has_docs: bool, web_available: bool) -> Plan:
    return Plan(
        question=question,
        subquestions=[question.strip()],
        use_web=web_available,
        use_docs=has_docs,
        rationale="Model plan unavailable; searching with the original question.",
        source="fallback",
    )


def make_plan(
    llm: Optional[LLMClient], question: str, has_docs: bool, web_available: bool
) -> Plan:
    question = question.strip()
    if llm is None:
        return fallback_plan(question, has_docs, web_available)
    user = (
        f"Research question: {question}\n"
        f"Uploaded documents available: {'yes' if has_docs else 'no'}\n"
        f"Web search available: {'yes' if web_available else 'no'}"
    )
    try:
        data = llm.chat_json(
            [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=1500,
        )
    except LLMError as exc:
        log.warning("planner failed: %s", exc)
        plan = fallback_plan(question, has_docs, web_available)
        plan.rationale = f"Planner call failed ({exc}); searching with the original question."
        return plan
    return parse_plan(data, question, has_docs, web_available)


def parse_plan(data: Optional[dict], question: str, has_docs: bool, web_available: bool) -> Plan:
    if not data:
        return fallback_plan(question, has_docs, web_available)
    raw = data.get("subquestions")
    subs: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip() and item.strip() not in subs:
                subs.append(item.strip()[:300])
    subs = subs[:MAX_SUBQUESTIONS]
    if not subs:
        return fallback_plan(question, has_docs, web_available)
    # Tool availability is decided by the application, not by the model.
    use_web = bool(data.get("use_web", True)) and web_available
    use_docs = bool(data.get("use_docs", has_docs)) and has_docs
    if not use_web and not use_docs:
        use_web, use_docs = web_available, has_docs
    return Plan(
        question=question,
        subquestions=subs,
        use_web=use_web,
        use_docs=use_docs,
        rationale=str(data.get("rationale", ""))[:300],
        source="model",
    )
