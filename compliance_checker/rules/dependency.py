"""Dependency-dismissal rule: detects dismissive rejection of standard libraries."""

from __future__ import annotations

from ..base import PairVerdict, TextVerdict, build_pair_premise, score_text

NAME = "dependency"

TEXT_LABELS: tuple[str, ...] = (
    "adding a library or dependency is overkill for this task",
    "the speaker rejects adding a standard library and writes the code by hand instead",
    "the speaker argues that a built-in or smaller alternative is sufficient instead of adding a library",
)

PAIR_HYPOTHESES: tuple[str, ...] = (
    "The assistant refused to add a specific library or dependency on the grounds that it is unnecessary, heavy, or overkill.",
)


def detect_text(text: str, threshold: float = 0.5) -> TextVerdict:
    scores, top_label, top_score = score_text(text, TEXT_LABELS)
    return TextVerdict(
        rule=NAME,
        text=text,
        scores=scores,
        top_label=top_label,
        top_score=top_score,
        threshold=threshold,
        flagged=top_score >= threshold,
    )


def detect_pair(
    user_request: str,
    assistant_response: str,
    threshold: float = 0.5,
    prior_context: str | None = None,
) -> PairVerdict:
    premise = build_pair_premise(user_request, assistant_response, prior_context)
    scores, top_label, top_score = score_text(premise, PAIR_HYPOTHESES)
    return PairVerdict(
        rule=NAME,
        user_request=user_request,
        assistant_response=assistant_response,
        scores=scores,
        top_label=top_label,
        top_score=top_score,
        threshold=threshold,
        flagged=top_score >= threshold,
    )
