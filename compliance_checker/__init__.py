"""Top-level API: run all rules over a piece of text or a user/assistant pair."""

from __future__ import annotations

from .base import PairVerdict, TextVerdict
from .rules import RULES


def check_text(text: str, threshold: float = 0.5) -> list[TextVerdict]:
    return [rule.detect_text(text, threshold=threshold) for rule in RULES]


def check_pair(
    user_request: str,
    assistant_response: str,
    threshold: float = 0.5,
    prior_context: str | None = None,
) -> list[PairVerdict]:
    return [
        rule.detect_pair(
            user_request,
            assistant_response,
            threshold=threshold,
            prior_context=prior_context,
        )
        for rule in RULES
    ]


__all__ = ["PairVerdict", "TextVerdict", "RULES", "check_text", "check_pair"]
