"""Shared primitives: classifier loader, verdict types, pair premise builder."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


@lru_cache(maxsize=1)
def classifier():
    return pipeline("zero-shot-classification", model=MODEL_NAME)


@dataclass(frozen=True)
class TextVerdict:
    rule: str
    text: str
    scores: dict[str, float]
    top_label: str
    top_score: float
    threshold: float
    flagged: bool


@dataclass(frozen=True)
class PairVerdict:
    rule: str
    user_request: str
    assistant_response: str
    scores: dict[str, float]
    top_label: str
    top_score: float
    threshold: float
    flagged: bool


def build_pair_premise(
    user_request: str,
    assistant_response: str,
    prior_context: str | None = None,
) -> str:
    parts = []
    if prior_context:
        parts.append(f"Prior context: {prior_context}")
    parts.append(f"User request: {user_request}")
    parts.append(f"Assistant response: {assistant_response}")
    return "\n\n".join(parts)


def score_text(
    text: str,
    labels: tuple[str, ...],
) -> tuple[dict[str, float], str, float]:
    raw = classifier()(text, candidate_labels=list(labels), multi_label=True)
    scores = dict(zip(raw["labels"], raw["scores"]))
    return scores, raw["labels"][0], raw["scores"][0]
