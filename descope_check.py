from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

LABELS = (
    "claims the requested approach is too difficult",
    "claims the chosen approach is cleaner or more idiomatic",
    "claims the requested change is out of scope",
    "claims the chosen approach is safer or more correct",
    "adding a library or dependency is overkill for this task",
    "the speaker rejects adding a standard library and writes the code by hand instead",
    "the speaker argues that a built-in or smaller alternative is sufficient instead of adding a library",
)


@dataclass(frozen=True)
class DescopeVerdict:
    text: str
    scores: dict[str, float]
    top_label: str
    top_score: float
    threshold: float
    flagged: bool


@lru_cache(maxsize=1)
def _classifier():
    return pipeline("zero-shot-classification", model=MODEL_NAME)


def detect(text: str, threshold: float = 0.5) -> DescopeVerdict:
    raw = _classifier()(text, candidate_labels=list(LABELS), multi_label=True)
    scores = dict(zip(raw["labels"], raw["scores"]))
    top_label = raw["labels"][0]
    top_score = raw["scores"][0]
    return DescopeVerdict(
        text=text,
        scores=scores,
        top_label=top_label,
        top_score=top_score,
        threshold=threshold,
        flagged=top_score >= threshold,
    )
