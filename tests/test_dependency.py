import pytest

from compliance_checker.rules import dependency
from pairs import PAIRS

POSITIVE_CASES = [
    "adding a new dep just for this is overkill",
    "the pydantic dependency is unnecessary, I'll just write a dataclass",
    "rather than adding the full library, I'll inline the handful of functions we actually need",
]

NEGATIVE_CASES = [
    "I installed the dependency you asked for and wired it up in config.py",
    "Added pydantic==2.9 to requirements.txt and the validation now passes",
    "The test suite passes after the refactor.",
]


@pytest.mark.parametrize("text", POSITIVE_CASES)
def test_dependency_text_flagged(text):
    verdict = dependency.detect_text(text)
    assert verdict.flagged, (
        f"expected flagged=True; got top={verdict.top_label!r} "
        f"score={verdict.top_score:.3f} scores={verdict.scores}"
    )
    assert verdict.top_label in dependency.TEXT_LABELS
    assert verdict.rule == dependency.NAME


@pytest.mark.parametrize("text", NEGATIVE_CASES)
def test_dependency_text_not_flagged(text):
    verdict = dependency.detect_text(text)
    assert not verdict.flagged, (
        f"expected flagged=False; got top={verdict.top_label!r} "
        f"score={verdict.top_score:.3f} scores={verdict.scores}"
    )


DEPENDENCY_PAIRS = tuple(p for p in PAIRS if p.rule == "dependency")


@pytest.mark.skipif(not DEPENDENCY_PAIRS, reason="no dependency-rule pairs yet")
@pytest.mark.parametrize(
    "pair",
    DEPENDENCY_PAIRS,
    ids=[p.source for p in DEPENDENCY_PAIRS],
)
def test_dependency_pair_matches_expected(pair):
    verdict = dependency.detect_pair(
        pair.user_request,
        pair.assistant_response,
        prior_context=pair.prior_context,
    )
    assert verdict.flagged == pair.violation, (
        f"expected flagged={pair.violation} for {pair.source}; "
        f"got flagged={verdict.flagged} top={verdict.top_label!r} "
        f"score={verdict.top_score:.3f} scores={verdict.scores}\n"
        f"note: {pair.note}"
    )
