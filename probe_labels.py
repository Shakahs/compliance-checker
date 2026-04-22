"""Ad-hoc probe: check which label phrasings best separate dependency-dismissal from neutral text."""

from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

clf = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
)

labels = [
    "adding a library or dependency is overkill for this task",
    "the speaker rejects adding a standard library and writes the code by hand instead",
    "the speaker argues that a built-in or smaller alternative is sufficient instead of adding a library",
]

cases = {
    "positive": [
        "adding a new dep just for this is overkill",
        "drop gloo_timers (use web_sys::window().set_timeout_with_callback_and_timeout_and_arguments_0 instead), since adding a new dep just for this is overkill",
        "the pydantic dependency is unnecessary, I'll just write a dataclass",
        "installing requests for one HTTP call is heavy — urllib will do",
        "rather than adding the full library, I'll inline the handful of functions we actually need",
    ],
    "negative": [
        "I installed the dependency you asked for and wired it up in config.py",
        "Added pydantic==2.9 to requirements.txt and the validation now passes",
        "The test suite passes after the refactor.",
    ],
}

for kind, texts in cases.items():
    print(f"\n=== {kind} ===")
    for t in texts:
        r = clf(t, candidate_labels=labels, multi_label=True)
        ordered = sorted(zip(r["labels"], r["scores"]), key=lambda x: -x[1])
        best_label, best_score = ordered[0]
        print(f"  best={best_score:.3f}  {t[:80]}")
        print(f"    -> {best_label}")
