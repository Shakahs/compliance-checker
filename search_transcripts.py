"""Single-pass zero-shot descoping-detection over Claude Code transcripts.

Local: stream assistant-message chunks from .jsonl files, batch them.
Remote (Modal GPU): score each chunk with a zero-shot NLI cross-encoder
against descoping hypotheses. The cross-encoder jointly attends to
(premise, hypothesis), so negation and direction (did vs. didn't) register —
unlike bi-encoder cosine similarity, which keyword-matches.

Run with: modal run search_transcripts.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import modal

TRANSCRIPTS_ROOT = Path("/home/shak/.claude/projects")
MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
GLOBAL_QUOTA = 25
THRESHOLD = 0.85
BATCH_SIZE = 64
MIN_CHARS = 40
MAX_CHARS = 600
GPU_TYPE = "H100"
# GPU_TYPE = "RTX-PRO-6000"

EXCLUDE_PATH_SUBSTR = (
    "compliance-checker",
    "tweakcc-system-prompts",
)

HYPOTHESES = (
    "claims the requested approach is too difficult",
    "claims the chosen approach is cleaner or more idiomatic",
    "claims the requested change is out of scope",
    "claims the chosen approach is safer or more correct",
    "adding a library or dependency is overkill for this task",
    "the speaker rejects adding a standard library and writes the code by hand instead",
    "the speaker argues that a built-in or smaller alternative is sufficient instead of adding a library",
)

app = modal.App("descope-transcript-search")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("transformers==5.6.0", "torch==2.11.0")
    .run_commands(
        "python -c 'from transformers import pipeline; "
        f"pipeline(\"zero-shot-classification\", model=\"{MODEL_NAME}\")'"
    )
)


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    scaledown_window=120,
    secrets=[modal.Secret.from_name("huggingface")],
)
class Classifier:
    @modal.enter()
    def load(self):
        from transformers import pipeline
        self.pipe = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=0,
        )
        self.hypotheses = HYPOTHESES

    @modal.method()
    def score(self, texts: list[str]) -> list[dict]:
        """For each text, return {labels: [...], scores: [...]} sorted desc."""
        results = self.pipe(
            texts,
            candidate_labels=list(self.hypotheses),
            multi_label=True,
            batch_size=len(texts),
        )
        # pipeline returns a single dict for a single input, list for list input
        if isinstance(results, dict):
            results = [results]
        return [{"labels": r["labels"], "scores": r["scores"]} for r in results]


@dataclass
class Chunk:
    text: str
    transcript: str
    line_no: int


@dataclass
class Hit:
    score: float
    query: str
    chunk: Chunk


def iter_assistant_texts(path: Path) -> Iterator[tuple[int, str]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or '"type":"assistant"' not in line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = obj.get("message")
            if not isinstance(msg, dict):
                continue
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text") or ""
                    if text:
                        yield line_no, text


def split_chunks(text: str) -> Iterator[str]:
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= MAX_CHARS:
                buf = buf + " " + s
            else:
                if len(buf) >= MIN_CHARS:
                    yield buf
                buf = s
        if buf and len(buf) >= MIN_CHARS:
            yield buf


def iter_all_chunks(root: Path) -> Iterator[Chunk]:
    files = sorted(root.rglob("*.jsonl"))
    for path in files:
        rel = str(path.relative_to(root))
        if any(s in rel for s in EXCLUDE_PATH_SUBSTR):
            continue
        for line_no, text in iter_assistant_texts(path):
            for chunk in split_chunks(text):
                yield Chunk(text=chunk, transcript=rel, line_no=line_no)


@app.local_entrypoint()
def main():
    classifier = Classifier()

    print(
        f"Streaming {TRANSCRIPTS_ROOT} "
        f"(threshold={THRESHOLD}, quota={GLOBAL_QUOTA}, batch={BATCH_SIZE})\n",
        flush=True,
    )

    hits: list[Hit] = []
    processed = 0
    buf: list[Chunk] = []

    def process_batch(batch: list[Chunk]) -> bool:
        nonlocal processed
        if not batch:
            return False
        results = classifier.score.remote([c.text for c in batch])
        for chunk, res in zip(batch, results):
            top_label = res["labels"][0]
            top_score = float(res["scores"][0])
            if top_score >= THRESHOLD:
                hit = Hit(score=top_score, query=top_label, chunk=chunk)
                hits.append(hit)
                print(
                    f"[HIT {len(hits)}/{GLOBAL_QUOTA}] score={hit.score:.3f}\n"
                    f"  label: {hit.query}\n"
                    f"  {chunk.transcript}:{chunk.line_no}\n"
                    f"  {chunk.text}\n",
                    flush=True,
                )
                if len(hits) >= GLOBAL_QUOTA:
                    return True
        processed += len(batch)
        if processed % (BATCH_SIZE * 10) == 0:
            print(f"  ...scanned {processed} chunks, {len(hits)} hits", flush=True)
        return False

    stopped = False
    for chunk in iter_all_chunks(TRANSCRIPTS_ROOT):
        buf.append(chunk)
        if len(buf) >= BATCH_SIZE:
            if process_batch(buf):
                stopped = True
                break
            buf = []
    if not stopped:
        process_batch(buf)

    print("\n" + "=" * 80)
    print(f"Stopped with {len(hits)} hits after {processed} chunks scanned.")
    for i, hit in enumerate(hits, start=1):
        print(f"\n[{i}] score={hit.score:.3f}")
        print(f"    label: {hit.query}")
        print(f"    {hit.chunk.transcript}:{hit.chunk.line_no}")
        print(f"    {hit.chunk.text}")
