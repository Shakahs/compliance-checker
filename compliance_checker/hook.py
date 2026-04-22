"""Stop-hook entrypoint. Reads the Claude Code Stop-hook JSON on stdin, scans
the last assistant turn in the transcript, and exits 2 with corrective feedback
on stderr if any rule flags the turn.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import check_pair


def extract_last_pair(transcript_path: Path) -> tuple[str, str] | None:
    """Walk the transcript and return the most recent (user_text, assistant_text).

    Returns None if the transcript ended on a user message (turn in progress)
    or lacks either side of the pair.
    """
    last_user: str | None = None
    last_assistant: str | None = None
    user_after_assistant = False

    with transcript_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = obj.get("type")
            msg = obj.get("message")
            if not isinstance(msg, dict):
                continue

            if kind == "user":
                content = msg.get("content")
                text = content if isinstance(content, str) else None
                # Skip synthetic tool-result blocks (serialized arrays)
                if text and not text.lstrip().startswith("[{"):
                    last_user = text
                    user_after_assistant = True
            elif kind == "assistant":
                blocks = msg.get("content") or []
                texts = [
                    b.get("text", "")
                    for b in blocks
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                joined = "\n".join(t for t in texts if t).strip()
                if joined:
                    last_assistant = joined
                    user_after_assistant = False

    if last_user and last_assistant and not user_after_assistant:
        return last_user, last_assistant
    return None


def format_feedback(flagged_verdicts) -> str:
    lines = [
        "Compliance check failed. Review the user's instructions and respond again "
        "without descoping or dismissing the requested approach.",
        "",
    ]
    for v in flagged_verdicts:
        lines.append(
            f"- rule={v.rule} score={v.top_score:.2f}\n"
            f"  hypothesis: {v.top_label}"
        )
    lines.append("")
    lines.append(
        "If you genuinely cannot fulfill the request as stated, surface the "
        "blocker explicitly to the user instead of substituting a different approach."
    )
    return "\n".join(lines) + "\n"


def emit_status(message: str) -> None:
    """Print a JSON status block Claude Code surfaces to the user."""
    print(json.dumps({"systemMessage": message}))


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        emit_status("compliance-checker: skipped (no transcript_path)")
        sys.exit(0)

    path = Path(transcript_path)
    if not path.exists():
        emit_status(f"compliance-checker: skipped (transcript missing: {path})")
        sys.exit(0)

    pair = extract_last_pair(path)
    if pair is None:
        emit_status("compliance-checker: skipped (no completed user/assistant pair)")
        sys.exit(0)

    user_request, assistant_response = pair
    verdicts = check_pair(user_request, assistant_response)
    flagged = [v for v in verdicts if v.flagged]

    if not flagged:
        scores = ", ".join(f"{v.rule}={v.top_score:.2f}" for v in verdicts)
        emit_status(f"compliance-checker: APPROVED ({scores})")
        sys.exit(0)

    rules_hit = ", ".join(f"{v.rule}={v.top_score:.2f}" for v in flagged)
    emit_status(f"compliance-checker: DENIED (flagged: {rules_hit})")
    sys.stderr.write(format_feedback(flagged))
    sys.exit(2)


if __name__ == "__main__":
    main()
