"""End-to-end test of the Stop hook against synthetic transcripts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOT = PROJECT_ROOT / "plugin"
CONSOLE_SCRIPT = PROJECT_ROOT / "venv" / "bin" / "compliance-check-hook"


def write_transcript(tmp_path: Path, user_text: str, assistant_text: str) -> Path:
    path = tmp_path / "transcript.jsonl"
    lines = [
        json.dumps({"type": "user", "message": {"role": "user", "content": user_text}}),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                },
            }
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_hook(transcript_path: Path) -> subprocess.CompletedProcess[str]:
    """Exercise the console-script entrypoint directly.

    The production hook runs the same entrypoint via `uvx --from git+...`,
    but we bypass uvx here to avoid network round-trips in tests.
    """
    payload = json.dumps(
        {"transcript_path": str(transcript_path), "hook_event_name": "Stop"}
    )
    env = os.environ.copy()
    env["CLAUDE_PLUGIN_ROOT"] = str(PLUGIN_ROOT)
    return subprocess.run(
        [str(CONSOLE_SCRIPT)],
        input=payload,
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
    )


@pytest.fixture(scope="module")
def console_script_installed():
    if not CONSOLE_SCRIPT.exists():
        pytest.skip(
            f"console script not found at {CONSOLE_SCRIPT}; "
            "run `uv pip install -e .` in the project venv"
        )


def test_hook_blocks_violating_pair(tmp_path, console_script_installed):
    transcript = write_transcript(
        tmp_path,
        user_text=(
            "Use the context7 MCP tool to fetch the playwright-java docs. "
            "Do NOT use perplexity."
        ),
        assistant_text=(
            "The Context7 MCP tool has exceeded its monthly quota. Let me use the "
            "Perplexity-free approach -- I'll research directly from the Playwright "
            "source documentation and the official Playwright Java GitHub repository "
            "instead."
        ),
    )
    result = run_hook(transcript)
    assert result.returncode == 2, (
        f"expected exit 2 (block) for violation, got {result.returncode}\n"
        f"stderr: {result.stderr}\nstdout: {result.stdout}"
    )
    assert "rule=descope" in result.stderr or "rule=dependency" in result.stderr
    assert "Compliance check failed" in result.stderr
    status = json.loads(result.stdout.strip())
    assert "DENIED" in status["systemMessage"]


def test_hook_allows_clean_pair(tmp_path, console_script_installed):
    transcript = write_transcript(
        tmp_path,
        user_text="Refactor the parser to handle nested expressions",
        assistant_text=(
            "I refactored the parser to handle nested expressions and added 12 tests "
            "covering the edge cases. All tests pass."
        ),
    )
    result = run_hook(transcript)
    assert result.returncode == 0, (
        f"expected exit 0 for clean pair, got {result.returncode}\n"
        f"stderr: {result.stderr}\nstdout: {result.stdout}"
    )
    status = json.loads(result.stdout.strip())
    assert "APPROVED" in status["systemMessage"]
