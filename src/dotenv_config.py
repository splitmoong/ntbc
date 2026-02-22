"""
Lightweight .env loader — no external dependencies required.

Reads KEY=VALUE lines from a .env file and returns them as a dict.
Lines starting with # and blank lines are ignored.
Values are stripped of surrounding whitespace and optional quotes.

Usage:
    from src.dotenv_config import load_env
    cfg = load_env()          # loads <project_root>/.env
    val = cfg.get("NTBC_PNG_FOLDER", "")
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict


def load_env(env_path: str | Path | None = None) -> Dict[str, str]:
    """Load .env file and return a {KEY: VALUE} dict."""
    if env_path is None:
        # Resolve relative to project root (two levels up from this file: src/dotenv_config.py)
        env_path = Path(__file__).resolve().parent.parent / ".env"
    env_path = Path(env_path)
    if not env_path.exists():
        return {}

    result: Dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip optional surrounding quotes
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def save_env(values: Dict[str, str], env_path: str | Path | None = None) -> None:
    """
    Persist a dict of KEY=VALUE pairs back to .env.
    Existing comment lines and blank lines are preserved.
    Key order follows the existing file; new keys are appended.
    """
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent / ".env"
    env_path = Path(env_path)

    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated_keys: set[str] = set()
    output_lines = []

    for raw_line in existing_lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            output_lines.append(raw_line)
            continue
        key = line.partition("=")[0].strip()
        if key in values:
            output_lines.append(f"{key}={values[key]}")
            updated_keys.add(key)
        else:
            output_lines.append(raw_line)

    # Append any new keys not already in the file
    for key, val in values.items():
        if key not in updated_keys:
            output_lines.append(f"{key}={val}")

    env_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
