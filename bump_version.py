#!/usr/bin/env python3
"""Bump the patch version in constants.py and pyproject.toml.

Usage:
    python bump_version.py           # patch bump: 0.1.1 → 0.1.2
    python bump_version.py minor     # minor bump: 0.1.1 → 0.2.0
    python bump_version.py major     # major bump: 0.1.1 → 1.0.0
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent

CONSTANTS_FILE = ROOT / "llmmanager" / "constants.py"
PYPROJECT_FILE = ROOT / "pyproject.toml"

CONSTANTS_PATTERN = r'(APP_VERSION\s*=\s*")[^"]+(")'
PYPROJECT_PATTERN = r'(^version\s*=\s*")[^"]+(")'


def parse_version(text: str, pattern: str) -> str:
    m = re.search(pattern, text, re.MULTILINE)
    if not m:
        raise ValueError(f"Version not found with pattern: {pattern!r}")
    full = m.group(0)
    return re.search(r'"([^"]+)"', full).group(1)  # type: ignore[union-attr]


def next_version(current: str, part: str) -> str:
    major, minor, patch = map(int, current.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def replace_version(text: str, pattern: str, new_version: str) -> str:
    def _repl(m: re.Match) -> str:
        return m.group(1) + new_version + m.group(2)
    result, count = re.subn(pattern, _repl, text, count=1, flags=re.MULTILINE)
    if count == 0:
        raise ValueError(f"Pattern not found: {pattern!r}")
    return result


def main() -> None:
    part = sys.argv[1] if len(sys.argv) > 1 else "patch"
    if part not in ("major", "minor", "patch"):
        print(f"Unknown part {part!r} — use major, minor, or patch", file=sys.stderr)
        sys.exit(1)

    # Read current version from constants.py (single source of truth)
    constants_text = CONSTANTS_FILE.read_text(encoding="utf-8")
    current = parse_version(constants_text, CONSTANTS_PATTERN)
    new = next_version(current, part)

    # Patch constants.py
    CONSTANTS_FILE.write_text(
        replace_version(constants_text, CONSTANTS_PATTERN, new),
        encoding="utf-8",
    )

    # Patch pyproject.toml
    pyproject_text = PYPROJECT_FILE.read_text(encoding="utf-8")
    PYPROJECT_FILE.write_text(
        replace_version(pyproject_text, PYPROJECT_PATTERN, new),
        encoding="utf-8",
    )

    print(f"Version bumped: {current} -> {new}")


if __name__ == "__main__":
    main()
