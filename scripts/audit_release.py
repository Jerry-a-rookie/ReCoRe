"""Verify the public ReCoRe package is self-contained."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "ts_benchmark" / "baselines" / "recore_public"
FORBIDDEN = ("pri" + "sm", "pf" + "rp")


def main() -> int:
    violations = []
    for path in PUBLIC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            if token in text:
                violations.append(f"{path}: {token}")
    if violations:
        raise SystemExit("Public ReCoRe naming audit failed:\n" + "\n".join(violations))
    print(f"Public ReCoRe naming audit passed: {PUBLIC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
