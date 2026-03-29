"""Phase 4b — KB Diff: human-readable diff between two market_knowledge.json versions.

Compares entries by ticker, grouped by section, and reports:
  + ADDED     — ticker present in NEW but not OLD
  - REMOVED   — ticker present in OLD but not NEW
  ~ MODIFIED  — ticker present in both, but one or more fields changed

Usage:
    python -m scripts.kb_diff old_kb.json new_kb.json
    python -m scripts.kb_diff data/market_knowledge.json.bak data/market_knowledge.json
    python -m scripts.kb_diff --no-values ...   # show changed field names only (no old/new text)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Max characters shown for a field value in the diff output.
_TRUNCATE = 120


def _truncate(val: object, max_len: int = _TRUNCATE) -> str:
    s = str(val)
    if len(s) > max_len:
        return s[:max_len] + " …"
    return s


def _diff_entry(old: dict, new: dict, *, show_values: bool) -> list[str]:
    """Return lines describing field-level changes between *old* and *new*."""
    lines: list[str] = []
    all_keys = sorted(set(old) | set(new))
    for key in all_keys:
        old_val = old.get(key)
        new_val = new.get(key)
        if old_val == new_val:
            continue
        if key not in old:
            tag = "  + (added field)"
        elif key not in new:
            tag = "  - (removed field)"
        else:
            tag = "  ~ (changed)"
        if show_values:
            if key not in old:
                lines.append(f"    {tag} {key!r}: {_truncate(new_val)!r}")
            elif key not in new:
                lines.append(f"    {tag} {key!r}")
            else:
                lines.append(f"    {tag} {key!r}")
                lines.append(f"        OLD: {_truncate(old_val)!r}")
                lines.append(f"        NEW: {_truncate(new_val)!r}")
        else:
            lines.append(f"    {tag} {key!r}")
    return lines


def diff_kb(old_kb: dict, new_kb: dict, *, show_values: bool = True) -> list[str]:
    """Compute a human-readable diff and return it as a list of lines."""
    output: list[str] = []

    all_sections = sorted(set(old_kb) | set(new_kb))
    meta_keys = {"kb_version", "kb_last_audit"}

    # --- Metadata diff ---
    meta_changes: list[str] = []
    for key in sorted(meta_keys):
        old_val = old_kb.get(key)
        new_val = new_kb.get(key)
        if old_val != new_val:
            meta_changes.append(f"  {key}: {old_val!r}  →  {new_val!r}")
    if meta_changes:
        output.append("[ metadata ]")
        output.extend(meta_changes)
        output.append("")

    # --- Section diffs ---
    for section in all_sections:
        if section in meta_keys:
            continue

        old_entries = old_kb.get(section, [])
        new_entries = new_kb.get(section, [])

        if not isinstance(old_entries, list):
            old_entries = []
        if not isinstance(new_entries, list):
            new_entries = []

        old_map = {e["ticker"]: e for e in old_entries if isinstance(e, dict) and "ticker" in e}
        new_map = {e["ticker"]: e for e in new_entries if isinstance(e, dict) and "ticker" in e}

        added = sorted(set(new_map) - set(old_map))
        removed = sorted(set(old_map) - set(new_map))
        common = sorted(set(old_map) & set(new_map))
        modified = [t for t in common if old_map[t] != new_map[t]]

        if not added and not removed and not modified:
            continue

        output.append(f"[ {section} ]")

        for ticker in added:
            output.append(f"  + ADDED    {ticker}")
            if show_values:
                entry = new_map[ticker]
                for k, v in entry.items():
                    if k != "ticker":
                        output.append(f"      {k}: {_truncate(v)!r}")

        for ticker in removed:
            output.append(f"  - REMOVED  {ticker}")

        for ticker in modified:
            output.append(f"  ~ MODIFIED {ticker}")
            field_lines = _diff_entry(old_map[ticker], new_map[ticker], show_values=show_values)
            output.extend(field_lines)

        output.append("")

    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show a human-readable diff between two market_knowledge.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("old", type=Path, help="Old (baseline) KB file.")
    parser.add_argument("new", type=Path, help="New (updated) KB file.")
    parser.add_argument(
        "--no-values",
        action="store_true",
        dest="no_values",
        help="Show changed field names only, not old/new text.",
    )
    args = parser.parse_args(argv)

    for path in (args.old, args.new):
        if not path.exists():
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            return 1

    try:
        old_kb = json.loads(args.old.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[ERROR] {args.old}: {exc}", file=sys.stderr)
        return 1

    try:
        new_kb = json.loads(args.new.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[ERROR] {args.new}: {exc}", file=sys.stderr)
        return 1

    lines = diff_kb(old_kb, new_kb, show_values=not args.no_values)

    if not lines:
        print("No differences found.")
        return 0

    print(f"--- {args.old}")
    print(f"+++ {args.new}")
    print()
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
