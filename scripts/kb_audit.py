"""Phase 4a — KB Auditor: validates market_knowledge.json and macro_context.json.

Checks for:
  - Root-level versioning metadata (kb_version, kb_last_audit)
  - Required fields on every ticker entry
  - Missing last_verified timestamps
  - Stale macro_context scenarios (> 30 days based on the date field)

Exit codes:
  0 — no issues found
  1 — one or more issues found (each printed to stdout)

Usage:
    python -m scripts.kb_audit                         # uses paths from settings.yaml
    python -m scripts.kb_audit --kb path/to/kb.json
    python -m scripts.kb_audit --macro path/to/macro.json
    python -m scripts.kb_audit --strict                # warn-as-error for last_verified
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Required fields per section type
# ---------------------------------------------------------------------------

# Every ticker entry in every section must have these.
_BASE_REQUIRED = frozenset(
    {
        "ticker",
        "company",
        "business_model",
        "primary_sensitivity",
        "constraints",
    }
)

# Fields whose absence is an error only in --strict mode (else a warning).
_OPTIONAL_BUT_AUDITED = frozenset({"last_verified"})

# Sections that are expected in a fully-expanded KB.  Sections not yet added
# (e.g. lng_data) are silently ignored — auditor only validates what is present.
_KNOWN_SECTIONS = frozenset(
    {
        "upstream_data",
        "integrated_majors_data",
        "trading_houses_data",
        "midstream_data",
        "refinery_data",
        "shipping_data",
        "broader_shipping_data",
        "airline_data",
        "agriculture_data",
        "petrochemical_data",
        "lng_data",
        "oilfield_services_data",
        "power_utilities_data",
    }
)

# Non-ticker sections that exist in the KB but are NOT lists of ticker profiles.
# The auditor skips ticker-level field validation for these sections.
_NON_TICKER_SECTIONS = frozenset(
    {
        "cross_sector_rules",
    }
)

# Macro scenario date field must be parseable and not older than this many days.
_MACRO_STALE_DAYS = 30

# Regex patterns for root versioning fields
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")  # e.g. "1.0.0"
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # e.g. "2026-03-25"

# ---------------------------------------------------------------------------
# Issue helpers
# ---------------------------------------------------------------------------


class Issue:
    """A single audit finding."""

    ERROR = "ERROR"
    WARN = "WARN"

    def __init__(self, severity: str, message: str) -> None:
        self.severity = severity
        self.message = message

    def __str__(self) -> str:
        return f"  [{self.severity}] {self.message}"


def _err(msg: str) -> Issue:
    return Issue(Issue.ERROR, msg)


def _warn(msg: str) -> Issue:
    return Issue(Issue.WARN, msg)


# ---------------------------------------------------------------------------
# KB auditing
# ---------------------------------------------------------------------------


def audit_kb(kb: dict, *, strict: bool = False) -> list[Issue]:
    """Return a list of Issues found in *kb* (market_knowledge.json contents)."""
    issues: list[Issue] = []

    # --- Root versioning ---
    if "kb_version" not in kb:
        issues.append(_err("Missing root field 'kb_version' (e.g. \"1.0.0\")"))
    elif not _VERSION_RE.match(str(kb["kb_version"])):
        issues.append(_err(f"'kb_version' must match MAJOR.MINOR.PATCH, got: {kb['kb_version']!r}"))

    if "kb_last_audit" not in kb:
        issues.append(_warn("Missing root field 'kb_last_audit' (e.g. \"2026-03-25\")"))
    elif not _DATE_RE.match(str(kb["kb_last_audit"])):
        issues.append(_err(f"'kb_last_audit' must be YYYY-MM-DD, got: {kb['kb_last_audit']!r}"))
    else:
        # Warn if the KB hasn't been audited in more than _MACRO_STALE_DAYS days.
        try:
            from datetime import date as _date

            audit_date = _date.fromisoformat(str(kb["kb_last_audit"]))
            age_days = (_date.today() - audit_date).days
            if age_days > _MACRO_STALE_DAYS:
                issues.append(
                    _warn(
                        f"'kb_last_audit' is {age_days} days old (>{_MACRO_STALE_DAYS}). "
                        "Run a full audit and bump kb_version if you edited ticker profiles."
                    )
                )
        except ValueError:
            pass  # already caught by the regex check above)

    # --- Ticker entries ---
    for section, entries in kb.items():
        if section in ("kb_version", "kb_last_audit"):
            continue
        # Non-ticker sections (e.g. cross_sector_rules) have a different schema —
        # skip ticker-level field validation for them entirely.
        if section in _NON_TICKER_SECTIONS:
            # Validate cross_sector_rules entries have required structural fields.
            if not isinstance(entries, list):
                issues.append(
                    _err(f"Section '{section}': expected a list, got {type(entries).__name__}")
                )
                continue
            for i, rule in enumerate(entries):
                if not isinstance(rule, dict):
                    issues.append(_err(f"{section}[{i}]: expected a dict, got {type(rule).__name__}"))
                    continue
                if "rule_id" not in rule:
                    issues.append(_err(f"{section}[{i}]: missing required field 'rule_id'"))
                if "last_updated" not in rule:
                    issues.append(_warn(f"{section}[{i}]: missing 'last_updated' field — add it when you edit this rule"))
                elif not _DATE_RE.match(str(rule["last_updated"])):
                    issues.append(_err(f"{section}[{i}]: 'last_updated' must be YYYY-MM-DD, got: {rule['last_updated']!r}"))
            continue
        if not isinstance(entries, list):
            issues.append(
                _err(f"Section '{section}': expected a list, got {type(entries).__name__}")
            )
            continue

        seen_tickers: set[str] = set()
        for i, entry in enumerate(entries):
            loc = f"{section}[{i}]"
            if not isinstance(entry, dict):
                issues.append(_err(f"{loc}: expected a dict, got {type(entry).__name__}"))
                continue

            ticker = entry.get("ticker", f"<entry {i}>")
            loc = f"{section}/{ticker}"

            # Duplicate ticker check
            if ticker in seen_tickers:
                issues.append(_err(f"{loc}: duplicate ticker '{ticker}' in section"))
            seen_tickers.add(ticker)

            # Required base fields
            for field in sorted(_BASE_REQUIRED):
                if field not in entry:
                    issues.append(_err(f"{loc}: missing required field '{field}'"))
                elif not str(entry[field]).strip():
                    issues.append(_err(f"{loc}: field '{field}' is empty"))

            # last_verified — warn or error depending on --strict
            if "last_verified" not in entry:
                fn = _err if strict else _warn
                issues.append(fn(f"{loc}: missing 'last_verified' timestamp (e.g. \"2026-03-25\")"))
            elif not _DATE_RE.match(str(entry["last_verified"])):
                issues.append(
                    _err(
                        f"{loc}: 'last_verified' must be YYYY-MM-DD, "
                        f"got: {entry['last_verified']!r}"
                    )
                )

    return issues


# ---------------------------------------------------------------------------
# Macro context auditing
# ---------------------------------------------------------------------------


def _parse_macro_date(date_str: str) -> datetime | None:
    """Try to parse 'Month YYYY' or 'YYYY-MM-DD' into a UTC datetime."""
    # Try ISO date first
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        pass
    # Try "March 2026" style
    for fmt in ("%B %Y", "%b %Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt).replace(tzinfo=UTC)
        except ValueError:
            pass
    return None


def audit_macro(macro: dict) -> list[Issue]:
    """Return a list of Issues found in *macro* (macro_context.json contents)."""
    issues: list[Issue] = []

    if "date" not in macro:
        issues.append(_err("macro_context.json missing 'date' field"))
    else:
        macro_dt = _parse_macro_date(str(macro["date"]))
        if macro_dt is None:
            issues.append(
                _err(
                    f"macro_context.json 'date' could not be parsed: {macro['date']!r}. "
                    "Use 'Month YYYY' or 'YYYY-MM-DD' format."
                )
            )
        else:
            age_days = (datetime.now(UTC) - macro_dt).days
            if age_days > _MACRO_STALE_DAYS:
                issues.append(
                    _warn(
                        f"macro_context.json is {age_days} days old "
                        f"(threshold: {_MACRO_STALE_DAYS}). "
                        "Run: python -m scripts.update_macro_context --dry-run"
                    )
                )

    if "situations" not in macro:
        issues.append(_err("macro_context.json missing 'situations' list"))
    elif not isinstance(macro["situations"], list):
        issues.append(_err("macro_context.json 'situations' must be a list"))
    elif len(macro["situations"]) == 0:
        issues.append(_warn("macro_context.json 'situations' list is empty"))
    else:
        for i, sit in enumerate(macro["situations"]):
            loc = f"macro_context.situations[{i}]"
            for field in ("theme", "detail", "physical_effect"):
                if field not in sit:
                    issues.append(_err(f"{loc}: missing required field '{field}'"))
                elif not str(sit[field]).strip():
                    issues.append(_err(f"{loc}: field '{field}' is empty"))

    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit AI-EDT knowledge base files for schema and freshness issues.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kb", type=Path, help="Path to market_knowledge.json.")
    parser.add_argument("--macro", type=Path, help="Path to macro_context.json.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings (e.g. missing last_verified) as errors.",
    )
    parser.add_argument(
        "--no-macro",
        action="store_true",
        dest="no_macro",
        help="Skip macro_context.json audit.",
    )
    args = parser.parse_args(argv)

    # Resolve paths: CLI args → settings.yaml defaults → hardcoded fallback
    kb_path = args.kb
    macro_path = args.macro

    if kb_path is None or (macro_path is None and not args.no_macro):
        try:
            from ai_edt.config import get_config

            cfg = get_config()
            kb_path = kb_path or cfg.knowledge_base_path
            if not args.no_macro:
                macro_path = macro_path or cfg.macro_context_path
        except Exception:  # noqa: BLE001
            kb_path = kb_path or PROJECT_ROOT / "data" / "market_knowledge.json"
            if not args.no_macro:
                macro_path = macro_path or PROJECT_ROOT / "data" / "macro_context.json"

    all_issues: list[Issue] = []
    section_header_printed: set[str] = set()

    def _print_section(header: str) -> None:
        if header not in section_header_printed:
            print(f"\n{header}")
            section_header_printed.add(header)

    # --- Audit KB ---
    if not kb_path.exists():
        print(f"[ERROR] KB file not found: {kb_path}", file=sys.stderr)
        return 1

    try:
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[ERROR] KB file is not valid JSON: {exc}", file=sys.stderr)
        return 1

    kb_issues = audit_kb(kb, strict=args.strict)
    if kb_issues:
        _print_section(f"market_knowledge.json  ({kb_path})")
        for iss in kb_issues:
            print(iss)
        all_issues.extend(kb_issues)
    else:
        # Print per-section summary
        section_count = sum(1 for k in kb if k not in ("kb_version", "kb_last_audit"))
        ticker_count = sum(
            len(v)
            for k, v in kb.items()
            if k not in ("kb_version", "kb_last_audit") and isinstance(v, list)
        )
        print(
            f"market_knowledge.json  OK  "
            f"({section_count} sections, {ticker_count} tickers, "
            f"version {kb.get('kb_version', '?')})"
        )

    # --- Audit macro ---
    if not args.no_macro and macro_path is not None:
        if not macro_path.exists():
            print(f"[WARN] macro_context.json not found: {macro_path}")
            all_issues.append(_warn(f"macro_context.json not found at {macro_path}"))
        else:
            try:
                macro = json.loads(macro_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"[ERROR] macro_context.json is not valid JSON: {exc}", file=sys.stderr)
                return 1

            macro_issues = audit_macro(macro)
            if macro_issues:
                _print_section(f"macro_context.json  ({macro_path})")
                for iss in macro_issues:
                    print(iss)
                all_issues.extend(macro_issues)
            else:
                sit_count = len(macro.get("situations", []))
                print(
                    f"macro_context.json     OK  ({sit_count} situations, date: {macro.get('date', '?')})"
                )

    # --- Summary ---
    errors = [i for i in all_issues if i.severity == Issue.ERROR]
    warnings = [i for i in all_issues if i.severity == Issue.WARN]

    print()
    if not all_issues:
        print("Audit passed — no issues found.")
        return 0

    summary_parts = []
    if errors:
        summary_parts.append(f"{len(errors)} error(s)")
    if warnings:
        summary_parts.append(f"{len(warnings)} warning(s)")
    print(f"Audit complete: {', '.join(summary_parts)}.")

    return 1 if errors or (args.strict and warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
