"""Unit tests for scripts/kb_audit.py and scripts/kb_diff.py — Phase 4a/4b.

Tests are fully offline — no API calls, no filesystem side-effects.
All KB fixtures are constructed in-memory or written to tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.kb_audit import Issue, audit_kb, audit_macro
from scripts.kb_audit import main as audit_main
from scripts.kb_diff import diff_kb
from scripts.kb_diff import main as diff_main

# ---------------------------------------------------------------------------
# Minimal valid KB fixture helpers
# ---------------------------------------------------------------------------

_VALID_ENTRY = {
    "ticker": "FRO",
    "company": "Frontline PLC",
    "business_model": "VLCC crude tanker operator.",
    "primary_sensitivity": "Tanker day rates → direct P&L impact.",
    "constraints": "Does not benefit from pipeline disruptions.",
    "last_verified": "2026-03-25",
}

_VALID_KB = {
    "kb_version": "1.0.0",
    "kb_last_audit": "2026-03-25",
    "shipping_data": [_VALID_ENTRY],
}

_VALID_MACRO = {
    "date": "April 2026",
    "situations": [
        {
            "theme": "Middle East Tension",
            "detail": "Strait of Hormuz risk elevated.",
            "physical_effect": "VLCC day rates elevated.",
        }
    ],
}


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# audit_kb — valid cases
# ---------------------------------------------------------------------------


class TestAuditKbValid:
    def test_valid_kb_returns_no_issues(self) -> None:
        issues = audit_kb(_VALID_KB)
        assert issues == []

    def test_multiple_sections_all_valid(self) -> None:
        kb = {
            "kb_version": "1.0.0",
            "kb_last_audit": "2026-03-25",
            "shipping_data": [_VALID_ENTRY],
            "refinery_data": [
                {**_VALID_ENTRY, "ticker": "PBF", "company": "PBF Energy"},
            ],
        }
        issues = audit_kb(kb)
        assert issues == []


# ---------------------------------------------------------------------------
# audit_kb — versioning metadata
# ---------------------------------------------------------------------------


class TestAuditKbVersioning:
    def test_missing_kb_version_is_error(self) -> None:
        kb = {k: v for k, v in _VALID_KB.items() if k != "kb_version"}
        issues = audit_kb(kb)
        errors = [i for i in issues if i.severity == Issue.ERROR]
        assert any("kb_version" in i.message for i in errors)

    def test_invalid_kb_version_format_is_error(self) -> None:
        kb = {**_VALID_KB, "kb_version": "v1"}
        issues = audit_kb(kb)
        errors = [i for i in issues if i.severity == Issue.ERROR]
        assert any("kb_version" in i.message for i in errors)

    def test_missing_kb_last_audit_is_warning(self) -> None:
        kb = {k: v for k, v in _VALID_KB.items() if k != "kb_last_audit"}
        issues = audit_kb(kb)
        warnings = [i for i in issues if i.severity == Issue.WARN]
        assert any("kb_last_audit" in i.message for i in warnings)

    def test_invalid_kb_last_audit_format_is_error(self) -> None:
        kb = {**_VALID_KB, "kb_last_audit": "March 2026"}
        issues = audit_kb(kb)
        errors = [i for i in issues if i.severity == Issue.ERROR]
        assert any("kb_last_audit" in i.message for i in errors)


# ---------------------------------------------------------------------------
# audit_kb — required fields
# ---------------------------------------------------------------------------


class TestAuditKbRequiredFields:
    @pytest.mark.parametrize(
        "missing_field",
        ["ticker", "company", "business_model", "primary_sensitivity", "constraints"],
    )
    def test_missing_required_field_is_error(self, missing_field: str) -> None:
        entry = {k: v for k, v in _VALID_ENTRY.items() if k != missing_field}
        kb = {**_VALID_KB, "shipping_data": [entry]}
        issues = audit_kb(kb)
        errors = [i for i in issues if i.severity == Issue.ERROR]
        assert any(missing_field in i.message for i in errors)

    def test_empty_required_field_is_error(self) -> None:
        entry = {**_VALID_ENTRY, "business_model": "   "}
        kb = {**_VALID_KB, "shipping_data": [entry]}
        issues = audit_kb(kb)
        errors = [i for i in issues if i.severity == Issue.ERROR]
        assert any("business_model" in i.message for i in errors)

    def test_missing_last_verified_is_warning_by_default(self) -> None:
        entry = {k: v for k, v in _VALID_ENTRY.items() if k != "last_verified"}
        kb = {**_VALID_KB, "shipping_data": [entry]}
        issues = audit_kb(kb, strict=False)
        warnings = [i for i in issues if i.severity == Issue.WARN]
        assert any("last_verified" in i.message for i in warnings)

    def test_missing_last_verified_is_error_in_strict_mode(self) -> None:
        entry = {k: v for k, v in _VALID_ENTRY.items() if k != "last_verified"}
        kb = {**_VALID_KB, "shipping_data": [entry]}
        issues = audit_kb(kb, strict=True)
        errors = [i for i in issues if i.severity == Issue.ERROR]
        assert any("last_verified" in i.message for i in errors)

    def test_invalid_last_verified_format_is_error(self) -> None:
        entry = {**_VALID_ENTRY, "last_verified": "March 2026"}
        kb = {**_VALID_KB, "shipping_data": [entry]}
        issues = audit_kb(kb)
        errors = [i for i in issues if i.severity == Issue.ERROR]
        assert any("last_verified" in i.message for i in errors)

    def test_duplicate_ticker_in_section_is_error(self) -> None:
        kb = {**_VALID_KB, "shipping_data": [_VALID_ENTRY, _VALID_ENTRY]}
        issues = audit_kb(kb)
        errors = [i for i in issues if i.severity == Issue.ERROR]
        assert any("duplicate" in i.message.lower() for i in errors)


# ---------------------------------------------------------------------------
# audit_macro
# ---------------------------------------------------------------------------


class TestAuditMacro:
    def test_valid_macro_returns_no_issues(self) -> None:
        issues = audit_macro(_VALID_MACRO)
        assert issues == []

    def test_missing_date_is_error(self) -> None:
        macro = {k: v for k, v in _VALID_MACRO.items() if k != "date"}
        issues = audit_macro(macro)
        errors = [i for i in issues if i.severity == Issue.ERROR]
        assert any("date" in i.message for i in errors)

    def test_unparseable_date_is_error(self) -> None:
        macro = {**_VALID_MACRO, "date": "not a date"}
        issues = audit_macro(macro)
        errors = [i for i in issues if i.severity == Issue.ERROR]
        assert any("date" in i.message for i in errors)

    def test_stale_macro_is_warning(self) -> None:
        macro = {**_VALID_MACRO, "date": "January 2020"}
        issues = audit_macro(macro)
        warnings = [i for i in issues if i.severity == Issue.WARN]
        assert any("days old" in i.message for i in warnings)

    def test_missing_situations_is_error(self) -> None:
        macro = {k: v for k, v in _VALID_MACRO.items() if k != "situations"}
        issues = audit_macro(macro)
        errors = [i for i in issues if i.severity == Issue.ERROR]
        assert any("situations" in i.message for i in errors)

    def test_empty_situations_list_is_warning(self) -> None:
        macro = {**_VALID_MACRO, "situations": []}
        issues = audit_macro(macro)
        warnings = [i for i in issues if i.severity == Issue.WARN]
        assert any("empty" in i.message.lower() for i in warnings)

    def test_situation_missing_field_is_error(self) -> None:
        sit = {k: v for k, v in _VALID_MACRO["situations"][0].items() if k != "physical_effect"}
        macro = {**_VALID_MACRO, "situations": [sit]}
        issues = audit_macro(macro)
        errors = [i for i in issues if i.severity == Issue.ERROR]
        assert any("physical_effect" in i.message for i in errors)

    def test_iso_date_format_also_accepted(self) -> None:
        macro = {**_VALID_MACRO, "date": "2026-03-25"}
        issues = audit_macro(macro)
        assert issues == []


# ---------------------------------------------------------------------------
# audit_main CLI (exit codes)
# ---------------------------------------------------------------------------


class TestAuditMainCli:
    def test_clean_kb_exits_zero(self, tmp_path: Path) -> None:
        kb_path = tmp_path / "kb.json"
        macro_path = tmp_path / "macro.json"
        _write_json(kb_path, _VALID_KB)
        _write_json(macro_path, _VALID_MACRO)
        code = audit_main(["--kb", str(kb_path), "--macro", str(macro_path)])
        assert code == 0

    def test_invalid_kb_exits_nonzero(self, tmp_path: Path) -> None:
        kb_path = tmp_path / "kb.json"
        bad_kb = {k: v for k, v in _VALID_KB.items() if k != "kb_version"}
        _write_json(kb_path, bad_kb)
        code = audit_main(["--kb", str(kb_path), "--no-macro"])
        assert code != 0

    def test_stale_macro_exits_nonzero_with_strict(self, tmp_path: Path) -> None:
        kb_path = tmp_path / "kb.json"
        macro_path = tmp_path / "macro.json"
        _write_json(kb_path, _VALID_KB)
        stale_macro = {**_VALID_MACRO, "date": "January 2020"}
        _write_json(macro_path, stale_macro)
        code = audit_main(["--kb", str(kb_path), "--macro", str(macro_path), "--strict"])
        assert code != 0

    def test_stale_macro_exits_zero_without_strict(self, tmp_path: Path) -> None:
        kb_path = tmp_path / "kb.json"
        macro_path = tmp_path / "macro.json"
        _write_json(kb_path, _VALID_KB)
        stale_macro = {**_VALID_MACRO, "date": "January 2020"}
        _write_json(macro_path, stale_macro)
        code = audit_main(["--kb", str(kb_path), "--macro", str(macro_path)])
        assert code == 0

    def test_missing_kb_file_returns_1(self, tmp_path: Path) -> None:
        code = audit_main(["--kb", str(tmp_path / "nonexistent.json"), "--no-macro"])
        assert code == 1


# ---------------------------------------------------------------------------
# diff_kb logic
# ---------------------------------------------------------------------------


class TestDiffKb:
    def test_identical_kbs_produce_no_diff(self) -> None:
        lines = diff_kb(_VALID_KB, _VALID_KB)
        assert lines == []

    def test_added_ticker_shown(self) -> None:
        new_entry = {**_VALID_ENTRY, "ticker": "TNK", "company": "Teekay Tankers"}
        new_kb = {**_VALID_KB, "shipping_data": [_VALID_ENTRY, new_entry]}
        lines = diff_kb(_VALID_KB, new_kb)
        assert any("ADDED" in line and "TNK" in line for line in lines)

    def test_removed_ticker_shown(self) -> None:
        old_kb = {
            **_VALID_KB,
            "shipping_data": [_VALID_ENTRY, {**_VALID_ENTRY, "ticker": "TNK", "company": "Teekay"}],
        }
        lines = diff_kb(old_kb, _VALID_KB)
        assert any("REMOVED" in line and "TNK" in line for line in lines)

    def test_modified_field_shown(self) -> None:
        new_entry = {**_VALID_ENTRY, "business_model": "Updated model."}
        new_kb = {**_VALID_KB, "shipping_data": [new_entry]}
        lines = diff_kb(_VALID_KB, new_kb, show_values=True)
        assert any("MODIFIED" in line and "FRO" in line for line in lines)
        assert any("business_model" in line for line in lines)

    def test_version_change_shown_in_metadata(self) -> None:
        new_kb = {**_VALID_KB, "kb_version": "1.1.0"}
        lines = diff_kb(_VALID_KB, new_kb)
        assert any("kb_version" in line for line in lines)

    def test_new_section_added(self) -> None:
        new_entry = {**_VALID_ENTRY, "ticker": "LNG", "company": "Cheniere Energy"}
        new_kb = {**_VALID_KB, "lng_data": [new_entry]}
        lines = diff_kb(_VALID_KB, new_kb)
        assert any("lng_data" in line for line in lines)


# ---------------------------------------------------------------------------
# diff_main CLI
# ---------------------------------------------------------------------------


class TestDiffMainCli:
    def test_identical_files_prints_no_differences(self, tmp_path: Path, capsys) -> None:
        path = tmp_path / "kb.json"
        _write_json(path, _VALID_KB)
        code = diff_main([str(path), str(path)])
        captured = capsys.readouterr()
        assert code == 0
        assert "No differences" in captured.out

    def test_diff_shows_added_ticker(self, tmp_path: Path, capsys) -> None:
        old_path = tmp_path / "old.json"
        new_path = tmp_path / "new.json"
        _write_json(old_path, _VALID_KB)
        new_kb = {
            **_VALID_KB,
            "shipping_data": [_VALID_ENTRY, {**_VALID_ENTRY, "ticker": "TNK", "company": "T"}],
        }
        _write_json(new_path, new_kb)
        code = diff_main([str(old_path), str(new_path)])
        captured = capsys.readouterr()
        assert code == 0
        assert "ADDED" in captured.out
        assert "TNK" in captured.out

    def test_missing_file_returns_1(self, tmp_path: Path) -> None:
        path = tmp_path / "kb.json"
        _write_json(path, _VALID_KB)
        code = diff_main([str(path), str(tmp_path / "nonexistent.json")])
        assert code == 1
