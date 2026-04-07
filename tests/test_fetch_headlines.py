"""Tests for scripts/fetch_headlines.py — GDELT headline sourcing."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests as _requests_lib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fetch_headlines import (
    _parse_gdelt_date,
    fetch_headlines,
    main,
    write_csv,
)

# ---------------------------------------------------------------------------
# _parse_gdelt_date
# ---------------------------------------------------------------------------


class TestParseGdeltDate:
    def test_valid_format(self):
        result = _parse_gdelt_date("20260115T143200Z")
        assert result is not None
        assert result.startswith("2026-01-15T14:32:00")

    def test_invalid_format_returns_none(self):
        assert _parse_gdelt_date("not-a-date") is None

    def test_none_input_returns_none(self):
        assert _parse_gdelt_date(None) is None


# ---------------------------------------------------------------------------
# write_csv
# ---------------------------------------------------------------------------


class TestWriteCsv:
    def test_writes_correct_csv(self, tmp_path):
        rows = [
            {"datetime": "2026-01-01T10:00:00+00:00", "headline": "Test headline", "feed_source": "reuters.com"},
        ]
        p = tmp_path / "out.csv"
        n = write_csv(rows, p)
        assert n == 1
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "datetime,headline,feed_source" in content
        assert "Test headline" in content

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "subdir" / "deep" / "out.csv"
        write_csv([{"datetime": "x", "headline": "y", "feed_source": "z"}], p)
        assert p.exists()


# ---------------------------------------------------------------------------
# fetch_headlines (mocked HTTP)
# ---------------------------------------------------------------------------

_SAMPLE_GDELT_RESPONSE = {
    "articles": [
        {
            "url": "https://reuters.com/article1",
            "title": "OPEC agrees to deep production cuts amid demand fears",
            "seendate": "20260301T120000Z",
            "domain": "reuters.com",
        },
        {
            "url": "https://bloomberg.com/article2",
            "title": "Tanker rates surge as Red Sea attacks intensify",
            "seendate": "20260302T080000Z",
            "domain": "bloomberg.com",
        },
        {
            "url": "https://reuters.com/article3",
            "title": "OPEC agrees to deep production cuts amid demand fears",  # duplicate title
            "seendate": "20260301T130000Z",
            "domain": "reuters.com",
        },
    ]
}


class TestFetchHeadlines:
    def test_deduplicates_titles(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _SAMPLE_GDELT_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch("scripts.fetch_headlines.requests.get", return_value=mock_resp):
            rows = fetch_headlines(
                queries=["test query"],
                start_date="2026-01-01",
                end_date="2026-04-01",
                limit=100,
            )

        # 3 articles but one is a duplicate title → 2 unique
        assert len(rows) == 2

    def test_respects_limit(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _SAMPLE_GDELT_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch("scripts.fetch_headlines.requests.get", return_value=mock_resp):
            rows = fetch_headlines(
                queries=["test query"],
                start_date="2026-01-01",
                end_date="2026-04-01",
                limit=1,
            )

        assert len(rows) <= 1

    def test_sorted_chronologically(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _SAMPLE_GDELT_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        with patch("scripts.fetch_headlines.requests.get", return_value=mock_resp):
            rows = fetch_headlines(
                queries=["test query"],
                start_date="2026-01-01",
                end_date="2026-04-01",
            )

        dates = [r["datetime"] for r in rows]
        assert dates == sorted(dates)

    def test_http_error_returns_empty_list(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = _requests_lib.RequestException("HTTP 500")

        with patch("scripts.fetch_headlines.requests.get", return_value=mock_resp):
            rows = fetch_headlines(
                queries=["failing query"],
                start_date="2026-01-01",
                end_date="2026-04-01",
            )

        assert rows == []


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMain:
    def test_writes_output_file(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _SAMPLE_GDELT_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        out = tmp_path / "headlines.csv"
        with patch("scripts.fetch_headlines.requests.get", return_value=mock_resp):
            rc = main(["-o", str(out), "--from", "2026-01-01", "--to", "2026-04-01"])

        assert rc == 0
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "OPEC" in content

    def test_returns_1_when_no_results(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"articles": []}
        mock_resp.raise_for_status = MagicMock()

        out = tmp_path / "empty.csv"
        with patch("scripts.fetch_headlines.requests.get", return_value=mock_resp):
            rc = main(["-o", str(out), "--query", "nonexistent query xyz"])

        assert rc == 1
