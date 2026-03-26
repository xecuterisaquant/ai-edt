"""Shared pytest fixtures for AI-EDT tests.

All LLM calls are mocked — no GPU or running Ollama instance required.
The mock_config fixture patches the global config singleton so every
module that calls get_config() receives a controlled test config
pointing to temporary files in tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import ai_edt.config as _config_mod
import ai_edt.db as _db_mod
from ai_edt.config import _Config

# ---------------------------------------------------------------------------
# Sample headlines
# ---------------------------------------------------------------------------

HIGH_ALPHA_HEADLINES = [
    "US grants Chevron expanded license for Venezuelan heavy crude",
    "New sanctions hit Iranian tanker fleet amid rising tensions",
    "Suez Canal blocked by grounded VLCC — global shipping rerouting",
]

IRRELEVANT_HEADLINES = [
    "Apple reports record iPhone sales for Q4",
    "Federal Reserve holds interest rates steady",
    "Tech giants lobby for AI regulation in Europe",
]

GENERAL_KEYWORD_HEADLINES = [
    "Oil prices rise on supply concerns",
    "Tanker rates climb as demand increases",
]

# ---------------------------------------------------------------------------
# Sample knowledge base
# ---------------------------------------------------------------------------

SAMPLE_KNOWLEDGE = {
    "upstream_data": [
        {
            "ticker": "CVX",
            "company": "Chevron",
            "business_model": "Upstream crude producer, Venezuela licence holder.",
        }
    ],
    "refinery_data": [
        {
            "ticker": "PBF",
            "company": "PBF Energy",
            "business_model": "Independent refiner, high coking capacity for heavy sour crude.",
        }
    ],
    "shipping_data": [
        {
            "ticker": "FRO",
            "company": "Frontline PLC",
            "business_model": "VLCC crude tanker operator, 85% spot market exposure.",
        }
    ],
}

# ---------------------------------------------------------------------------
# Sample LLM responses
# ---------------------------------------------------------------------------

VALID_SIGNAL_RESPONSE = """\
- Ticker: FRO
- Signal: LONG
- Confidence: 88%
- Rationale: Sanctions on Iranian tankers reduce global VLCC supply, spiking \
spot rates for Frontline's exposed fleet. FRO's 85% spot exposure means \
earnings respond immediately to day-rate increases.
"""

VALID_SIGNAL_WITH_THINK = """\
<think>
Let me reason through this carefully...
Iran sanctions reduce tanker supply globally.
</think>
- Ticker: FRO
- Signal: LONG
- Confidence: 88%
- Rationale: Sanctions on Iranian tankers reduce global VLCC supply, spiking \
spot rates for Frontline's exposed fleet. FRO's 85% spot exposure means \
earnings respond immediately to day-rate increases.
"""

VALID_SHORT_SIGNAL_RESPONSE = """\
- Ticker: FRO
- Signal: SHORT
- Confidence: 80%
- Rationale: A surge in VLCC newbuild deliveries increases fleet supply, pushing \
spot day rates lower across the market. FRO's 85% spot exposure means earnings \
decline almost immediately when rates fall, making it the highest-leverage SHORT.
"""

MALFORMED_SIGNAL_RESPONSE = """\
The market situation is complex. I believe uncertainty is high and no
clear directional trade can be established at this time.
"""

VALID_TNK_SIGNAL_RESPONSE = """\
- Ticker: TNK
- Signal: LONG
- Confidence: 75%
- Rationale: Venezuelan Aframax Caribbean-to-Gulf cargo volumes increase, \
directly benefiting Teekay's regional fleet. TNK specialises in exactly the \
Aframax size-class required for Venezuela's shallow-draft export terminals.
"""

SIEVE_YES_RESPONSE = "YES"
SIEVE_NO_RESPONSE = "NO"
SIEVE_AMBIGUOUS_RESPONSE = "It is unclear from the headline alone."

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_mock_config(tmp_path: Path) -> MagicMock:
    """Build a MagicMock that satisfies all _Config attribute accesses.

    Uses ``spec=_Config`` so accessing typo'd attributes raises
    AttributeError instead of silently returning a new MagicMock.
    """
    kb_path = tmp_path / "market_knowledge.json"
    kb_path.write_text(json.dumps(SAMPLE_KNOWLEDGE), encoding="utf-8")

    cfg = MagicMock(spec=_Config)
    cfg.no_pass_keywords = [
        "ceo resigns",
        "cfo resigns",
        "analysts forecast",
        "forecast stable",
        "pay deal",
        "union votes",
        "workers accept",
        # Corporate financial calendar
        "quarterly dividend",
        "within expectations",
        "annual dividend",
        "quarterly earnings",
        "reports earnings",
        # Non-event corporate communications
        "sustainability report",
        "annual report",
        "investor day",
    ]
    cfg.high_alpha_keywords = ["venezuela", "sanction", "hormuz", "suez", "heavy crude"]
    cfg.general_keywords = ["oil", "tanker", "refinery", "bpd", "fleet", "vessel"]
    cfg.sieve_model = "llama3.2:1b"
    cfg.reasoning_model = "deepseek-r1:8b"
    cfg.reasoning_provider = "ollama"  # tests always route through mocked ollama
    cfg.gemini_model = "gemini-2.0-flash"
    cfg.sieve_timeout = 30
    cfg.reasoning_timeout = 120
    cfg.knowledge_base_path = kb_path
    cfg.signal_log_path = tmp_path / "signals.jsonl"
    cfg.db_path = tmp_path / "signals.db"
    cfg.min_confidence = 0  # tests log all signals regardless of confidence
    cfg.keep_jsonl_backup = False
    cfg.market_hours_only = False
    cfg.max_entries_per_feed = 50
    return cfg


@pytest.fixture()
def mock_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the global config singleton for the duration of one test.

    Also initializes a fresh SQLite database and patches the db singleton so
    all signal writes and duplicate checks go to an isolated per-test database.
    Both singletons are automatically restored after the test by monkeypatch.
    """
    cfg = make_mock_config(tmp_path)
    monkeypatch.setattr(_config_mod, "_instance", cfg)
    # Fresh DB for every test — isolates signal state between tests.
    conn = _db_mod.init_db(tmp_path / "signals.db")
    monkeypatch.setattr(_db_mod, "_conn", conn)
    return cfg
