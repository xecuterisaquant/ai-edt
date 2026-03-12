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
    "refinery_data": [
        {
            "ticker": "PBF",
            "company": "PBF Energy",
            "edge": "High coking capacity; ideal for Heavy Sour crude.",
            "strategy": "Primary beneficiary of increased heavy crude supply.",
        }
    ],
    "logistics_data": [
        {
            "ticker": "FRO",
            "company": "Frontline PLC",
            "edge": "85% spot market exposure.",
            "strategy": "Long on geopolitical disruption.",
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

MALFORMED_SIGNAL_RESPONSE = """\
The market situation is complex. I believe uncertainty is high and no
clear directional trade can be established at this time.
"""

SIEVE_YES_RESPONSE = "YES"
SIEVE_NO_RESPONSE = "NO"
SIEVE_AMBIGUOUS_RESPONSE = "It is unclear from the headline alone."

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_mock_config(tmp_path: Path) -> MagicMock:
    """Build a MagicMock that satisfies all _Config attribute accesses."""
    kb_path = tmp_path / "market_knowledge.json"
    kb_path.write_text(json.dumps(SAMPLE_KNOWLEDGE), encoding="utf-8")

    cfg = MagicMock()
    cfg.high_alpha_keywords = ["venezuela", "sanction", "hormuz", "suez", "heavy crude"]
    cfg.general_keywords = ["oil", "tanker", "refinery", "bpd", "fleet", "vessel"]
    cfg.sieve_model = "llama3.2:1b"
    cfg.reasoning_model = "deepseek-r1:8b"
    cfg.sieve_timeout = 30
    cfg.reasoning_timeout = 120
    cfg.knowledge_base_path = kb_path
    cfg.signal_log_path = tmp_path / "signals.jsonl"
    return cfg


@pytest.fixture()
def mock_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the global config singleton for the duration of one test.

    All modules that call get_config() will receive this mock, so no
    YAML files or real file paths are needed during tests.
    """
    cfg = make_mock_config(tmp_path)
    monkeypatch.setattr(_config_mod, "_instance", cfg)
    return cfg
