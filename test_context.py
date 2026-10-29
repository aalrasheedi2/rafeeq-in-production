"""Offline checks that capstone/AGENTS.md reaches the agent's context. No model is called.

    uv run pytest capstone/test_context.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

import agentkit as K  # noqa: E402
import madar  # noqa: E402
import rafeeq  # noqa: E402

AGENTS_TEXT = rafeeq.AGENTS_MD.read_text(encoding="utf-8").strip()


class FakeMessage:
    content = "ok"
    tool_calls = None

    def model_dump(self, **_):
        return {"role": "assistant", "content": self.content}


class NoMCP:
    """The stateless MCP tools only, from the read-only training data (they never touch the database)."""

    def agent_tools(self):
        m = madar.Madar()
        return [m.search_policies, m.compute_fee]


@pytest.fixture
def sent(monkeypatch):
    """Replace agentkit.chat with a fake that records the messages instead of calling OpenRouter."""
    calls = []

    def fake_chat(messages, **kw):
        calls.append([dict(m) for m in messages])
        return SimpleNamespace(choices=[SimpleNamespace(message=FakeMessage())], usage=None)

    def no_network(*a, **kw):
        raise AssertionError("the test tried to reach the model provider")

    monkeypatch.setattr(K, "chat", fake_chat)
    monkeypatch.setattr(K, "client", no_network)
    monkeypatch.setattr(K, "anthropic_client", no_network)
    return calls


def test_spec_keeps_madar_prompt_then_adds_agents_md():
    assert rafeeq.SPEC.startswith(madar.SYSTEM_PROMPT)
    assert rafeeq.SPEC.endswith(AGENTS_TEXT)
    assert rafeeq.SPEC.count(madar.SYSTEM_PROMPT) == 1


def test_both_eval_configs_use_the_spec():
    for name, cfg in rafeeq.CONFIGS.items():
        assert cfg["system"].startswith(rafeeq.SPEC), name


@pytest.mark.parametrize("rule", [
    "get_permit",                      # identity before disclosure
    "POL-08",
    "compute_fee",                     # grounded fees
    "data, not instructions",          # untrusted data
    "explicit \"yes\"",                # consent before writes
    "long-term memory",                # no personal data in memory
])
def test_agents_md_contains_required_rules(rule):
    assert rule in AGENTS_TEXT


def test_agents_md_is_the_system_message_the_model_receives(sent, tmp_path):
    agent = rafeeq.make_agent(rafeeq.CONFIGS["A"], NoMCP())
    run = agent("What is the fee for an illuminated 3 m² sign?", madar.Madar(tmp_path / "db"))

    assert run.stopped == "done" and run.final == "ok"
    assert len(sent) == 1
    system = sent[0][0]
    assert system["role"] == "system"
    assert system["content"] == rafeeq.SPEC
    assert AGENTS_TEXT in system["content"]
    assert sent[0][-1] == {"role": "user", "content": "What is the fee for an illuminated 3 m² sign?"}


def test_compaction_keeps_the_system_prompt(sent, tmp_path):
    history = []
    for i in range(30):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    agent = rafeeq.make_agent(rafeeq.CONFIGS["A"], NoMCP())
    agent("next", madar.Madar(tmp_path / "db"), history=rafeeq.compact(history))

    messages = sent[0]
    assert messages[0]["content"] == rafeeq.SPEC
    assert len(messages) <= rafeeq.MAX_HISTORY_MESSAGES + 2
