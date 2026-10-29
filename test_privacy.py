"""Offline checks that verification data never reaches session files, the audit log or the approval prompt.

Uses Madar's synthetic applicants (Noura, FTR-30311) in a temporary copy of the database. No model is called, and no
write tool runs.

    uv run pytest capstone/test_privacy.py -v
"""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

import agentkit as K  # noqa: E402
import madar  # noqa: E402
import rafeeq  # noqa: E402

PERMIT, LAST4, MOBILE = "FTR-30311", "6654", "0532236654"  # synthetic test applicant from labs/README.md
SECRETS = (LAST4, MOBILE, MOBILE[1:], "+966" + MOBILE[1:])


def assert_no_secrets(text: str) -> None:
    for s in SECRETS:
        assert s not in text, "verification data written to disk"


# ---------------------------------------------------------------- a scripted model, no network
class Msg:
    def __init__(self, content="", calls=()):
        self.content = content
        self.tool_calls = [SimpleNamespace(id=f"c{i}", type="function", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
                           for i, (n, a) in enumerate(calls)] or None

    def model_dump(self, **_):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}}
                               for c in self.tool_calls]
        return d


@pytest.fixture
def model(monkeypatch):
    """Replace agentkit.chat with a script of replies; any real provider call fails the test."""
    script: list[Msg] = []

    def fake_chat(messages, **kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=script.pop(0))], usage=None)

    def no_network(*a, **kw):
        raise AssertionError("the test tried to reach the model provider")

    monkeypatch.setattr(K, "chat", fake_chat)
    monkeypatch.setattr(K, "client", no_network)
    monkeypatch.setattr(K, "anthropic_client", no_network)
    return script


@pytest.fixture
def audit_file(monkeypatch, tmp_path):
    path = tmp_path / "audit.jsonl"
    monkeypatch.setattr(rafeeq, "AUDIT", path)
    return path


class FakeMCPServer:
    """Stands in for servers/madar_mcp.py: all four read tools, on the server's own database copy."""

    def __init__(self, workdir: Path):
        self.m = madar.Madar(workdir)

    def agent_tools(self):
        return [getattr(self.m, n) for n in madar.Madar.READ_TOOLS]


# ---------------------------------------------------------------- redaction
@pytest.mark.parametrize("text", [
    f"My ID ends {LAST4}, mobile {MOBILE}.",
    f"mobile +966 {MOBILE[1:3]} {MOBILE[3:6]} {MOBILE[6:]} and national ID last 4 digits: {LAST4}",
    f"رقم الجوال {MOBILE} وآخر 4 أرقام من الهوية {LAST4}",
    "رقم الجوال ٠٥٣٢٢٣٦٦٥٤ والهوية ٦٦٥٤",
    f"Verify with {LAST4} · {MOBILE}",
    "national ID 1023456654 and IBAN SA03 8000 0000 6080 1016 7519",
])
def test_redact_text_masks_verification_data(text):
    out = rafeeq.redact_text(text)
    assert_no_secrets(out)
    for s in ("٠٥٣٢٢٣٦٦٥٤", "٦٦٥٤", "1023456654", "6080 1016"):
        assert s not in out


def test_redact_text_keeps_operational_details():
    text = "Fee for FTR-30311 is 1,500 SAR (POL-02); inspection on 2026-11-04 09:00, 2 working days ahead."
    assert rafeeq.redact_text(text) == text


# ---------------------------------------------------------------- Madar record IDs vs. verification data
@pytest.mark.parametrize("text", [
    "Booked. Inspection ID: INS-9002 on 4 November 2026 at 09:00.",       # the live-demo answer that was over-redacted
    "Refund ID REF-5001 created (1,280 SAR, needs a finance officer).",
    "Escalation ID ESC-0001 sent to the duty supervisor.",
    "Escalation id: ESC-301, reply within 2 working days.",
    "Outbox message ID OUT-0001 queued.",
    "Your ID was verified. Inspection INS-9002 is booked for FTR-30311.",
    "رقم الهوية تم التحقق منه، ورقم الحجز INS-9002",
    "Permit ID BLD-20417, inspection ID INS-9001, refund ID REF-5002.",
])
def test_madar_record_ids_stay_readable(text):
    assert rafeeq.redact_text(text) == text


@pytest.mark.parametrize("text, kept, gone", [
    (f"ID ends {LAST4}; booking INS-9002.", ["INS-9002"], [LAST4]),
    (f"Inspection ID: INS-9002. My ID: {LAST4}", ["INS-9002"], [LAST4]),
    (f"INS-9002 · {MOBILE}", ["INS-9002"], [MOBILE]),
    (f"{LAST4} · {MOBILE} for REF-5001", ["REF-5001"], [LAST4, MOBILE]),
    (f"ID INS {LAST4}", [], [LAST4]),            # no hyphen: not a Madar ID
    (f"ID XYZ-{LAST4}", [], [LAST4]),           # unknown prefix
    (f"ID ins-{LAST4}", [], [LAST4]),           # Madar IDs are upper case
    (f"الهوية {LAST4} والحجز INS-9002", ["INS-9002"], [LAST4]),
])
def test_verification_data_is_still_masked_next_to_madar_ids(text, kept, gone):
    out = rafeeq.redact_text(text)
    for k in kept:
        assert k in out
    for g in gone:
        assert g not in out


# fake credentials, assembled at run time so no key-shaped string sits in the repository
FAKE = "fakefakefake" + "0123456789"
SK_OR, HF = "sk-" + "or-v1-", "hf" + "_"


@pytest.mark.parametrize("text, secret", [
    (f"key {SK_OR}{FAKE}", FAKE),
    (f"OPENROUTER_API_KEY={SK_OR}{FAKE}", FAKE),
    ("api_key: 'plainvalue123'", "plainvalue123"),
    (f"Authorization: Bearer {FAKE}.payload.sig", FAKE),
    (f"{HF}{FAKE.upper()} is my token", FAKE.upper()),
    ("password=hunter2hunter2", "hunter2hunter2"),
])
def test_credentials_are_masked(text, secret):
    assert secret not in rafeeq.redact_text(text)


def test_sensitive_keys_are_masked_whatever_their_case():
    args = {"Mobile": MOBILE, "NATIONAL_ID_LAST4": LAST4, "api_key": "x" * 20, "inspection_id": "INS-9002",
            "refund_id": "REF-5001", "escalation_id": "ESC-0001"}
    out = rafeeq.redact(args)
    assert out["Mobile"] == out["NATIONAL_ID_LAST4"] == out["api_key"] == rafeeq.REDACTED
    assert (out["inspection_id"], out["refund_id"], out["escalation_id"]) == ("INS-9002", "REF-5001", "ESC-0001")


def test_session_keeps_the_booking_id_but_not_verification(tmp_path):
    path = tmp_path / "session.json"
    rafeeq.save_session(path, [
        {"role": "user", "content": f"Book {PERMIT}. ID ends {LAST4}, mobile {MOBILE}."},
        {"role": "assistant", "content": "Booked. Inspection ID: INS-9002 on 4 November 2026 at 09:00."},
    ])
    saved = path.read_text(encoding="utf-8")
    assert "INS-9002" in saved
    assert_no_secrets(saved)


def test_redact_drops_sensitive_keys():
    args = {"permit_id": PERMIT, "slot": "2026-11-04 09:00", "national_id_last4": LAST4, "mobile": MOBILE}
    assert rafeeq.redact(args) == {"permit_id": PERMIT, "slot": "2026-11-04 09:00",
                                   "national_id_last4": rafeeq.REDACTED, "mobile": rafeeq.REDACTED}


# ---------------------------------------------------------------- sessions
def test_session_file_keeps_context_but_not_verification(model, audit_file, tmp_path):
    m = madar.Madar(tmp_path / "db")
    model += [Msg(calls=[("get_permit", {"permit_id": PERMIT, "national_id_last4": LAST4, "mobile": MOBILE})]),
              Msg(f"Your food truck permit {PERMIT} is approved. The health inspection in Al-Rawdah is not booked yet.")]
    agent = rafeeq.make_agent(rafeeq.CONFIGS["A"], FakeMCPServer(tmp_path / "mcp"))
    run = agent(f"Status of {PERMIT}? ID ends {LAST4}, mobile {MOBILE}", m)
    assert run.stopped == "done"
    assert any(msg.get("role") == "tool" for msg in run.state), "the tool really ran with the real values"

    path = tmp_path / "session-test.json"
    rafeeq.save_session(path, run.state[1:])
    saved = path.read_text(encoding="utf-8")

    assert_no_secrets(saved)
    applicant = m.sql("SELECT a.name_en FROM permits p JOIN applicants a USING (applicant_id) WHERE permit_id = ?",
                      (PERMIT,))[0]["name_en"]
    assert applicant not in saved  # tool results (the permit file) are not kept
    history = json.loads(saved)
    assert [h["role"] for h in history] == ["user", "assistant"]
    assert PERMIT in history[0]["content"] and "approved" in history[1]["content"]  # non-sensitive context kept


def test_old_unsafe_session_is_sanitised_on_load(tmp_path):
    path = tmp_path / "session-old.json"
    path.write_text(json.dumps([
        {"role": "user", "content": f"Book for {PERMIT}, ID ends {LAST4}, mobile {MOBILE}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function", "function": {
            "name": "book_inspection", "arguments": json.dumps({"national_id_last4": LAST4, "mobile": MOBILE})}}]},
        {"role": "tool", "tool_call_id": "c0", "content": json.dumps({"mobile": MOBILE})},
        {"role": "assistant", "content": "Please confirm the slot."},
    ]), encoding="utf-8")
    history = rafeeq.load_session(path)
    assert_no_secrets(json.dumps(history, ensure_ascii=False))
    assert [h["role"] for h in history] == ["user", "assistant"]
    assert PERMIT in history[0]["content"]


# ---------------------------------------------------------------- audit log
def test_audit_records_tool_time_and_outcome_without_verification(model, audit_file, tmp_path):
    m = madar.Madar(tmp_path / "db")
    model += [Msg(calls=[("get_permit", {"permit_id": PERMIT, "national_id_last4": LAST4, "mobile": MOBILE}),
                         ("get_permit", {"permit_id": PERMIT, "national_id_last4": "0000", "mobile": "0500000000"})]),
              Msg("done")]
    rafeeq.make_agent(rafeeq.CONFIGS["A"], FakeMCPServer(tmp_path / "mcp"))("status please", m)

    text = audit_file.read_text(encoding="utf-8")
    assert_no_secrets(text)
    assert "0500000000" not in text
    ok, failed = [json.loads(line) for line in text.splitlines()]
    assert ok["event"] == "tool_call" and ok["tool"] == "get_permit" and ok["ts"]
    assert ok["outcome"] == "ok" and ok["result"]["permit_id"] == PERMIT and ok["result"]["status"]
    assert ok["args"] == {"permit_id": PERMIT, "national_id_last4": rafeeq.REDACTED, "mobile": rafeeq.REDACTED}
    assert set(ok["result"]) <= set(rafeeq.AUDIT_RESULT_FIELDS)
    assert failed["outcome"] == "error" and "verification failed" in failed["error"]


# ---------------------------------------------------------------- human approval
@pytest.mark.parametrize("tool", sorted(rafeeq.WRITE_TOOLS))
def test_write_tools_still_need_approval(tool):
    assert rafeeq.permission(tool, {}) == "ask"


@pytest.mark.parametrize("tool", madar.Madar.READ_TOOLS)
def test_read_tools_still_run_without_approval(tool):
    assert rafeeq.permission(tool, {}) == "allow"


def test_denied_write_does_not_run_and_is_logged(model, audit_file, tmp_path, monkeypatch):
    m = madar.Madar(tmp_path / "db")
    before = m.sql("SELECT count(*) AS n FROM inspections")[0]["n"]
    prompts = []
    monkeypatch.setattr(builtins, "input", lambda p="": prompts.append(p) or "n")  # the officer says no
    model += [Msg(calls=[("book_inspection", {"permit_id": PERMIT, "slot": "2026-11-04 09:00",
                                              "national_id_last4": LAST4, "mobile": MOBILE})]),
              Msg("I could not book it; an officer will follow up.")]
    rafeeq.make_agent(rafeeq.CONFIGS["A"], FakeMCPServer(tmp_path / "mcp"))("book it", m)

    assert len(prompts) == 1 and "book_inspection" in prompts[0] and PERMIT in prompts[0]
    assert_no_secrets(prompts[0])  # the officer sees the action, not the verification values
    assert m.sql("SELECT count(*) AS n FROM inspections")[0]["n"] == before  # nothing was written
    events = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events] == ["approval"]
    assert events[0]["decision"] == "denied" and events[0]["tool"] == "book_inspection"
    assert_no_secrets(json.dumps(events, ensure_ascii=False))


# ---------------------------------------------------------------- context still loads
def test_agents_md_still_reaches_the_model(monkeypatch, tmp_path, audit_file):
    seen = []
    monkeypatch.setattr(K, "chat", lambda messages, **kw: seen.append(messages[0]) or
                        SimpleNamespace(choices=[SimpleNamespace(message=Msg("ok"))], usage=None))
    m = madar.Madar(tmp_path / "db")
    rafeeq.make_agent(rafeeq.CONFIGS["A"], FakeMCPServer(tmp_path / "mcp"))("hi", m)
    assert seen[0]["content"] == rafeeq.SPEC
    assert rafeeq.AGENTS_MD.read_text(encoding="utf-8").strip() in seen[0]["content"]
