"""Offline rehearsal of the official `book-inspection` task: read freely, write only after a human says yes.

A scripted model replaces OpenRouter, and every database is a temporary copy of Madar's synthetic data. The fake MCP
server has its own copy, like servers/madar_mcp.py. This proves the harness, the tool wiring and the database checks,
not that the real model will choose these calls: only the live run shows that.

    uv run pytest capstone/test_booking.py -v
"""

from __future__ import annotations

import builtins
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

import agentkit as K  # noqa: E402
import evals  # noqa: E402
import madar  # noqa: E402
import rafeeq  # noqa: E402
from test_privacy import (LAST4, MOBILE, PERMIT, FakeMCPServer, Msg, assert_no_secrets,  # noqa: E402, F401
                          audit_file, model)

TASK = next(t for t in evals.load_tasks(rafeeq.LABS / "data" / "eval" / "tasks.jsonl") if t["id"] == "book-inspection")
SLOT = TASK["graders"][0]["expect"]  # "2026-11-04 09:00"
VERIFY = {"national_id_last4": LAST4, "mobile": MOBILE}
GET_PERMIT = ("get_permit", {"permit_id": PERMIT, **VERIFY})
LIST_SLOTS = ("list_slots", {"district": "Al-Rawdah", "date_from": "2026-11-03", "date_to": "2026-11-04"})
BOOK = ("book_inspection", {"permit_id": PERMIT, "slot": SLOT, **VERIFY})
POLICY = ("search_policies", {"query": "inspection booking rules"})


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def demo_script() -> list[Msg]:
    """What a good model does for this task: verify, find a morning slot, book it, confirm."""
    return [Msg(calls=[GET_PERMIT]), Msg(calls=[LIST_SLOTS]), Msg(calls=[BOOK]),
            Msg(f"Booked: health inspection for {PERMIT} on 4 November 2026 at 09:00 (POL-05).")]


@pytest.fixture
def officer(monkeypatch):
    """The human at the approval prompt. Set .answer to 'y' or 'n'; .prompts records what they were shown."""
    o = SimpleNamespace(answer="n", prompts=[])
    monkeypatch.setattr(builtins, "input", lambda p="": o.prompts.append(p) or o.answer)
    return o


@pytest.fixture
def db(tmp_path):
    return madar.Madar(tmp_path / "db")


@pytest.fixture
def server(tmp_path):
    return FakeMCPServer(tmp_path / "mcp")


def run(model, db, server, script, show=False, **kw):
    model += script
    return rafeeq.make_agent(rafeeq.CONFIGS["A"], server, show=show)(TASK["prompt"], db, **kw)


def booked(m: madar.Madar) -> list[dict]:
    return m.sql("SELECT inspection_id, slot, status FROM inspections WHERE permit_id = ?", (PERMIT,))


def slot_taken(m: madar.Madar) -> int:
    return m.sql("SELECT taken FROM slots WHERE district = 'Al-Rawdah' AND slot = ?", (SLOT,))[0]["taken"]


def tool_results(r: K.Run) -> list[str]:
    return [msg["content"] for msg in r.state if msg.get("role") == "tool"]


# ---------------------------------------------------------------- the task and the data
def test_task_expects_the_only_morning_slot():
    """'before 12:00' on 3 or 4 November leaves exactly one free slot in Al-Rawdah: the grader's answer."""
    m = madar.Madar()  # the training data, read only
    free = m.sql("SELECT slot FROM slots WHERE district = 'Al-Rawdah' AND taken = 0 "
                 "AND substr(slot,1,10) IN ('2026-11-03','2026-11-04') AND substr(slot,12) < '12:00'")
    assert [r["slot"] for r in free] == [SLOT]
    assert m.sql("SELECT status, district FROM permits WHERE permit_id = ?", (PERMIT,)) == \
        [{"status": "approved", "district": "Al-Rawdah"}]
    assert booked(m) == []


# ---------------------------------------------------------------- tool wiring
def test_tools_are_unique_and_stateful_ones_use_the_run_database(monkeypatch, db, server, audit_file):
    offered = []
    monkeypatch.setattr(K, "chat", lambda messages, **kw: offered.append(kw["tools"]) or
                        SimpleNamespace(choices=[SimpleNamespace(message=Msg("ok"))], usage=None))
    rafeeq.make_agent(rafeeq.CONFIGS["A"], server)("hi", db)

    tools = {t.__name__: t for t in offered[0]}
    assert len(tools) == len(offered[0]) == 8  # no name is offered twice
    assert set(tools) == {"get_permit", "list_slots", "book_inspection", "reschedule_inspection", "create_refund",
                          "escalate", "search_policies", "compute_fee"}
    for name in ("get_permit", "list_slots", "book_inspection", "reschedule_inspection", "create_refund", "escalate"):
        assert tools[name].__self__ is db and rafeeq.tool_source(name) == "local"
    for name in rafeeq.MCP_TOOLS:
        assert tools[name].__self__ is server.m and rafeeq.tool_source(name) == "mcp"


def test_each_eval_trial_gets_its_own_database(monkeypatch, server, tmp_path, audit_file):
    offered = []
    monkeypatch.setattr(K, "chat", lambda messages, **kw: offered.append(kw["tools"]) or
                        SimpleNamespace(choices=[SimpleNamespace(message=Msg("ok"))], usage=None))
    agent = rafeeq.make_agent(rafeeq.CONFIGS["A"], server)
    first, second = madar.Madar(tmp_path / "t1"), madar.Madar(tmp_path / "t2")
    agent("a", first)
    agent("b", second)
    by_name = [{t.__name__: t for t in tools} for tools in offered]
    assert by_name[0]["get_permit"].__self__ is first and by_name[1]["get_permit"].__self__ is second


# ---------------------------------------------------------------- reads, denial, approval
def test_reads_run_without_asking_anyone(model, db, server, audit_file, officer):
    run(model, db, server, [Msg(calls=[GET_PERMIT]), Msg(calls=[LIST_SLOTS]), Msg("4 Nov 09:00 is free. Book it?")])

    assert officer.prompts == []
    events = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines()]
    assert [(e["tool"], e["source"], e["outcome"]) for e in events] == [("get_permit", "local", "ok"),
                                                                         ("list_slots", "local", "ok")]
    assert booked(db) == []


def test_denied_booking_writes_nothing(model, db, server, audit_file, officer):
    tables = ("inspections", "slots", "refunds", "escalations", "outbox")
    before = {t: db.sql(f"SELECT * FROM {t}") for t in tables}
    officer.answer = "n"
    r = run(model, db, server, demo_script())

    assert len(officer.prompts) == 1 and "book_inspection" in officer.prompts[0]
    assert {t: db.sql(f"SELECT * FROM {t}") for t in tables} == before  # no permit data changed
    assert booked(db) == [] and slot_taken(db) == 0
    assert db.sql("SELECT count(*) AS n FROM audit_log WHERE action = 'book_inspection'") == [{"n": 0}]
    assert not evals.grade(TASK, r.final, r.tool_calls, db)[0]["pass"]
    events = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines()]
    assert [e.get("decision") for e in events if e["event"] == "approval"] == ["denied"]
    assert "book_inspection" not in [e["tool"] for e in events if e["event"] == "tool_call"]


def test_approved_booking_lands_in_the_run_database_only(model, db, server, audit_file, officer):
    training = rafeeq.LABS / "data" / "madar.sqlite"
    before = fingerprint(training)
    officer.answer = "y"
    r = run(model, db, server, demo_script())

    assert r.stopped == "done" and len(officer.prompts) == 1
    assert evals.grade(TASK, r.final, r.tool_calls, db) == [{"type": "db", "pass": True, "detail": ""}]
    [row] = booked(db)
    assert row["slot"] == SLOT and row["status"] == "booked" and row["inspection_id"].startswith("INS-")
    assert slot_taken(db) == 1
    assert db.sql("SELECT actor FROM audit_log WHERE action = 'book_inspection'") == [{"actor": "rafeeq"}]
    assert fingerprint(training) == before and booked(madar.Madar()) == []  # training data never written
    assert booked(server.m) == []  # the MCP server's copy is not the record of bookings


def test_booking_is_visible_when_asked_again_in_the_same_session(model, db, server, audit_file, officer):
    officer.answer = "y"
    r = run(model, db, server, [Msg(calls=[GET_PERMIT]), Msg(calls=[LIST_SLOTS]), Msg(calls=[BOOK]),
                                Msg(calls=[GET_PERMIT]), Msg(calls=[LIST_SLOTS]), Msg("Booked and confirmed.")])

    first_permit, first_slots, booking, again_permit, again_slots = tool_results(r)
    assert json.loads(first_permit)["inspections"] == [] and SLOT in json.loads(first_slots)
    inspection_id = json.loads(booking)["inspection_id"]
    assert {"inspection_id": inspection_id, "kind": "health", "slot": SLOT, "status": "booked"} in \
        json.loads(again_permit)["inspections"]
    assert SLOT not in json.loads(again_slots)


# ---------------------------------------------------------------- what is shown and stored
def test_screen_shows_a_redacted_trace_and_never_the_raw_one(model, db, server, audit_file, officer, capsys, tmp_path):
    officer.answer = "y"
    r = run(model, db, server, [Msg(calls=[GET_PERMIT]), Msg(calls=[POLICY]), Msg(calls=[LIST_SLOTS]),
                                Msg(calls=[BOOK]), Msg(f"Booked {SLOT}. Your mobile {MOBILE} will get an SMS.")],
            show=True, verbose=True)  # verbose=True must not switch agentkit's raw printer back on
    screen = capsys.readouterr().out

    assert_no_secrets(screen)
    for line in ("🔧 [local] get_permit", "🔧 [mcp] search_policies", "🔧 [local] list_slots",
                 "🔧 [local] book_inspection", "🛡️  approved: book_inspection"):
        assert line in screen
    prompt = officer.prompts[0]
    assert prompt.strip() == f"🛡️  Allow book_inspection(permit_id={PERMIT}, slot={SLOT})? [y/N]"
    assert_no_secrets(prompt)

    path = tmp_path / "session.json"
    rafeeq.save_session(path, r.state[1:])
    assert_no_secrets(path.read_text(encoding="utf-8"))
    audit_text = audit_file.read_text(encoding="utf-8")
    assert_no_secrets(audit_text)
    assert [json.loads(line)["source"] for line in audit_text.splitlines()] == ["local", "mcp", "local", "local",
                                                                                 "local"]
