"""Offline checks of the web server: token, chat, tool trace, real approve/reject on the server, no key or
verification data in responses. A scripted model replaces OpenRouter.

    uv run pytest capstone/test_web_api.py -v
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import agentkit as K  # noqa: E402
import rafeeq  # noqa: E402
import web_api as W  # noqa: E402
from test_eval_runner import FAKE_KEY, FakeProvider, reply  # noqa: E402, F401
from test_privacy import LAST4, MOBILE, FakeMCPServer, Msg  # noqa: E402

TOKEN = "test-token-123456"
SLOT = "2026-11-04 09:00"
BOOK_PROMPT = f"Book the health inspection for FTR-30311 on {SLOT}. ID ends {LAST4}, mobile {MOBILE}."


@pytest.fixture
def client(tmp_path, monkeypatch):
    provider = FakeProvider()
    monkeypatch.setattr(K, "chat", provider)
    monkeypatch.setattr(K, "client", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("network")))
    monkeypatch.setattr(rafeeq, "AUDIT", tmp_path / "audit.jsonl")
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    app = W.create_app(FakeMCPServer(tmp_path / "mcp"), workdir=tmp_path / "db", token=TOKEN)
    c = TestClient(app)
    c.provider, c.db = provider, tmp_path / "db" / "madar.sqlite"
    c.headers.update({"X-Rafeeq-Token": TOKEN})
    return c


def wait(c: TestClient, job_id: str, until=("done", "error"), timeout=10) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in until:
            return j
        time.sleep(0.05)
    raise AssertionError(f"job stuck: {j}")


def booked(db: Path) -> list:
    import sqlite3

    con = sqlite3.connect(db)
    try:
        return con.execute("SELECT slot, status FROM inspections WHERE permit_id = 'FTR-30311'").fetchall()
    finally:
        con.close()


def test_every_api_endpoint_needs_the_token(client):
    for method, path, body in (("post", "/api/chat", {"session_id": "s", "message": "hi"}),
                               ("get", "/api/jobs/x", None), ("post", "/api/approvals/x", {"decision": "approve"})):
        r = getattr(client, method)(path, json=body, headers={"X-Rafeeq-Token": "wrong"}) if body else \
            getattr(client, method)(path, headers={"X-Rafeeq-Token": "wrong"})
        assert r.status_code == 401
    assert client.get("/healthz").json()["mcp_tools"] == ["search_policies", "compute_fee"]
    assert "Rafeeq" in client.get("/").text


def test_a_short_token_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        W.create_app(FakeMCPServer(tmp_path / "mcp"), workdir=tmp_path / "db", token="short")


def test_fee_question_runs_the_mcp_tool_and_answers(client):
    prompt = "What is the fee for an illuminated 3 m2 sign?"
    client.provider.agent[prompt] = [Msg(calls=[("compute_fee", {"permit_type": "SGN", "sign_m2": 3, "illuminated": True})]),
                                     Msg("The fee is 750 SAR (POL-02).")]
    job = client.post("/api/chat", json={"session_id": "s1", "message": prompt}).json()["job_id"]
    j = wait(client, job)
    assert j["status"] == "done" and "750 SAR" in j["final"]
    assert any(line.startswith("🔧 [mcp] compute_fee") and '"fee_sar": 750' in line for line in j["trace"])


def book_script(c: TestClient) -> None:
    c.provider.agent[BOOK_PROMPT] = [
        Msg(calls=[("book_inspection", {"permit_id": "FTR-30311", "slot": SLOT, "national_id_last4": LAST4,
                                        "mobile": MOBILE})]),
        Msg("Done: INS-9002 on 4 November at 09:00. Your mobile 0532236654 will get an SMS.")]


@pytest.mark.parametrize("decision, rows", [("approve", [(SLOT, "booked")]), ("reject", [])])
def test_the_officer_decision_on_the_server_controls_the_write(client, decision, rows):
    book_script(client)
    job = client.post("/api/chat", json={"session_id": "s2", "message": BOOK_PROMPT}).json()["job_id"]
    j = wait(client, job, until=("awaiting_approval",))
    assert j["approval"]["summary"] == f"book_inspection(permit_id=FTR-30311, slot={SLOT})"
    assert booked(client.db) == []  # nothing is written while the officer has not decided

    approval_id = j["approval"]["id"]
    assert client.post(f"/api/approvals/{approval_id}", json={"decision": decision}).status_code == 200
    j = wait(client, job)
    assert booked(client.db) == rows
    assert j["decisions"][0]["decision"] == ("approved" if decision == "approve" else "rejected")
    assert client.post(f"/api/approvals/{approval_id}", json={"decision": "approve"}).status_code == 404  # used once


def test_responses_never_carry_the_key_or_verification_data(client):
    book_script(client)
    job = client.post("/api/chat", json={"session_id": "s3", "message": BOOK_PROMPT}).json()["job_id"]
    j = wait(client, job, until=("awaiting_approval",))
    texts = [json.dumps(j, ensure_ascii=False)]
    client.post(f"/api/approvals/{j['approval']['id']}", json={"decision": "approve"})
    texts.append(json.dumps(wait(client, job), ensure_ascii=False))
    for t in texts:
        for secret in (FAKE_KEY, MOBILE, MOBILE[1:], f"ends {LAST4}"):
            assert secret not in t


def test_unicode_hyphens_keep_madar_ids_readable_and_still_mask_verification():
    out = W.display(f"Inspection ID: INS‑9002. ID ends {LAST4}, mobile 053‑2236654")
    assert "INS-9002" in out and LAST4 not in out and "2236654" not in out


def test_an_invalid_decision_is_rejected(client):
    assert client.post("/api/approvals/anything", json={"decision": "maybe"}).status_code == 422
