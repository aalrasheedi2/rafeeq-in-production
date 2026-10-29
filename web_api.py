# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=3.30", "pydantic>=2.12", "python-dotenv>=1.1", "pandas>=2.2", "mcp>=2.3", "httpx>=0.28",
#                 "fastapi>=0.150", "uvicorn>=0.40"]
# ///
"""Rafeeq (our rafeeq.py, configuration A) behind FastAPI, with a small web page and real human approval.

    RAFEEQ_WEB_TOKEN=choose-a-long-random-string uv run capstone/web_api.py      # then open http://127.0.0.1:8800

- The agent is rafeeq.make_agent(configs A): our spec (AGENTS.md), tools (local + MCP), permissions, budgets,
  redaction and audit. It does not change rafeeq.py, so the eval fingerprints of A and B stay the same.
- A write tool pauses the run on the server; the page shows the action (permit and slot only) with Approve / Reject,
  and the officer's click resumes it. No answer within APPROVAL_TIMEOUT seconds counts as a rejection.
- The OpenRouter key stays in labs/.env on the server; the browser never sees it. Every endpoint needs the
  X-Rafeeq-Token header; the server listens on 127.0.0.1 only.
- Conversations live in memory only (never on disk); answers, trace lines and approval prompts are redacted.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import secrets
import sys
import threading
import time
import uuid
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(LABS), str(Path(__file__).resolve().parent)]

from fastapi import Depends, FastAPI, Header, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import agentkit as K  # noqa: E402
import madar  # noqa: E402
import rafeeq  # noqa: E402

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
APPROVAL_TIMEOUT = 600


class ChatIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=4000)


class DecisionIn(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")


HYPHENS = str.maketrans({c: "-" for c in "‐‑‒–—−"})


def display(text: str) -> str:
    """Redacted text for the page. Models sometimes write "INS‑9002" with a Unicode hyphen, which rafeeq.redact_text
    would not recognise as a Madar ID; normalising hyphens first keeps IDs readable and masks exactly the same data."""
    return rafeeq.redact_text(text.translate(HYPHENS))


def trace_lines(run: K.Run) -> list[str]:
    """Redacted one-liners for each tool call of a finished run (rafeeq.trace_line, as on the CLI)."""
    lines, pending = [], None
    for e in run.trace:
        if e.kind == "call":
            pending = e.data
        elif e.kind in ("result", "error") and pending:
            lines.append(rafeeq.trace_line(pending["name"], pending["args"], e.text))
            pending = None
        elif e.kind == "guard":
            lines.append(f"🛡️ {e.text}")
    return lines


def create_app(mcp, workdir: Path | None = None, token: str | None = None) -> FastAPI:
    token = token or os.getenv("RAFEEQ_WEB_TOKEN", "")
    if len(token) < 12:
        raise SystemExit("Set RAFEEQ_WEB_TOKEN to a random string of 12+ characters.")
    m = madar.Madar(workdir or rafeeq.OUT / "db", actor="rafeeq-web")  # fresh copy of the training data
    sessions: dict[str, list] = {}  # in memory only
    jobs: dict[str, dict] = {}
    approvals: dict[str, dict] = {}
    lock = threading.Lock()  # one agent run at a time (agentkit's LEDGER is process-wide)
    app = FastAPI(title="Rafeeq web", version="1.0.0")

    def auth(x_rafeeq_token: str = Header(default="")) -> None:
        if not secrets.compare_digest(x_rafeeq_token, token):
            raise HTTPException(401, "Missing or invalid X-Rafeeq-Token")

    def work(job: dict, message: str, session_id: str) -> None:
        def approve(name: str, args: dict) -> bool:
            a = {"id": uuid.uuid4().hex, "summary": rafeeq.approval_summary(name, args), "decision": None,
                 "event": threading.Event()}
            approvals[a["id"]] = a
            job.update(status="awaiting_approval", approval={"id": a["id"], "summary": a["summary"]})
            a["event"].wait(APPROVAL_TIMEOUT)
            ok = a["decision"] == "approve"
            job["decisions"].append({"summary": a["summary"], "decision": "approved" if ok else "rejected"})
            job.update(status="running", approval=None)
            approvals.pop(a["id"], None)
            return ok

        try:
            with lock:
                history = rafeeq.compact(sessions.get(session_id, []))
                agent = rafeeq.make_agent(rafeeq.configs(("A",))["A"], mcp, approve=approve)
                run = agent(message, m, history=history, use_cache=False)
            sessions[session_id] = run.state[1:]
            job.update(status="done", final=display(run.final or f"(stopped: {run.stopped})"),
                       trace=trace_lines(run), steps=run.steps, stopped=run.stopped)
        except Exception as e:  # noqa: BLE001 - the page shows the kind of failure, never secrets
            job.update(status="error", error=rafeeq.redact_text(f"{type(e).__name__}: {str(e)[:200]}"))

    @app.get("/")
    def page() -> FileResponse:
        return FileResponse(PAGE)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "model": K.MODEL, "mcp_tools": list(rafeeq.MCP_TOOLS)}

    @app.post("/api/chat", dependencies=[Depends(auth)])
    def chat(req: ChatIn) -> dict:
        job = {"id": uuid.uuid4().hex, "status": "running", "approval": None, "decisions": [], "final": "",
               "trace": [], "steps": 0, "stopped": "", "error": "", "started": time.time()}
        jobs[job["id"]] = job
        threading.Thread(target=work, args=(job, req.message, req.session_id), daemon=True).start()
        return {"job_id": job["id"]}

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(auth)])
    def job_status(job_id: str) -> dict:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "No such job")
        return {k: job[k] for k in ("id", "status", "approval", "decisions", "final", "trace", "steps", "stopped",
                                     "error")}

    @app.post("/api/approvals/{approval_id}", dependencies=[Depends(auth)])
    def decide(approval_id: str, req: DecisionIn) -> dict:
        a = approvals.get(approval_id)
        if not a or a["event"].is_set():
            raise HTTPException(404, "No pending approval with this id")
        a["decision"] = req.decision
        a["event"].set()
        return {"ok": True, "decision": req.decision}

    return app


def main() -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8800)
    args = ap.parse_args()
    with K.MCPBridge(rafeeq.mcp_server()) as mcp:
        app = create_app(mcp)
        print(f"Rafeeq web on http://127.0.0.1:{args.port}  (model {K.MODEL}; enter RAFEEQ_WEB_TOKEN in the page)")
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        sys.exit(main())
