# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=3.30", "pydantic>=2.12", "python-dotenv>=1.1", "pandas>=2.2", "mcp>=2.3", "httpx>=0.28"]
# ///
"""Batch runner for the official eval suite: resumable, metered and fair, for free-tier quotas.

    uv run capstone/eval_runner.py plan --config A --ids fee-sign-illuminated,book-inspection
    uv run capstone/eval_runner.py run  --config A --ids fee-sign-illuminated,book-inspection --max-requests 30
    uv run capstone/eval_runner.py run  --config A --ids ... --max-requests 30 --resume
    uv run capstone/eval_runner.py merge --config A
    uv run capstone/eval_runner.py judge-check       # exactly one judge request on a synthetic example

Grading is the course's own `evals.grade` on the course's task file, one grader at a time. What this adds around it:
- one CSV row per finished trial, written and fsync'ed at once; a second CSV keeps every attempt, failed ones too;
- --resume runs only the (task, trial) pairs with no finished row, so nothing is re-run or counted twice;
- provider errors (rate limit, connection), judge errors and budget stops are not task failures: they stay pending;
- a state file marks the trial in progress, so a crash or a closed window is detected on the next run;
- every model call is metered and split into agent and judge: requests, tokens, cost, requested and actual model,
  finish reason, reasoning tokens; a judge reply that is empty, cut or not {"pass": bool, "reason"} is a judge error;
- each grader's result is kept on its own (graders_json), and Rafeeq's final answer is kept redacted (final_redacted);
- fairness: a pinned model, no fallback models, no response cache, one fingerprint per configuration and code version;
- a request budget that never starts a trial it cannot finish, sized from OpenRouter's remaining free requests.

Run from labs/. Outputs go to artifacts/capstone/evals/ (git-ignored).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(LABS), str(LABS / "capstone")]

import agentkit as K  # noqa: E402
import evals  # noqa: E402
import madar  # noqa: E402
import rafeeq  # noqa: E402

PINNED_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"  # the model of the successful live demo
PINNED_JUDGE = "nvidia/nemotron-3.5-lightning:free"
# evals.judge asks for max_tokens=300. The pinned judge is a reasoning model: its hidden reasoning used all 300 and
# left an empty verdict (first A batch). Only judge calls get this larger cap; the rubric and the prompt are unchanged.
JUDGE_MAX_TOKENS = 2000
FINAL_CHARS = 1500  # how much of Rafeeq's (redacted) final answer the attempts file keeps for error analysis
EVAL_DIR = K.ARTIFACTS / "capstone" / "evals"
TASKS_FILE = evals.TASKS_FILE
APPROVAL = "auto: eval prompts state consent (as in the course's run_suite)"

FINISHED = {"passed", "failed"}  # graded completely: the only outcomes that enter the results file
STOPPING = {"provider_auth", "budget_stop"}  # no point trying the next trial
FIELDS = [
    "run_id", "run_started_utc", "row_written_utc", "config", "config_id", "code_version",
    "task_id", "category", "lang", "trial", "outcome", "passed", "error_type", "failed_graders",
    "tools", "steps", "seconds", "stop_reason", "judge_status", "graders_json", "final_redacted",
    "agent_requests", "agent_failed_requests", "agent_tokens_in", "agent_tokens_out", "agent_cost_usd",
    "agent_finish_reasons",
    "judge_requests", "judge_failed_requests", "judge_tokens_in", "judge_tokens_out", "judge_cost_usd",
    "judge_finish_reasons", "judge_reasoning_tokens",
    "cache_hits", "requested_model", "actual_models", "judge_model", "judge_actual_models", "approval",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


# ---------------------------------------------------------------- errors
class BudgetExhausted(RuntimeError):
    """Raised before a model call that would go over the request budget of this batch."""


def classify(exc: BaseException) -> str:
    """Provider and budget problems are not wrong answers: they get their own outcome and are retried later."""
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if isinstance(exc, BudgetExhausted):
        return "budget_stop"
    if name == "RateLimitError" or status == 429:
        return "provider_rate_limit"
    if name in ("AuthenticationError", "PermissionDeniedError") or status in (401, 403):
        return "provider_auth"
    if name in ("APIConnectionError", "APITimeoutError", "ConnectError", "ConnectTimeout", "ReadTimeout",
                "RemoteProtocolError") or isinstance(exc, (ConnectionError, TimeoutError)):
        return "provider_connection"
    if name in ("InternalServerError", "APIStatusError", "NotFoundError", "BadRequestError") or (status or 0) >= 500:
        return "provider_error"
    return "harness_error"  # a bug in our code or a tool: a failed trial, as in the course's run_suite


# ---------------------------------------------------------------- metering
class Meter:
    """Wraps agentkit.chat for one batch: forces the fairness policy, counts and splits every call, keeps the budget."""

    def __init__(self, budget: int | None):
        self.budget, self.used, self.phase = budget, 0, "agent"
        self.calls: dict[str, list[dict]] = {"agent": [], "judge": []}

    def new_trial(self) -> None:
        self.calls, self.phase = {"agent": [], "judge": []}, "agent"

    def remaining(self) -> int | None:
        return None if self.budget is None else self.budget - self.used

    def wrap(self, chat):
        def metered(messages, model=None, tools=None, **kw):
            if self.budget is not None and self.used >= self.budget:
                raise BudgetExhausted(f"request budget of {self.budget} reached")
            kw["use_cache"] = False  # a measurement is a fresh sample, never a replay
            kw["fallbacks"] = False  # the pinned model answers, or the call fails and the trial stays pending
            if self.phase == "judge":
                kw["max_tokens"] = JUDGE_MAX_TOKENS  # judge calls only: room for reasoning plus the JSON verdict
            self.used += 1
            hits = K.LEDGER.cache_hits
            rec = {"requested": model or K.MODEL, "actual": "", "in": 0, "out": 0, "cost": 0.0, "cached": False,
                   "ok": False, "finish": "", "reasoning": 0, "chars": 0, "verdict_ok": None}
            self.calls[self.phase].append(rec)
            resp = chat(messages, model=model, tools=tools, **kw)
            usage = getattr(resp, "usage", None)
            u = usage if isinstance(usage, dict) else (usage.model_dump() if usage is not None else {})
            choice = resp.choices[0]
            text = getattr(choice.message, "content", None) or ""
            rec.update(actual=getattr(resp, "model", "") or "", ok=True, cached=K.LEDGER.cache_hits > hits,
                       cost=float(u.get("cost") or 0), finish=getattr(choice, "finish_reason", "") or "",
                       reasoning=(u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
                       chars=len(text), **{"in": u.get("prompt_tokens") or 0, "out": u.get("completion_tokens") or 0})
            if self.phase == "judge":
                rec["verdict_ok"] = valid_verdict(text)  # only the length and validity are kept, never the text
            return resp
        return metered

    @contextlib.contextmanager
    def installed(self):
        original = K.chat
        K.chat = self.wrap(original)  # agentkit's loop and evals.judge both call agentkit.chat
        try:
            yield self
        finally:
            K.chat = original

    def metrics(self) -> dict:
        out = {}
        for phase in ("agent", "judge"):
            calls = self.calls[phase]
            out |= {f"{phase}_requests": len(calls),
                    f"{phase}_failed_requests": sum(not c["ok"] for c in calls),
                    f"{phase}_tokens_in": sum(c["in"] for c in calls),
                    f"{phase}_tokens_out": sum(c["out"] for c in calls),
                    f"{phase}_cost_usd": round(sum(c["cost"] for c in calls), 6)}
        out["cache_hits"] = sum(c["cached"] for p in self.calls.values() for c in p)
        out["actual_models"] = "|".join(sorted({c["actual"] for c in self.calls["agent"] if c["actual"]}))
        out["judge_actual_models"] = "|".join(sorted({c["actual"] for c in self.calls["judge"] if c["actual"]}))
        out["agent_finish_reasons"] = " ".join(c["finish"] or "-" for c in self.calls["agent"])
        out["judge_finish_reasons"] = " ".join(c["finish"] or "-" for c in self.calls["judge"])
        out["judge_reasoning_tokens"] = sum(c["reasoning"] for c in self.calls["judge"])
        return out


def valid_verdict(text: str) -> bool:
    """The judge's reply is usable only as a JSON object with a boolean "pass" and a "reason" (evals.judge's format).
    evals.judge would read a missing or non-boolean "pass" as a fail; here it is a judge error instead."""
    try:
        data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except ValueError:
        return False
    return isinstance(data, dict) and isinstance(data.get("pass"), bool) and "reason" in data


# ---------------------------------------------------------------- identity of a run
def git_version() -> str:
    git = shutil.which("git") or str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git" / "cmd" / "git.exe")
    try:
        head = subprocess.run([git, "-C", str(LABS), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=20).stdout.strip()
        dirty = subprocess.run([git, "-C", str(LABS), "status", "--porcelain", "--", "."], capture_output=True,
                               text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (head or "unknown") + ("-dirty" if dirty else "")


def shared_files() -> list[Path]:
    """Files that shape every configuration: agent code and rules, runner, graders, helpers, data, tools, tasks.
    Documentation (README, notes) is not here, and neither is any configuration's own file."""
    return [Path(rafeeq.__file__), rafeeq.AGENTS_MD, Path(__file__), Path(evals.__file__), Path(K.__file__),
            Path(madar.__file__), TASKS_FILE, LABS / "servers" / "madar_mcp.py", K.DATA / "madar.sqlite"]


def config_files(config: str) -> list[Path]:
    """Files only this configuration reads."""
    return [rafeeq.CONFIG_B_MD] if config == "B" else []


def fingerprint(config: str, model: str, judge_model: str) -> tuple[str, dict]:
    """Everything that can change this configuration's results: the shared files plus its own instructions and files.
    Two runs share results only if this is identical. B's file never enters A's fingerprint."""
    details = {
        "config": config, "system_sha256": sha256(rafeeq.configs((config,))[config]["system"]),
        "model": model, "judge_model": judge_model, "judge_max_tokens": JUDGE_MAX_TOKENS,
        "fallbacks": False, "response_cache": False,
        "max_steps": rafeeq.MAX_STEPS, "max_cost": rafeeq.MAX_COST, "max_result_chars": rafeeq.MAX_RESULT_CHARS,
        "mcp_tools": list(rafeeq.MCP_TOOLS), "local_tools": [f.__name__ for f in rafeeq.local_tools(madar.Madar())],
        "files": {p.name: sha256(p.read_bytes())[:16] for p in shared_files()},
        "config_files": {p.name: sha256(p.read_bytes())[:16] for p in config_files(config)},
    }
    return sha256(json.dumps(details, sort_keys=True))[:12], details


def paths(out_dir: Path, config: str, config_id: str) -> dict[str, Path]:
    stem = f"{config}_{config_id}"
    return {"results": out_dir / f"results_{stem}.csv", "attempts": out_dir / f"attempts_{stem}.csv",
            "state": out_dir / f"state_{stem}.json", "audit": out_dir / f"audit_{stem}.jsonl",
            "meta": out_dir / f"config_{stem}.json", "db": out_dir / "db" / stem}


# ---------------------------------------------------------------- CSV, written row by row
def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: rafeeq.redact_text(str(row.get(k, ""))) for k in FIELDS})
        f.flush()
        os.fsync(f.fileno())


def finished_keys(results: Path) -> set[tuple[str, int]]:
    return {(r["task_id"], int(r["trial"])) for r in read_rows(results)}


# ---------------------------------------------------------------- OpenRouter free quota (no generation request)
def free_quota(timeout: float = 20) -> dict | None:
    """Remaining free-model requests today, from GET /api/v1/key. The key goes only into the request header."""
    import httpx

    try:
        r = httpx.get("https://openrouter.ai/api/v1/key", timeout=timeout,
                      headers={"Authorization": f"Bearer {K.api_key('openrouter')}"})
        r.raise_for_status()
        d = r.json()["data"]
    except Exception as e:  # noqa: BLE001 - report the kind of failure, never the request
        print(f"quota check failed: {type(e).__name__}")
        return None
    free = d.get("free_model_daily_requests") or {}
    return {"used": free.get("used"), "limit": free.get("limit"), "remaining": free.get("remaining"),
            "is_free_tier": d.get("is_free_tier")}


# ---------------------------------------------------------------- one judge request, to check the judge setup
JUDGE_PROBE = {  # synthetic and short: the known answer to fee-sign-illuminated
    "rubric": "The reply states that the yearly fee is 750 SAR and cites POL-02.",
    "prompt": "We want an illuminated sign of 3 square metres. How much will the permit cost per year?",
    "reply": "The permit costs 750 SAR per year (POL-02).",
    "tools": ["compute_fee"],
}


def judge_check(judge_model: str = PINNED_JUDGE) -> dict:
    """Exactly one judge request (budget 1) through evals.judge, with the same policy as an eval run."""
    meter = Meter(budget=1)
    meter.new_trial()
    meter.phase = "judge"
    with meter.installed():
        try:
            verdict = evals.judge(JUDGE_PROBE["rubric"], JUDGE_PROBE["prompt"], JUDGE_PROBE["reply"],
                                  JUDGE_PROBE["tools"], judge_model)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": classify(e), "requests": meter.used}
    c = meter.calls["judge"][0]
    return {"ok": bool(c["verdict_ok"]), "pass": verdict["pass"], "reason": rafeeq.redact_text(verdict["reason"])[:200],
            "expected_pass": True, "finish_reason": c["finish"], "tokens_in": c["in"], "tokens_out": c["out"],
            "reasoning_tokens": c["reasoning"], "content_chars": c["chars"], "model": c["actual"],
            "max_tokens": JUDGE_MAX_TOKENS, "requests": meter.used}


# ---------------------------------------------------------------- one trial
def run_trial(task: dict, trial: int, agent, meter: Meter, judge_model: str, db_dir: Path) -> dict:
    meter.new_trial()
    m = madar.Madar(db_dir / task["id"] / str(trial))  # a fresh copy of the training data for every trial
    t0 = time.perf_counter()
    row = {"task_id": task["id"], "category": task["category"], "lang": task["lang"], "trial": trial,
           "outcome": "", "passed": False, "error_type": "", "failed_graders": "", "tools": "", "steps": 0,
           "stop_reason": "", "judge_status": "none", "graders_json": "[]", "final_redacted": ""}
    try:
        run = agent(task["prompt"], m, use_cache=False)
    except Exception as e:  # noqa: BLE001
        kind = classify(e)
        detail = f"{type(e).__name__}: {str(e)[:200]}"
        row.update(outcome="failed" if kind == "harness_error" else kind, error_type=kind, failed_graders=detail,
                   stop_reason="exception")
        return row | {"seconds": round(time.perf_counter() - t0, 1)} | meter.metrics()

    final = run.final or ""  # graders see the real answer; files get only the redacted copy
    row.update(tools=" ".join(n for n, _ in run.tool_calls), steps=run.steps, stop_reason=run.stopped,
               final_redacted=rafeeq.redact_text(final)[:FINAL_CHARS])
    if run.stopped != "done":
        row["error_type"] = f"stopped_{run.stopped}"  # the agent's own budget: a real failure of the trial
    grades = grade_each(task, final, list(run.tool_calls), m, judge_model, meter)
    row["graders_json"] = json.dumps(grades, ensure_ascii=False)
    judges = [g for g in grades if g["type"] == "judge"]
    row["judge_status"] = " ".join(g["status"] for g in judges) or "none"

    broken = [g for g in grades if g["status"] != "ok"]
    if broken:
        g = broken[0]
        kind = "budget_stop" if any(b["status"] == "budget_stop" for b in broken) else (
            "judge_error" if g["type"] == "judge" else "grading_error")
        row.update(outcome=kind, error_type=f"{'judge' if g['type'] == 'judge' else 'grader'}_{g['status']}",
                   failed_graders="; ".join(f"{b['type']}: {b['status']}: {b['detail']}" for b in broken)[:500])
    else:
        failed = [g for g in grades if not g["pass"]]
        row.update(passed=not failed and not row["error_type"], outcome="failed" if failed or row["error_type"] else "passed",
                   failed_graders="; ".join(f"{g['type']}: {g['detail']}" for g in failed)[:500])
    return row | {"seconds": round(time.perf_counter() - t0, 1)} | meter.metrics()


def grade_each(task: dict, final: str, calls: list, m: madar.Madar, judge_model: str, meter: Meter) -> list[dict]:
    """The course's evals.grade, one grader at a time: a broken judge cannot hide the other graders' results.
    status is ok, unparseable, truncated, budget_stop or error:<kind>; pass is None unless status is ok."""
    out = []
    meter.phase = "judge"
    for g in task["graders"]:
        before = len(meter.calls["judge"])
        try:
            [r] = evals.grade({**task, "graders": [g]}, final, calls, m, judge_model)
        except Exception as e:  # noqa: BLE001
            kind = classify(e)
            out.append({"type": g["type"], "pass": None, "status": "budget_stop" if kind == "budget_stop" else f"error:{kind}",
                        "detail": f"{type(e).__name__}: {str(e)[:150]}"})
            continue
        status = "ok"
        if g["type"] == "judge":
            made = meter.calls["judge"][before:]
            if not made or not all(c["verdict_ok"] for c in made):
                status = "truncated" if any(c["finish"] == "length" for c in made) else "unparseable"
        out.append({"type": g["type"], "pass": r["pass"] if status == "ok" else None, "status": status,
                    "detail": rafeeq.redact_text(str(r["detail"]))[:300]})
    return out


# ---------------------------------------------------------------- a batch
def run_batch(config: str, tasks: list[dict], trials: int, mcp, *, model: str = PINNED_MODEL,
              judge_model: str = PINNED_JUDGE, max_requests: int | None = None, quota: dict | None = None,
              safety_margin: int = 5, reserve: int | None = None, resume: bool = False, out_dir: Path = EVAL_DIR,
              max_consecutive_errors: int = 3, log=print) -> dict:
    """Run the pending (task, trial) pairs of one configuration. Returns a summary; rows are already on disk."""
    config_id, details = fingerprint(config, model, judge_model)
    p = paths(out_dir, config, config_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    p["meta"].write_text(json.dumps(details, indent=1, ensure_ascii=False), encoding="utf-8")

    done = finished_keys(p["results"])
    if done and not resume:
        raise SystemExit(f"{p['results'].name} already has {len(done)} finished trials: pass --resume to continue it.")

    run_id, started = uuid.uuid4().hex[:8], utc_now()
    base = {"run_id": run_id, "run_started_utc": started, "config": config, "config_id": config_id,
            "code_version": git_version(), "requested_model": model, "judge_model": judge_model, "approval": APPROVAL}

    if p["state"].exists():  # the last run stopped in the middle of a trial (crash, closed window, power)
        stale = json.loads(p["state"].read_text(encoding="utf-8"))
        log(f"⚠️  interrupted trial found: {stale.get('task_id')} #{stale.get('trial')} (started {stale.get('started_utc')})")
        append_row(p["attempts"], base | stale | {"outcome": "interrupted", "error_type": "interrupted",
                                                  "row_written_utc": utc_now()})
        p["state"].unlink()

    budget = max_requests
    if quota and quota.get("remaining") is not None:
        allowed = max(0, int(quota["remaining"]) - safety_margin)
        budget = allowed if budget is None else min(budget, allowed)
    if budget is None:
        raise SystemExit("No request budget: give --max-requests, or let the quota check run.")
    reserve = reserve if reserve is not None else rafeeq.MAX_STEPS + 1  # worst case: every step plus one judge call

    pending = [(t, k) for t in tasks for k in range(trials) if (t["id"], k) not in done]
    log(f"config {config} [{config_id}] · {len(pending)} pending of {len(tasks) * trials} · budget {budget} requests")

    meter = Meter(budget)
    consent = lambda name, a: True  # noqa: E731 - unattended; db graders check what was written
    agent = rafeeq.make_agent(dict(rafeeq.configs((config,))[config], model=model), mcp, approve=consent)
    summary = {"config": config, "config_id": config_id, "run_id": run_id, "finished": 0, "pending": [],
               "stopped": "", "requests_used": 0, "results": p["results"], "attempts": p["attempts"]}
    errors_in_a_row = 0
    saved_audit, rafeeq.AUDIT = rafeeq.AUDIT, p["audit"]  # eval tool calls get their own audit log
    try:
        with meter.installed():
            for i, (task, k) in enumerate(pending):
                if meter.remaining() < reserve:
                    summary["stopped"] = f"budget: {meter.remaining()} left, a trial may need {reserve}"
                    summary["pending"] += [(t["id"], kk) for t, kk in pending[i:]]
                    break
                p["state"].write_text(json.dumps({"task_id": task["id"], "trial": k, "started_utc": utc_now(),
                                                  "pid": os.getpid()}), encoding="utf-8")
                try:
                    row = base | run_trial(task, k, agent, meter, judge_model, p["db"]) | {"row_written_utc": utc_now()}
                except KeyboardInterrupt:
                    append_row(p["attempts"], base | {"task_id": task["id"], "trial": k, "outcome": "interrupted",
                                                      "error_type": "interrupted", "row_written_utc": utc_now()})
                    p["state"].unlink(missing_ok=True)
                    summary["stopped"] = "interrupted by the user"
                    summary["pending"] += [(t["id"], kk) for t, kk in pending[i:]]
                    break
                append_row(p["attempts"], row)
                if row["outcome"] in FINISHED:
                    append_row(p["results"], row)
                    summary["finished"] += 1
                    errors_in_a_row = 0
                else:
                    summary["pending"].append((task["id"], k))
                    errors_in_a_row += 1
                p["state"].unlink(missing_ok=True)
                log(f"{'✅' if row['passed'] else '❌' if row['outcome'] == 'failed' else '⏸️ '} {task['id']:<28} "
                    f"#{k} {row['outcome']:<20} agent {row['agent_requests']} · judge {row['judge_requests']} · "
                    f"used {meter.used}/{budget}")
                if row["outcome"] in STOPPING or row["error_type"] == "judge_budget_stop":
                    summary["stopped"] = row["outcome"]
                    summary["pending"] += [(t["id"], kk) for t, kk in pending[i + 1:]]
                    break
                if errors_in_a_row >= max_consecutive_errors:
                    summary["stopped"] = f"{errors_in_a_row} provider/judge errors in a row"
                    summary["pending"] += [(t["id"], kk) for t, kk in pending[i + 1:]]
                    break
    finally:
        rafeeq.AUDIT = saved_audit
    summary["requests_used"] = meter.used
    return summary


# ---------------------------------------------------------------- merge and report
def merge(out_dir: Path, configs: list[str], out: Path, config_ids: dict[str, str] | None = None) -> list[dict]:
    """One CSV of finished trials across configurations: one row per (config, config_id, task, trial)."""
    rows, seen = [], set()
    for config in configs:
        files = sorted(out_dir.glob(f"results_{config}_*.csv"))
        wanted = (config_ids or {}).get(config)
        if wanted:
            files = [f for f in files if f.stem.endswith(wanted)]
        if len(files) != 1:
            raise SystemExit(f"config {config}: {len(files)} result files {[f.name for f in files]}; pick one with "
                             f"--config-id {config}=<id>, never mix versions.")
        for r in read_rows(files[0]):
            key = (r["config"], r["config_id"], r["task_id"], r["trial"])
            if key not in seen:
                seen.add(key)
                rows.append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def report(rows: list[dict]) -> list[dict]:
    out = []
    for config in sorted({r["config"] for r in rows}):
        rs = [r for r in rows if r["config"] == config]
        by_task: dict[str, list[bool]] = {}
        for r in rs:
            by_task.setdefault(r["task_id"], []).append(r["passed"] == "True")
        n = len(rs)
        avg = lambda col: round(sum(float(r[col] or 0) for r in rs) / n, 1)  # noqa: E731
        out.append({"config": config, "config_id": rs[0]["config_id"], "tasks": len(by_task), "trials": n,
                    "pass_rate": round(sum(r["passed"] == "True" for r in rs) / n, 2),
                    "pass_at_k": round(sum(any(v) for v in by_task.values()) / len(by_task), 2),
                    "pass_hat_k": round(sum(all(v) for v in by_task.values()) / len(by_task), 2),
                    "avg_agent_requests": avg("agent_requests"), "avg_agent_tokens_in": avg("agent_tokens_in"),
                    "avg_agent_tokens_out": avg("agent_tokens_out"), "judge_requests": sum(int(r["judge_requests"] or 0) for r in rs),
                    "cost_usd": round(sum(float(r["agent_cost_usd"] or 0) + float(r["judge_cost_usd"] or 0) for r in rs), 4),
                    "actual_models": "|".join(sorted({m for r in rs for m in r["actual_models"].split("|") if m}))})
    return out


# ---------------------------------------------------------------- command line
def select_tasks(ids: str | None, all_tasks: bool) -> list[dict]:
    tasks = evals.load_tasks(TASKS_FILE)
    if all_tasks:
        return tasks
    if not ids:
        raise SystemExit("Choose tasks with --ids a,b,c (or --all).")
    wanted = [i for i in ids.replace(",", " ").split() if i]
    unknown = sorted(set(wanted) - {t["id"] for t in tasks})
    if unknown:
        raise SystemExit(f"unknown task ids: {unknown}")
    return [t for t in tasks if t["id"] in wanted]


def check_config(config: str, allow_unreviewed_b: bool) -> None:
    if config == "B" and not allow_unreviewed_b:
        raise SystemExit("Configuration B is not chosen yet: it is decided from A's error analysis. "
                         "(--allow-unreviewed-b runs the current draft anyway.)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        s = sub.add_parser(name)
        s.add_argument("--config", choices=["A", "B"], required=True)
        s.add_argument("--ids")
        s.add_argument("--all", action="store_true")
        s.add_argument("--trials", type=int, default=1)
        s.add_argument("--model", default=PINNED_MODEL)
        s.add_argument("--judge-model", default=PINNED_JUDGE)
        s.add_argument("--allow-unreviewed-b", action="store_true")
        s.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    plan_p, run_p = sub.choices["plan"], sub.choices["run"]
    plan_p.add_argument("--quota", action="store_true", help="also read today's remaining free requests")
    run_p.add_argument("--max-requests", type=int, help="hard cap for this batch (agent + judge calls)")
    run_p.add_argument("--safety-margin", type=int, default=5, help="free requests left untouched")
    run_p.add_argument("--reserve", type=int, help="start a trial only with this many requests left")
    run_p.add_argument("--resume", action="store_true")
    run_p.add_argument("--no-quota-check", action="store_true")
    j = sub.add_parser("judge-check", help="one real judge request on a synthetic example")
    j.add_argument("--judge-model", default=PINNED_JUDGE)
    m = sub.add_parser("merge")
    m.add_argument("--config", nargs="+", default=["A"])
    m.add_argument("--config-id", nargs="*", default=[], help="CONFIG=ID when a configuration has several versions")
    m.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    m.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.cmd == "judge-check":
        quota = free_quota()
        print("free quota:", quota)
        if not quota or (quota.get("remaining") or 0) < 1:
            raise SystemExit("No free request left (or the quota could not be read): not sending the judge request.")
        print(json.dumps(judge_check(args.judge_model), ensure_ascii=False, indent=1))
        return 0

    if args.cmd == "merge":
        ids = dict(x.split("=", 1) for x in args.config_id)
        rows = merge(args.out_dir, args.config, args.out or args.out_dir / f"eval_{'_'.join(args.config)}.csv", ids)
        for line in report(rows):
            print(json.dumps(line, ensure_ascii=False))
        return 0

    check_config(args.config, args.allow_unreviewed_b)
    tasks = select_tasks(args.ids, args.all)
    config_id, details = fingerprint(args.config, args.model, args.judge_model)
    done = finished_keys(paths(args.out_dir, args.config, config_id)["results"])
    pending = [(t["id"], k) for t in tasks for k in range(args.trials) if (t["id"], k) not in done]
    print(f"config {args.config} [{config_id}] · code {git_version()} · model {args.model} · judge {args.judge_model}")
    print(f"tasks {len(tasks)} × trials {args.trials} · finished {len(tasks) * args.trials - len(pending)} · "
          f"pending {len(pending)}")
    if args.cmd == "plan":
        for tid, k in pending:
            print(f"   pending: {tid} #{k}")
        if args.quota:
            print("free quota:", free_quota())
        return 0

    quota = None if args.no_quota_check else free_quota()
    if not args.no_quota_check and quota is None:
        raise SystemExit("Could not read the free quota; retry, or pass --no-quota-check with --max-requests.")
    if quota:
        print(f"free requests today: used {quota['used']} / {quota['limit']} · remaining {quota['remaining']}")
    with K.MCPBridge(rafeeq.mcp_server()) as mcp:
        s = run_batch(args.config, tasks, args.trials, mcp, model=args.model, judge_model=args.judge_model,
                      max_requests=args.max_requests, quota=quota, safety_margin=args.safety_margin,
                      reserve=args.reserve, resume=args.resume, out_dir=args.out_dir)
    print(f"\nfinished {s['finished']} · pending {len(s['pending'])} · requests used {s['requests_used']}"
          + (f" · stopped: {s['stopped']}" if s["stopped"] else ""))
    print(f"results: {s['results']}\nattempts: {s['attempts']}")
    if s["pending"]:
        print("resume later with the same command and --resume")
    return 0


if __name__ == "__main__":
    sys.exit(main())
