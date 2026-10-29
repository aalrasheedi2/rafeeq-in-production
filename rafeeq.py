# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=3.30", "pydantic>=2.12", "python-dotenv>=1.1", "pandas>=2.2", "mcp>=2.3"]
# ///
"""Rafeeq in Production (SDAIA capstone, Track 1), built from the course's capstone starter.

    uv run capstone/rafeeq.py ask "What is the fee for an illuminated 3 m² sign?"
    uv run capstone/rafeeq.py chat --session me              # a conversation, kept in artifacts/capstone/
    uv run capstone/rafeeq.py eval capstone/tasks.example.jsonl --config A B

What is wired in (each one maps to a capstone requirement; search for "CAPSTONE:" to find where to change it):
1. tools: stateless tools (search_policies, compute_fee) from the MCP server servers/madar_mcp.py over stdio, and the
   stateful ones (get_permit, list_slots and the writes) locally on the run's own database copy;
2. context: a spec file as the system prompt (AGENTS.md style), a session history with a compaction rule, and a cap
   on tool-result size;
3. harness: a permission policy (read → allow, write → ask a human), step and cost budgets, an audit log;
4. evals: the Day 3 eval runner over your tasks, on two configurations, with a per-task CSV for error analysis.
5. data protection: verification data (IDs, mobiles) is kept out of session files, the audit log, the approval prompt
   and the screen (agentkit's raw printer is off; a redacted trace is printed instead), and ask/chat skip the
   response cache (tests: capstone/test_privacy.py, capstone/test_booking.py).
Security (requirement 5) is yours to analyse and defend: start from Lab 4.3's egress guard.

Run from labs/ (it imports agentkit, madar and evals from there).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))

from mcp import StdioServerParameters  # noqa: E402

import agentkit as K  # noqa: E402
import evals  # noqa: E402
import madar  # noqa: E402

OUT = K.lab_dir("capstone")
AUDIT = OUT / "audit.jsonl"

# ---------------------------------------------------------------- CAPSTONE: your agent's spec (system prompt)
AGENTS_MD = Path(__file__).with_name("AGENTS.md")


def load_spec() -> str:
    """Madar's base prompt (role, today's date, core rules) followed by our project rules in capstone/AGENTS.md."""
    return madar.SYSTEM_PROMPT + "\n\n" + AGENTS_MD.read_text(encoding="utf-8").strip()


SPEC = load_spec()

# Two configurations to compare in your eval (requirement 4): change ONE thing between them. B's one change lives in
# its own file, so editing it never touches A's instructions or A's config_id in eval_runner.py.
CONFIG_B_MD = Path(__file__).with_name("config_B.md")


def config_b_addition() -> str:
    """The text B appends to A's instructions: config_B.md without its leading HTML comment."""
    text = re.sub(r"<!--.*?-->", "", CONFIG_B_MD.read_text(encoding="utf-8"), flags=re.S).strip()
    if not text:
        raise ValueError(f"{CONFIG_B_MD.name} has no instruction text after its comment")
    return text


def configs(names: tuple[str, ...] = ("A", "B")) -> dict:
    """The named configurations, read from disk now. Building A never reads config_B.md."""
    spec = load_spec()
    out = {}
    if "A" in names:
        out["A"] = {"system": spec, "model": None}
    if "B" in names:
        out["B"] = {"system": spec + "\n" + config_b_addition(), "model": None}
    return out


def _import_configs() -> dict:
    """At import, a missing or broken config_B.md must not stop A: B is then simply absent (and fails when asked)."""
    try:
        return configs()
    except (OSError, UnicodeDecodeError, ValueError):
        return configs(("A",))


CONFIGS = _import_configs()

MAX_STEPS, MAX_COST, MAX_RESULT_CHARS, MAX_HISTORY_MESSAGES = 8, 0.05, 3000, 24


# ---------------------------------------------------------------- CAPSTONE: tools
# The MCP server keeps its own copy of the database, so only stateless tools come from it. Every tool that reads or
# changes permits, inspections, slots or refunds runs locally on the run's own Madar copy (one per eval trial), so a
# booking is visible to the next get_permit/list_slots call in the same conversation.
MCP_TOOLS = ("search_policies", "compute_fee")  # policy documents and the fee schedule: same answer from any copy


def local_tools(m: madar.Madar) -> list:
    """Stateful tools, all on the same Madar copy: the reads that must see this run's writes, and the writes."""
    return [m.get_permit, m.list_slots, m.book_inspection, m.reschedule_inspection, m.create_refund, m.escalate]


def mcp_tools(mcp: K.MCPBridge) -> list:
    """The stateless tools from the MCP server (requirement 1). Its stateful tools are not offered to the model."""
    offered = {t.__name__: t for t in mcp.agent_tools()}
    missing = [n for n in MCP_TOOLS if n not in offered]
    if missing:
        raise RuntimeError(f"The MCP server does not offer {missing}")
    return [offered[n] for n in MCP_TOOLS]


def tool_source(name: str) -> str:
    return "mcp" if name in MCP_TOOLS else "local"


def mcp_server() -> StdioServerParameters:
    """The MCP server whose tools the agent also gets (requirement 1). Point it at your own server."""
    return StdioServerParameters(command="uv", args=["run", "--quiet", str(LABS / "servers" / "madar_mcp.py")])


WRITE_TOOLS = {"book_inspection", "reschedule_inspection", "create_refund", "escalate", "send_email"}


# ---------------------------------------------------------------- CAPSTONE: data protection (POL-08, PDPL)
# Verification data is used for the tool call in hand and never written to disk: sessions, the audit log and the
# officer's approval prompt see redacted values. Enforced here in code, not left to the prompt.
SENSITIVE_KEYS = {"national_id_last4", "national_id", "mobile", "phone", "iban", "email",
                  "api_key", "apikey", "openrouter_api_key", "token", "access_token", "password", "secret",
                  "authorization"}
REDACTED = "[redacted]"
_D = r"[0-9٠-٩]"  # Western and Arabic-Indic digits
# Madar's own record IDs (INS-9002, REF-5001, ESC-0001, OUT-0001, BLD-20417, ...) are not personal data. The last-4
# rules below skip digits that follow exactly one of these upper-case prefixes and a hyphen; any other spelling
# ("ID 1234", "ID XYZ-1234", "ins-1234") is still masked.
_NOT_MADAR_ID = r"(?-i:(?<!INS-)(?<!REF-)(?<!ESC-)(?<!OUT-)(?<!BLD-)(?<!REN-)(?<!SGN-)(?<!FTR-)(?<!EVT-))"
_PATTERNS = [
    # credentials first, so their digits are not half-matched by the rules below
    (re.compile(r"\b(?:sk-or-v1-|sk-or-|sk-|hf_)[A-Za-z0-9_-]{16,}"), "[secret]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{16,}"), "Bearer [secret]"),
    (re.compile(r"(?i)\b((?:openrouter_)?api[_-]?key|access[_-]?token|token|secret|password|authorization)"
                r"(\s*[:=]\s*)[\"']?[^\s\"',;]+"), r"\1\2[secret]"),
    (re.compile(rf"(?<![0-9٠-٩])(?:(?:(?:\+|[0٠]{{2}})?[9٩][6٦]{{2}}|[0٠])[\s-]?)?[5٥]"
                rf"(?:[\s-]?{_D}){{8}}(?![0-9٠-٩])"), "[mobile]"),
    (re.compile(rf"(?<![0-9٠-٩])[12١٢]{_D}{{9}}(?![0-9٠-٩])"), "[national-id]"),
    (re.compile(r"\bSA\d{2}(?:\s?[0-9A-Z]{4}){5}\b", re.IGNORECASE), "[iban]"),
    # last 4 ID digits named as such: "ID ends 1234", "last 4 digits: 1234", "آخر 4 أرقام من الهوية 1234"
    (re.compile(rf"(?i)((?:national\s*id|\bid\b|iqama|identity|last\s*(?:4|four)|\bends?\b\s*(?:in|with)?|هوية|الهوية|إقامة|آخر\s*(?:4|٤|أربع))"
                rf"[^0-9٠-٩\n]{{0,30}}){_NOT_MADAR_ID}{_D}{{4}}(?![0-9٠-٩])"), r"\1[id-last4]"),
    # last 4 digits written next to a mobile: "1234 · [mobile]" or "[mobile], 1234"
    (re.compile(rf"(?<![0-9٠-٩]){_NOT_MADAR_ID}{_D}{{4}}(?=[\s·,/|-]{{1,5}}\[mobile\])"), "[id-last4]"),
    (re.compile(rf"(\[mobile\][\s·,/|-]{{1,5}}){_NOT_MADAR_ID}{_D}{{4}}(?![0-9٠-٩])"), r"\1[id-last4]"),
]


def is_sensitive_key(key) -> bool:
    return isinstance(key, str) and key.lower() in SENSITIVE_KEYS


def redact_text(text: str) -> str:
    """Mask credentials, mobiles, national IDs, IBANs and last-4 ID digits in free text; keep Madar record IDs."""
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def redact(obj):
    """Redact a JSON-like value: sensitive keys lose their value, every string is passed through redact_text."""
    if isinstance(obj, dict):
        return {k: (REDACTED if is_sensitive_key(k) and v not in (None, "") else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return redact_text(obj) if isinstance(obj, str) else obj


def safe_history(history: list) -> list:
    """What a session file may keep: the user's and Rafeeq's words, redacted. Tool calls (their arguments carry
    verification data) and tool results (permit-file details) are dropped, so a resumed session must verify again."""
    out = []
    for msg in history:
        if msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), str) and msg["content"].strip():
            out.append({"role": msg["role"], "content": redact_text(msg["content"])})
    return out


def save_session(path: Path, history: list) -> None:
    path.write_text(json.dumps(safe_history(history), ensure_ascii=False, indent=1), encoding="utf-8")


def load_session(path: Path) -> list:
    """Load a session, sanitising it again in case it was written before this protection existed."""
    return safe_history(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else []


# fields of a tool result that the audit log keeps for tracing (identifiers, statuses, amounts; no personal data)
AUDIT_RESULT_FIELDS = ("permit_id", "inspection_id", "refund_id", "escalation_id", "message_id", "type", "status",
                       "kind", "slot", "old_slot", "new_slot", "fee_sar", "amount_sar", "policy", "working_days_overdue")


def audit(event: dict) -> None:
    """Append one redacted, timestamped event to the audit log."""
    event = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **redact(event)}
    with AUDIT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def audit_result(result: str) -> dict:
    """The operational outcome of a tool call: ok/error, the traceable fields, or how many items came back."""
    if result.startswith('{"error"'):
        try:
            message = json.loads(result)["error"]
        except (json.JSONDecodeError, KeyError, TypeError):
            message = result
        return {"outcome": "error", "error": str(message)[:300]}
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        return {"outcome": "ok"}
    if isinstance(data, dict):
        return {"outcome": "ok", "result": {k: data[k] for k in AUDIT_RESULT_FIELDS if k in data}}
    if isinstance(data, list):
        return {"outcome": "ok", "result": {"items": len(data)}}
    return {"outcome": "ok"}


# ---------------------------------------------------------------- CAPSTONE: harness controls
def permission(name: str, args: dict) -> str:
    return "ask" if name in WRITE_TOOLS else "allow"


# what the officer sees when asked to approve a write: the action and its target, never the verification values
APPROVAL_FIELDS = ("permit_id", "inspection_id", "slot", "new_slot", "reason")


def approval_summary(name: str, args: dict) -> str:
    shown = ", ".join(f"{k}={redact_text(str(args[k]))}" for k in APPROVAL_FIELDS if args.get(k) not in (None, ""))
    return f"{name}({shown})"


def approve_cli(name: str, args: dict) -> bool:
    return input(f"\n🛡️  Allow {approval_summary(name, args)}? [y/N] ").strip().lower() == "y"


def audited(approve, show: bool = False):
    """Wrap an approver so that every human decision is logged (without the verification values)."""
    def decide(name: str, args: dict) -> bool:
        ok = bool(approve(name, args))
        audit({"event": "approval", "tool": name, "source": tool_source(name), "args": args,
               "decision": "approved" if ok else "denied"})
        if show:
            print(f"🛡️  {'approved' if ok else 'denied'}: {name}")
        return ok
    return decide


def trace_line(name: str, args: dict, result: str) -> str:
    """One screen line per tool call, built only from redacted arguments and the audited outcome."""
    shown = {k: v for k, v in redact(args).items() if not is_sensitive_key(k)}
    outcome = audit_result(result)
    detail = outcome.get("result") or redact_text(outcome.get("error", ""))[:160]
    return (f"🔧 [{tool_source(name)}] {name}({json.dumps(shown, ensure_ascii=False)}) → {outcome['outcome']} "
            f"{json.dumps(detail, ensure_ascii=False) if detail else ''}").rstrip()


def cap_result(name: str, args: dict, result: str) -> str | None:
    """after_tool hook: keep tool results inside the context budget, and log every call (redacted)."""
    audit({"event": "tool_call", "tool": name, "source": tool_source(name), "args": args, **audit_result(result),
           "chars": len(result)})
    if len(result) > MAX_RESULT_CHARS:
        return result[:MAX_RESULT_CHARS] + f"\n[… {len(result) - MAX_RESULT_CHARS} characters cut; ask for less]"
    return None


def compact(history: list) -> list:
    """Context rule: keep the last messages, never starting on a tool result (its call would be missing)."""
    if len(history) <= MAX_HISTORY_MESSAGES:
        return history
    tail = history[-MAX_HISTORY_MESSAGES:]
    while tail and tail[0].get("role") != "user":
        tail = tail[1:]
    return tail


# ---------------------------------------------------------------- the agent
def make_agent(config: dict, mcp: K.MCPBridge, approve=approve_cli, show: bool = False):
    """show=True prints our redacted trace. agentkit's own printer (verbose) shows raw arguments, so it stays off."""
    def after_tool(name: str, args: dict, result: str) -> str | None:
        if show:
            print(trace_line(name, args, result))
        return cap_result(name, args, result)

    def agent(prompt: str, m: madar.Madar, history: list | None = None, **kw) -> K.Run:
        tools = local_tools(m) + mcp_tools(mcp)
        names = [t.__name__ for t in tools]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate tool names offered to the model: {names}")
        kw.pop("verbose", None)
        return K.run_agent(prompt, tools, system=config["system"], model=config["model"], history=history,
                           max_steps=MAX_STEPS, max_cost=MAX_COST, permission=permission,
                           approve=audited(approve, show), after_tool=after_tool, verbose=False, **kw)
    return agent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    c = sub.add_parser("chat")
    c.add_argument("--session", default="default")
    e = sub.add_parser("eval")
    e.add_argument("tasks", type=Path)
    e.add_argument("--config", nargs="+", default=["A", "B"])
    e.add_argument("--trials", type=int, default=1)
    args = ap.parse_args()

    with K.MCPBridge(mcp_server()) as mcp:
        # ask and chat take real people's details, so their model calls skip agentkit's response cache (cache/ would
        # keep the tool-call arguments on disk). eval keeps the cache: its tasks use synthetic Madar data only.
        if args.cmd == "ask":
            print("🧑", redact_text(args.question))
            run = make_agent(CONFIGS["A"], mcp, show=True)(args.question, madar.Madar(OUT / "db"), use_cache=False)
            print("\n🤖", redact_text(run.final) or f"(stopped: {run.stopped})")
            print(f"   steps: {run.steps} · stopped: {run.stopped} · model calls so far: {K.LEDGER}")
        elif args.cmd == "chat":
            path = OUT / f"session-{args.session}.json"
            history = load_session(path)  # redacted words only: verification is asked for again
            m, agent = madar.Madar(OUT / "db"), make_agent(CONFIGS["A"], mcp, show=True)
            print("Type your message (empty line to quit).")
            while (msg := input("\n🧑 ").strip()):
                run = agent(msg, m, history=compact(history), use_cache=False)
                print("🤖", redact_text(run.final) or f"(stopped: {run.stopped})")
                history = run.state[1:]  # full context stays in memory for this conversation only
                save_session(path, history)
        else:
            tasks = evals.load_tasks(args.tasks)
            consent = lambda name, a: True  # noqa: E731 - unattended: task prompts give consent; db graders check writes
            frames = [evals.run_suite(make_agent(CONFIGS[c], mcp, approve=consent), tasks, trials=args.trials, name=c)
                      for c in args.config]
            import pandas as pd

            df = pd.concat(frames)
            df.drop(columns="final").to_csv(OUT / "eval.csv", index=False)
            overall, per_task = evals.report(df)
            print("\n", overall, "\n\nper task written to", OUT / "eval.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
