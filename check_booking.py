"""Read-only check of what Rafeeq really wrote, for the live demo. Opens the databases with mode=ro: it changes nothing.

    uv run python capstone/check_booking.py FTR-30311

The agent database is the record: get_permit, list_slots and every write tool run on it. The MCP server's copy only
backs the stateless tools (search_policies, compute_fee) and is NOT a source for bookings; it is listed to show that.
Run this right after `rafeeq.py ask`/`chat`: the next run starts from a fresh copy of the training data and the
booking is gone (by design, so data/madar.sqlite is never changed).
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

LABS = Path(__file__).resolve().parents[1]
DATABASES = {
    "agent (the record: stateful reads and all writes)": LABS / "artifacts" / "capstone" / "db" / "madar.sqlite",
    "MCP server (stateless tools only, not a booking source)": LABS / "artifacts" / "mcp_server" / "madar.sqlite",
    "training data (never written)": LABS / "data" / "madar.sqlite",
}
AUDIT = LABS / "artifacts" / "capstone" / "audit.jsonl"


def query(path: Path, sql: str, params: tuple = ()) -> list[tuple]:
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def bookings(path: Path, permit_id: str) -> list[tuple]:
    return query(path, "SELECT inspection_id, kind, slot, district, status FROM inspections WHERE permit_id = ?",
                 (permit_id,))


def main(permit_id: str = "FTR-30311") -> int:
    for label, path in DATABASES.items():
        if not path.exists():
            print(f"{label}: {path.relative_to(LABS)} not found")
            continue
        rows = bookings(path, permit_id)
        print(f"{label}: {path.relative_to(LABS)}")
        print(f"   inspections for {permit_id}: {rows or 'none'}")

    agent_db = next(iter(DATABASES.values()))
    if agent_db.exists():
        log = query(agent_db, "SELECT ts, actor, action, detail FROM audit_log WHERE action = 'book_inspection'")
        print("Madar audit_log (agent db):", log or "no booking")

    if AUDIT.exists():
        events = [json.loads(line) for line in AUDIT.read_text(encoding="utf-8").splitlines() if line.strip()]
        print("Rafeeq audit.jsonl, last 5 events:")
        for e in events[-5:]:
            print("  ", json.dumps(e, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
