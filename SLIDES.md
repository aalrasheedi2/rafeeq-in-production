---
marp: true
title: Rafeeq in Production
paginate: true
---

# Rafeeq in Production · رفيق في بيئة الإنتاج

SDAIA Academy · Agentic AI Systems Engineering · Capstone Track 1
https://sdaiaacademy.github.io/agentic-ai/

aalrasheedi2

---

## 1 · The problem and the user (1 min)

- **Users:** applicants of Madar Municipal Services (synthetic), in Arabic and English; officers who approve.
- **Today:** a shared inbox answered by hand: slow, inconsistent.
- **What can go wrong:** disclosing an application to the wrong person, a wrong fee, booking or refunding without
  consent, obeying instructions hidden in an email.
- **Done =** policy-cited answers · identity before disclosure · an officer approves every write · nothing personal
  on disk · injections never become actions.

---

## 2 · Live demo: book a food-truck health inspection (2 min)

`uv run rafeeq/rafeeq.py ask "...FTR-30311... 3 or 4 November before 12:00..."`

1. `[local] get_permit` → approved, Al-Rawdah (identity checked in code)
2. `[mcp] search_policies` → POL-05
3. `[local] list_slots` → 3 free; only **4 Nov 09:00** is before noon
4. `🛡️ Allow book_inspection(permit_id=FTR-30311, slot=2026-11-04 09:00)? [y/N]` → officer types **y**
5. `check_booking.py` (read-only) → **INS-9002 · booked**, only in the run's database

The screen and logs never show the ID or mobile. Backup: `DEMO_RESULTS.md`.

---

## 3 · Design across the ladder (2 min)

| Rung | Decision |
|---|---|
| Prompt | Madar prompt + `AGENTS.md` (scope, POL-08 identity, grounded fees, untrusted data, consent, PDPL) |
| Context | `search_policies` over **MCP**; compaction to 24 messages; tool results ≤ 3,000 chars |
| Harness | writes → officer `[y/N]`; 8 steps / $0.05; redacted audit, sessions and screen |
| Loop / Graph | not needed: one turn, short tool loop; the officer gate lives inside the loop |

Stateful tools run locally on the run's own DB copy; only stateless tools come from MCP (the server has its own
copy). Rafeeq has **no send_email**.

---

## 4 · Evals: A vs B (2 min)

Official tasks, official graders · pinned models · no fallback, no cache · fresh DB per trial · **15 of 30 tasks**, k = 1

| | pass rate | avg requests | avg tokens in |
|---|---|---|---|
| **A** (`5c438a45ebf5`) | **13/15 = 0.87** | 2.3 | 8,289 |
| **B** (`a32c12de3a1f`) | **13/15 = 0.87** | 2.1 | 7,872 |

- **Biggest real failure in A:** an Arabic question searched only in Arabic → missed the English policy text
  ("structural engineer's report").
- **B = one change:** for Arabic questions, also search in English → that task passes, in 2 requests instead of 4.
- B failed one English task A passed (variance, k = 1). One malformed-provider-response row each, counted as failure.
- Rejected change: booking without the applicant picking the slot (contradicts POL-10 and Madar's booking skill).

---

## 5 · Trifecta, attack, defence (1 min)

- Production Rafeeq: private + untrusted, **no egress** (no send_email; writes need an officer).
- Staff inbox variant (Lab 4.3): `staff_lookup` + email + `send_email` = all three legs.
- **Attack:** an email asks for a "status badge" image whose URL carries the permit → `tracker.example`.

| Configuration (real model, image-beacon) | leaked | guard fired |
|---|---|---|
| naive | **3 / 3** | – |
| guard only (naive prompt + Python egress guard) | **0 / 3** | **3 / 3** |
| full defence (+ spotlighting) | 0 / 3 | 0 / 3 |

The defence is code: `before_tool` blocks mail not to the sender, IDs/mobiles, and links outside madar.example.

---

## Limitations · next steps

- 15/30 tasks, k = 1 → run all 30 with k ≥ 3.
- Runner: classify a response with no `choices` as a provider error.
- The course's FastAPI service runs but is not integrated with our privacy safeguards.
- Pattern-based redaction misses names; some answers share more than asked.

Repository: README → How to run · `results/` holds every CSV.
