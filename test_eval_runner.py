"""Offline checks of capstone/eval_runner.py with a scripted model: persistence, resume, metering, error classes.

No request reaches OpenRouter; tasks are the course's official ones; databases are temporary copies.

    uv run pytest capstone/test_eval_runner.py -v
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

import agentkit as K  # noqa: E402
import eval_runner as R  # noqa: E402
import evals  # noqa: E402
from test_privacy import LAST4, MOBILE, FakeMCPServer, Msg  # noqa: E402

TASKS = {t["id"]: t for t in evals.load_tasks(R.TASKS_FILE)}
FEE, NO_CREDS, BOOK = TASKS["fee-sign-illuminated"], TASKS["status-no-credentials"], TASKS["book-inspection"]
SLOT = BOOK["graders"][0]["expect"]
FAKE_KEY = "sk-" + "or-v1-" + "fakefakefake0123456789"


class RateLimitError(Exception):
    """Same name as openai's, which is all the runner looks at."""


class APIConnectionError(Exception):
    pass


def reply(msg: Msg, model="nvidia/nemotron-3-ultra-550b-a55b:free", tin=100, tout=10, finish="stop", reasoning=0):
    usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": tin, "completion_tokens": tout, "cost": 0,
                                                "completion_tokens_details": {"reasoning_tokens": reasoning}})
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)], usage=usage, model=model)


class FakeProvider:
    """A scripted OpenRouter: per task prompt, a list of agent replies (or exceptions); judge verdicts separately."""

    def __init__(self):
        self.agent: dict[str, list] = {}
        self.judge: list = []
        self.kwargs: list[dict] = []
        self.prompts: list[str] = []

    def __call__(self, messages, model=None, tools=None, **kw):
        self.kwargs.append(kw | {"model": model})
        if messages[0]["content"] == evals.JUDGE_PROMPT:
            item = self.judge.pop(0) if self.judge else '{"pass": true, "reason": "ok"}'
            if isinstance(item, Exception):
                raise item
            text, finish = item if isinstance(item, tuple) else (item, "stop")
            return reply(Msg(text), model="nvidia/nemotron-3.5-lightning:free", tin=50, tout=5, finish=finish,
                         reasoning=40)
        prompt = next(m["content"] for m in messages if m["role"] == "user")
        self.prompts.append(prompt)
        item = self.agent[prompt].pop(0)
        if isinstance(item, Exception):
            raise item
        return reply(item)


def script_fee(p: FakeProvider, answer="The fee is 750 SAR per year (POL-02)."):
    p.agent.setdefault(FEE["prompt"], []).extend([
        Msg(calls=[("compute_fee", {"permit_type": "SGN", "sign_m2": 3, "illuminated": True})]), Msg(answer)])


def script_no_creds(p: FakeProvider):
    p.agent.setdefault(NO_CREDS["prompt"], []).append(
        Msg("Please give the last 4 digits of the national ID and the registered mobile first (POL-08)."))


def script_book(p: FakeProvider):
    p.agent.setdefault(BOOK["prompt"], []).extend([
        Msg(calls=[("book_inspection", {"permit_id": "FTR-30311", "slot": SLOT, "national_id_last4": LAST4,
                                        "mobile": MOBILE})]),
        Msg(f"Booked INS-9002 on {SLOT}.")])


@pytest.fixture
def provider(monkeypatch):
    p = FakeProvider()
    monkeypatch.setattr(K, "chat", p)
    monkeypatch.setattr(K, "client", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("network")))
    return p


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "git_version", lambda: "test")
    return SimpleNamespace(out=tmp_path / "evals", mcp=FakeMCPServer(tmp_path / "mcp"), logs=[])


def batch(env, tasks, config="A", **kw):
    kw.setdefault("max_requests", 40)
    return R.run_batch(config, tasks, 1, env.mcp, out_dir=env.out, log=env.logs.append, **kw)


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- persistence and resume
def test_rows_are_appended_and_never_overwritten(provider, env):
    script_fee(provider)
    first = batch(env, [FEE])
    before = rows(first["results"])
    script_no_creds(provider)
    second = batch(env, [FEE, NO_CREDS], resume=True)

    after = rows(second["results"])
    assert first["results"] == second["results"]
    assert after[0] == before[0]  # the first row is untouched
    assert [r["task_id"] for r in after] == ["fee-sign-illuminated", "status-no-credentials"]
    assert provider.prompts.count(FEE["prompt"]) == 2  # 2 agent calls in run 1, none in run 2


def test_a_second_run_without_resume_is_refused(provider, env):
    script_fee(provider)
    batch(env, [FEE])
    with pytest.raises(SystemExit, match="--resume"):
        batch(env, [FEE])


def test_resume_reruns_only_unfinished_trials_and_never_duplicates(provider, env):
    script_fee(provider)
    provider.agent[NO_CREDS["prompt"]] = [RateLimitError("429 Provider returned error")]
    s = batch(env, [FEE, NO_CREDS])
    assert s["finished"] == 1 and s["pending"] == [("status-no-credentials", 0)]

    script_no_creds(provider)
    s2 = batch(env, [FEE, NO_CREDS], resume=True)
    assert s2["finished"] == 1 and s2["pending"] == []
    s3 = batch(env, [FEE, NO_CREDS], resume=True)  # nothing left: no call at all
    assert s3["finished"] == 0 and s3["requests_used"] == 0

    keys = [(r["task_id"], r["trial"]) for r in rows(s["results"])]
    assert len(keys) == len(set(keys)) == 2
    attempts = [r["outcome"] for r in rows(s["attempts"])]
    assert attempts == ["passed", "provider_rate_limit", "passed"]


def test_an_interrupted_trial_is_detected_and_rerun(provider, env):
    config_id, _ = R.fingerprint("A", R.PINNED_MODEL, R.PINNED_JUDGE)
    state = R.paths(env.out, "A", config_id)["state"]
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({"task_id": FEE["id"], "trial": 0, "started_utc": "2026-11-01T00:00:00+00:00"}))
    script_fee(provider)
    s = batch(env, [FEE], resume=True)

    assert any("interrupted trial found" in line for line in env.logs)
    assert [r["outcome"] for r in rows(s["attempts"])] == ["interrupted", "passed"]
    assert len(rows(s["results"])) == 1 and not state.exists()


# ---------------------------------------------------------------- metering and fairness
def test_each_row_is_metered_and_agent_and_judge_are_kept_apart(provider, env):
    script_fee(provider)
    script_no_creds(provider)
    s = batch(env, [FEE, NO_CREDS])
    fee, creds = rows(s["results"])

    assert (fee["agent_requests"], fee["judge_requests"]) == ("2", "0")
    assert (creds["agent_requests"], creds["judge_requests"]) == ("1", "1")  # status-no-credentials has a judge
    assert (creds["agent_tokens_in"], creds["judge_tokens_in"]) == ("100", "50")
    assert fee["requested_model"] == R.PINNED_MODEL and fee["actual_models"] == R.PINNED_MODEL
    assert creds["judge_model"] == R.PINNED_JUDGE and creds["judge_actual_models"] == R.PINNED_JUDGE
    assert fee["tools"] == "compute_fee" and fee["cache_hits"] == "0" and fee["config_id"] == s["config_id"]
    assert fee["code_version"] == "test" and fee["approval"].startswith("auto")
    assert s["requests_used"] == 4


def test_every_call_is_forced_fresh_and_without_fallback_models(provider, env):
    script_fee(provider)
    script_no_creds(provider)
    batch(env, [FEE, NO_CREDS])
    assert provider.kwargs and all(kw["use_cache"] is False and kw["fallbacks"] is False for kw in provider.kwargs)
    assert {kw["model"] for kw in provider.kwargs} == {R.PINNED_MODEL, R.PINNED_JUDGE}


def test_configurations_and_versions_never_share_files(provider, env):
    script_fee(provider)
    script_fee(provider)
    a = batch(env, [FEE], config="A")
    b = batch(env, [FEE], config="B")
    assert a["config_id"] != b["config_id"] and a["results"] != b["results"]
    assert {r["config"] for r in rows(a["results"])} == {"A"} and {r["config"] for r in rows(b["results"])} == {"B"}
    other_model, _ = R.fingerprint("A", "some/other-model:free", R.PINNED_JUDGE)
    assert other_model != a["config_id"]


def test_b_needs_an_explicit_choice_on_the_command_line():
    with pytest.raises(SystemExit, match="not chosen yet"):
        R.check_config("B", allow_unreviewed_b=False)
    R.check_config("A", allow_unreviewed_b=False)


# ---------------------------------------------------------------- task failure vs. provider, judge and budget
def test_a_wrong_answer_is_a_failed_task(provider, env):
    script_fee(provider, answer="It costs 600 SAR.")
    s = batch(env, [FEE])
    [row] = rows(s["results"])
    assert (row["outcome"], row["passed"], row["error_type"]) == ("failed", "False", "")
    assert "contains_any" in row["failed_graders"]


@pytest.mark.parametrize("exc, kind", [(RateLimitError("429"), "provider_rate_limit"),
                                       (APIConnectionError("reset"), "provider_connection"),
                                       (ConnectionError("network down"), "provider_connection")])
def test_provider_errors_stay_pending_and_are_not_failures(provider, env, exc, kind):
    provider.agent[FEE["prompt"]] = [exc]
    s = batch(env, [FEE])
    assert not s["results"].exists() or rows(s["results"]) == []
    [attempt] = rows(s["attempts"])
    assert attempt["outcome"] == kind and attempt["passed"] == "False"
    assert s["pending"] == [("fee-sign-illuminated", 0)]


def test_a_judge_error_keeps_the_trial_pending_and_the_rest_of_the_batch_running(provider, env):
    script_no_creds(provider)
    script_fee(provider)
    provider.judge = [RateLimitError("judge 429")]
    s = batch(env, [NO_CREDS, FEE])
    assert [r["task_id"] for r in rows(s["results"])] == ["fee-sign-illuminated"]
    first = rows(s["attempts"])[0]
    assert first["outcome"] == "judge_error" and first["error_type"] == "judge_error:provider_rate_limit"
    assert first["agent_requests"] == "1" and first["judge_failed_requests"] == "1"
    graders = json.loads(first["graders_json"])  # the other grader's result survives the broken judge
    assert [(g["type"], g["status"], g["pass"]) for g in graders] == [("not_contains", "ok", True),
                                                                      ("judge", "error:provider_rate_limit", None)]


def test_an_unparseable_judge_verdict_is_not_a_pass_or_a_fail(provider, env):
    script_no_creds(provider)
    provider.judge = ["I think it is fine"]
    s = batch(env, [NO_CREDS])
    [attempt] = rows(s["attempts"])
    assert attempt["outcome"] == "judge_error" and attempt["error_type"] == "judge_unparseable"
    assert s["pending"] == [("status-no-credentials", 0)]


@pytest.mark.parametrize("judge_reply, status", [
    (("", "length"), "truncated"),                   # what the first A batch saw: reasoning used every token
    ("", "unparseable"),                             # empty, but not cut
    (('{"pass": tr', "length"), "truncated"),        # cut in the middle of the JSON
    ('{"reason": "looks fine"}', "unparseable"),     # evals.judge would call this a fail
    ('{"pass": "yes", "reason": "ok"}', "unparseable"),
    ("Sure! The reply is good.", "unparseable"),
])
def test_bad_judge_replies_are_judge_errors_and_keep_the_other_graders(provider, env, judge_reply, status):
    script_no_creds(provider)
    provider.judge = [judge_reply]
    s = batch(env, [NO_CREDS])
    [attempt] = rows(s["attempts"])
    assert (attempt["outcome"], attempt["error_type"], attempt["judge_status"]) == ("judge_error", f"judge_{status}",
                                                                                   status)
    assert attempt["judge_finish_reasons"] == (judge_reply[1] if isinstance(judge_reply, tuple) else "stop")
    assert json.loads(attempt["graders_json"])[0] == {"type": "not_contains", "pass": True, "status": "ok",
                                                      "detail": ""}
    assert s["pending"] == [("status-no-credentials", 0)] and not s["results"].exists()


def test_a_valid_verdict_is_graded_and_its_trace_recorded(provider, env):
    script_no_creds(provider)
    provider.judge = ['Verdict: {"pass": false, "reason": "it does not ask for the mobile"}']
    s = batch(env, [NO_CREDS])
    [row] = rows(s["results"])
    assert (row["outcome"], row["judge_status"], row["stop_reason"]) == ("failed", "ok", "done")
    assert row["judge_reasoning_tokens"] == "40" and row["agent_finish_reasons"] == "stop"
    assert "it does not ask for the mobile" in row["failed_graders"]


def test_only_judge_calls_get_the_larger_token_cap(provider, env):
    script_fee(provider)
    script_no_creds(provider)
    batch(env, [FEE, NO_CREDS])
    judge = [kw for kw in provider.kwargs if kw["model"] == R.PINNED_JUDGE]
    agent = [kw for kw in provider.kwargs if kw["model"] == R.PINNED_MODEL]
    assert judge and all(kw["max_tokens"] == R.JUDGE_MAX_TOKENS and kw["temperature"] == 0 for kw in judge)
    assert agent and all("max_tokens" not in kw for kw in agent)  # the agent's calls are exactly as before


def test_the_judge_fix_leaves_rafeeqs_tools_permissions_and_prompt_alone(monkeypatch, env):
    import rafeeq

    offered = []
    monkeypatch.setattr(K, "chat", lambda messages, **kw: offered.append((messages[0]["content"], kw["tools"])) or
                        reply(Msg("ok")))
    batch(env, [FEE])
    system, tools = offered[0]
    assert system == rafeeq.CONFIGS["A"]["system"]
    assert sorted(t.__name__ for t in tools) == sorted(["get_permit", "list_slots", "book_inspection",
                                                       "reschedule_inspection", "create_refund", "escalate",
                                                       "search_policies", "compute_fee"])
    assert all(rafeeq.permission(n, {}) == "ask" for n in rafeeq.WRITE_TOOLS)
    assert all(rafeeq.permission(n, {}) == "allow" for n in ("get_permit", "list_slots", "search_policies",
                                                             "compute_fee"))


def test_graders_see_the_real_answer_and_files_get_the_redacted_one(provider, env):
    task = {"id": "synthetic-echo", "category": "privacy", "lang": "en", "prompt": "Repeat my mobile.",
            "graders": [{"type": "contains_any", "values": [MOBILE]}]}
    provider.agent[task["prompt"]] = [Msg(f"Your mobile is {MOBILE} and ID ends {LAST4}.")]
    s = batch(env, [task])
    [row] = rows(s["results"])
    assert row["passed"] == "True"  # the grader matched the real number in memory
    assert "[mobile]" in row["final_redacted"] and MOBILE not in row["final_redacted"]
    for p in env.out.rglob("*.csv"):
        assert MOBILE not in p.read_text(encoding="utf-8")


def test_judge_check_makes_exactly_one_request(provider):
    result = R.judge_check()
    assert result["ok"] is True and result["pass"] is True and result["requests"] == 1
    assert result["finish_reason"] == "stop" and result["max_tokens"] == R.JUDGE_MAX_TOKENS
    assert [kw["model"] for kw in provider.kwargs] == [R.PINNED_JUDGE]


def test_judge_check_reports_a_bad_reply_without_retrying(provider):
    provider.judge = [("", "length")]
    result = R.judge_check()
    assert result["ok"] is False and result["finish_reason"] == "length" and result["requests"] == 1
    assert len(provider.kwargs) == 1


def test_a_harness_crash_is_a_failed_trial(provider, env):
    provider.agent[FEE["prompt"]] = [ValueError("bug in our code")]
    s = batch(env, [FEE])
    [row] = rows(s["results"])
    assert (row["outcome"], row["error_type"]) == ("failed", "harness_error")


def test_the_budget_never_starts_a_trial_it_cannot_finish(provider, env):
    script_fee(provider)
    s = batch(env, [FEE, NO_CREDS], max_requests=10)  # reserve is MAX_STEPS + 1 = 9; 8 left after the first
    assert s["finished"] == 1 and s["pending"] == [("status-no-credentials", 0)]
    assert s["stopped"].startswith("budget") and s["requests_used"] == 2


def test_the_quota_lowers_the_budget_with_a_safety_margin(provider, env):
    script_fee(provider)
    s = batch(env, [FEE], max_requests=None, quota={"remaining": 13, "used": 37, "limit": 50})
    assert s["finished"] == 0 and s["stopped"].startswith("budget: 8 left")  # 13 - 5 < 9


def test_a_budget_hit_inside_a_trial_is_a_budget_stop(provider, env):
    script_fee(provider)
    s = batch(env, [FEE], max_requests=1, reserve=0)
    [attempt] = rows(s["attempts"])
    assert attempt["outcome"] == "budget_stop" and s["stopped"] == "budget_stop"
    assert not s["results"].exists()


def test_consecutive_provider_errors_stop_the_batch(provider, env):
    for t in (FEE, NO_CREDS, BOOK):
        provider.agent[t["prompt"]] = [RateLimitError("429")]
    s = batch(env, [FEE, NO_CREDS, BOOK], max_consecutive_errors=2)
    assert len(rows(s["attempts"])) == 2 and s["stopped"].startswith("2 provider")
    assert [k for k, _ in s["pending"]] == ["fee-sign-illuminated", "status-no-credentials", "book-inspection"]


# ---------------------------------------------------------------- privacy and data safety
def test_no_key_or_verification_data_in_any_output(provider, env):
    script_book(provider)
    provider.agent[NO_CREDS["prompt"]] = [APIConnectionError(f"failed for {MOBILE}, key {FAKE_KEY}")]
    s = batch(env, [BOOK, NO_CREDS])
    files = [p for p in env.out.rglob("*") if p.is_file() and p.suffix in (".csv", ".jsonl", ".json")]
    text = "".join(p.read_text(encoding="utf-8") for p in files)
    assert rows(s["results"])[0]["passed"] == "True"  # the booking itself worked
    for secret in (MOBILE, MOBILE[1:], FAKE_KEY, "fakefakefake0123456789"):
        assert secret not in text
    assert not re.search(rf"(?<![0-9a-fA-F]){LAST4}(?![0-9a-fA-F])", text)  # hashes may contain the digits by chance
    assert R.paths(env.out, "A", s["config_id"])["audit"].exists()  # eval tool calls have their own audit log


def test_the_training_database_is_never_written(provider, env):
    training = K.DATA / "madar.sqlite"
    before = hashlib.sha256(training.read_bytes()).hexdigest()
    script_book(provider)
    batch(env, [BOOK])
    assert hashlib.sha256(training.read_bytes()).hexdigest() == before


def test_free_quota_reads_the_key_endpoint_without_printing_the_key(monkeypatch, capsys):
    import httpx

    seen = {}
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)

    def fake_get(url, headers, timeout):
        seen.update(url=url, auth=headers["Authorization"])
        data = {"data": {"is_free_tier": True, "free_model_daily_requests": {"used": 7, "limit": 50, "remaining": 43}}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    monkeypatch.setattr(httpx, "get", fake_get)
    assert R.free_quota() == {"used": 7, "limit": 50, "remaining": 43, "is_free_tier": True}
    assert seen["url"].endswith("/api/v1/key") and seen["auth"] == f"Bearer {FAKE_KEY}"

    monkeypatch.setattr(httpx, "get", lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError(f"boom {FAKE_KEY}")))
    assert R.free_quota() is None
    assert FAKE_KEY not in capsys.readouterr().out


def test_the_script_declares_httpx_for_uv_run():
    """`uv run capstone/eval_runner.py` builds its own environment from the header: the quota check needs httpx."""
    header = Path(R.__file__).read_text(encoding="utf-8").split("# ///", 2)[1]
    assert '"httpx' in header


# ---------------------------------------------------------------- merge
def test_merge_keeps_one_row_per_trial_and_refuses_mixed_versions(provider, env, tmp_path):
    script_fee(provider)
    script_no_creds(provider)
    batch(env, [FEE])
    batch(env, [FEE, NO_CREDS], resume=True)
    merged = R.merge(env.out, ["A"], tmp_path / "merged.csv")
    assert [r["task_id"] for r in merged] == ["fee-sign-illuminated", "status-no-credentials"]
    [summary] = R.report(merged)
    assert summary["trials"] == 2 and summary["pass_rate"] == 1.0 and summary["judge_requests"] == 1

    (env.out / "results_A_000000000000.csv").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="never mix versions"):
        R.merge(env.out, ["A"], tmp_path / "x.csv")
