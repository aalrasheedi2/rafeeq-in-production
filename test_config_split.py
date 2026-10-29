"""Offline checks that B lives in its own file: editing it changes B only, while shared changes move both fingerprints.

    uv run pytest capstone/test_config_split.py -v
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

import eval_runner as R  # noqa: E402
import madar  # noqa: E402
import rafeeq  # noqa: E402

# sha256 of each configuration's full system prompt, recorded before the split (commit a507565)
A_SYSTEM_SHA = "fb4fbd6d59cd5319e3306d4f9f784339b7262f3cd4bbf4a0262afae041bbd77d"
B_SYSTEM_SHA = "d644a8ce2dbefb2eb7f71fcd0b92bc21aa6fb00e484dc225314639bf777cc394"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ids() -> tuple[str, str]:
    return R.fingerprint("A", R.PINNED_MODEL, R.PINNED_JUDGE)[0], R.fingerprint("B", R.PINNED_MODEL, R.PINNED_JUDGE)[0]


@pytest.fixture
def b_file(tmp_path, monkeypatch):
    path = tmp_path / "config_B.md"
    path.write_text(rafeeq.CONFIG_B_MD.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(rafeeq, "CONFIG_B_MD", path)
    return path


# ---------------------------------------------------------------- the split changed nothing the model sees
def test_a_instructions_are_byte_for_byte_what_they_were():
    assert sha(rafeeq.configs()["A"]["system"]) == A_SYSTEM_SHA == sha(rafeeq.CONFIGS["A"]["system"])
    assert rafeeq.configs()["A"]["system"] == madar.SYSTEM_PROMPT + "\n\n" + rafeeq.AGENTS_MD.read_text(
        encoding="utf-8").strip()


def test_b_is_a_plus_exactly_one_retrieval_paragraph():
    """B_SYSTEM_SHA was the course's placeholder draft; the chosen B replaces only config_B.md's text."""
    b = rafeeq.configs()["B"]["system"]
    assert sha(b) != B_SYSTEM_SHA
    assert b == rafeeq.configs(("A",))["A"]["system"] + "\n" + rafeeq.config_b_addition()
    addition = rafeeq.config_b_addition()
    assert "search_policies with an English query" in addition and addition.count("\n\n") == 0
    for word in ("approve", "approval", "book", "permission", "consent", "go ahead"):
        assert word not in addition.lower()  # B touches retrieval only, never write actions or consent


def test_the_comment_in_config_b_never_reaches_the_model():
    system = rafeeq.configs()["B"]["system"]
    assert "<!--" not in system and "CHOSEN" not in system and "Rejected" not in system


def test_both_configurations_share_model_and_budgets():
    a, b = rafeeq.configs()["A"], rafeeq.configs()["B"]
    assert a["model"] == b["model"] is None
    assert b["system"].startswith(a["system"] + "\n") and b["system"] != a["system"]


# ---------------------------------------------------------------- B's file moves B only
def test_editing_config_b_changes_b_and_never_a(b_file):
    a_before, b_before = ids()
    a_system = rafeeq.configs()["A"]["system"]
    b_file.write_text("<!-- note -->\nA different single change.", encoding="utf-8")
    a_after, b_after = ids()

    assert a_after == a_before and b_after != b_before
    assert rafeeq.configs()["A"]["system"] == a_system
    assert rafeeq.configs()["B"]["system"] == a_system + "\nA different single change."


def test_a_comment_only_edit_of_config_b_still_marks_b_as_changed(b_file):
    """The file is hashed as a whole: even a note edit gives B a new id, so B results are never mixed by accident."""
    a_before, b_before = ids()
    b_file.write_text(b_file.read_text(encoding="utf-8").replace("<!--", "<!-- edited note", 1), encoding="utf-8")
    assert ids()[0] == a_before and ids()[1] != b_before


def test_a_never_lists_or_reads_config_b():
    assert R.config_files("A") == [] and R.config_files("B") == [rafeeq.CONFIG_B_MD]
    assert rafeeq.CONFIG_B_MD not in R.shared_files()
    _, details = R.fingerprint("A", R.PINNED_MODEL, R.PINNED_JUDGE)
    assert "config_B.md" not in details["files"] and details["config_files"] == {}


def test_documentation_is_not_part_of_any_fingerprint():
    names = {p.name for p in R.shared_files()}
    assert not names & {"EVAL_NOTES_A.md", "PROJECT_README_DRAFT.md", "DEMO_RESULTS.md", "README.md",
                        "IMPLEMENTATION_PLAN_AR.md", "config_B.md"}


# ---------------------------------------------------------------- shared changes move both
def test_changing_the_shared_rules_changes_both(tmp_path, monkeypatch):
    a_before, b_before = ids()
    rules = tmp_path / "AGENTS.md"
    rules.write_text(rafeeq.AGENTS_MD.read_text(encoding="utf-8") + "\n- One more rule.", encoding="utf-8")
    monkeypatch.setattr(rafeeq, "AGENTS_MD", rules)
    a_after, b_after = ids()
    assert a_after != a_before and b_after != b_before


def test_changing_a_shared_file_changes_both(tmp_path, monkeypatch):
    shared = tmp_path / "evals.py"
    shared.write_text("v1", encoding="utf-8")
    monkeypatch.setattr(R, "shared_files", lambda: [shared])
    a_before, b_before = ids()
    shared.write_text("v2: a grader changed", encoding="utf-8")
    a_after, b_after = ids()
    assert a_after != a_before and b_after != b_before


@pytest.mark.parametrize("change", [{"model": "other/agent:free"}, {"judge_model": "other/judge:free"}])
def test_changing_the_agent_or_judge_model_changes_both(change):
    kw = {"model": R.PINNED_MODEL, "judge_model": R.PINNED_JUDGE} | change
    a_before, b_before = ids()
    assert R.fingerprint("A", **kw)[0] != a_before and R.fingerprint("B", **kw)[0] != b_before


def test_changing_a_budget_changes_both(monkeypatch):
    a_before, b_before = ids()
    monkeypatch.setattr(rafeeq, "MAX_STEPS", rafeeq.MAX_STEPS + 1)
    a_after, b_after = ids()
    assert a_after != a_before and b_after != b_before


# ---------------------------------------------------------------- A runs even when B's file is missing or broken
@pytest.mark.parametrize("state", ["missing", "empty", "comment only", "not utf-8"])
def test_a_works_whatever_state_config_b_is_in(tmp_path, monkeypatch, state):
    a_id = ids()[0]
    path = tmp_path / "config_B.md"
    if state == "empty":
        path.write_text("", encoding="utf-8")
    elif state == "comment only":
        path.write_text("<!-- nothing for the model yet -->\n", encoding="utf-8")
    elif state == "not utf-8":
        path.write_bytes(b"\xff\xfe\x00broken")
    monkeypatch.setattr(rafeeq, "CONFIG_B_MD", path)

    assert set(rafeeq._import_configs()) == {"A"}  # importing rafeeq would still succeed, with A only
    assert sha(rafeeq.configs(("A",))["A"]["system"]) == A_SYSTEM_SHA
    assert R.fingerprint("A", R.PINNED_MODEL, R.PINNED_JUDGE)[0] == a_id
    with pytest.raises((OSError, UnicodeDecodeError, ValueError)):
        rafeeq.configs(("B",))  # B fails loudly when it is actually asked for


def test_an_a_batch_runs_with_config_b_missing(tmp_path, monkeypatch):
    from test_eval_runner import FEE, FakeProvider, script_fee
    from test_privacy import FakeMCPServer

    import agentkit as K

    provider = FakeProvider()
    script_fee(provider)
    monkeypatch.setattr(K, "chat", provider)
    monkeypatch.setattr(R, "git_version", lambda: "test")
    monkeypatch.setattr(rafeeq, "CONFIG_B_MD", tmp_path / "absent.md")
    s = R.run_batch("A", [FEE], 1, FakeMCPServer(tmp_path / "mcp"), max_requests=20, out_dir=tmp_path / "evals",
                    log=lambda *_: None)
    assert s["finished"] == 1 and s["config_id"] == R.fingerprint("A", R.PINNED_MODEL, R.PINNED_JUDGE)[0]


def test_b_still_needs_explicit_approval_to_run():
    with pytest.raises(SystemExit, match="not chosen yet"):
        R.check_config("B", allow_unreviewed_b=False)
