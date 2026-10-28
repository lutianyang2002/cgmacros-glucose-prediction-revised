"""Checks of the offline analysis of search logs (counts, audit, pools, traits) on small synthetic logs. No model, no data files.

Run with `python tests/test_llm_results.py` from the repository root.
"""

import json
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import llm_factor_search as L  # noqa: E402
from src import llm_results as R  # noqa: E402

REPLIES = [json.dumps([{"expression": e, "hypothesis": "h"} for e in group]) for group in (
    ['ts_mean("hr", 30)', 'ts_mean("hr", 60)', 'bmi()'], ['native("last")', 'exp_decay("hr", 30)', 'ts_max("hr", 15)'], ['ts_mean("calories", 30)', 'bmi()', 'native("diff")'])]
MEMORY = L.MemoryConfig(n_islands=2, top_k=3, n_examples=2, temperature=0.05)
LIMITS = L.BudgetLimits(max_calls=6, proposals_per_call=3, max_proposals=100, max_evaluations=100)
POLICY = L.CountingPolicy(failed_call_counts_as_call=True, max_retries_per_call=1, invalid_proposal_counts_as_proposal=True)
SETTINGS = {"model": {"id": "pinned-model", "allowed_key_sources": ["none"]}, "seed": 7, "budget": {"rounds_per_island": 3}}
FITNESS = lambda e: (sum(map(ord, e)) % 97) / 1000


class Fake:
    """Scripted replies with the metadata of a call that went right (or the one given)."""

    def __init__(self, replies, **changes):
        self.replies, self.n, self.changes, self.last = replies, 0, changes, None

    def __call__(self, request):
        self.n += 1
        self.last = {"model": "pinned-model", "assistant_models": ["pinned-model"], "models_used": ["pinned-model"], "api_key_source": "none", "exit_code": 0, "is_error": False,
                     "result_subtype": "success", "tools": [], "rate_limit_events": [{"status": "allowed", "isUsingOverage": False}], "input_tokens": 3, "output_tokens": 5, "seconds": 1.0,
                     **self.changes}
        return self.replies[(self.n - 1) % len(self.replies)]


def _run(path, replies, **changes):
    return L.run_search(Fake(replies, **changes), FITNESS, "A", 3, MEMORY, LIMITS, POLICY, L.SearchLog(path), 7, "m", None, "digest")


def _audit(path, **kw):
    return R.validate_llm_log(path, "A", SETTINGS, "digest", MEMORY, LIMITS, POLICY, **kw)


def test_counts_keep_invocations_apart_from_parsed_replies():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.jsonl"
        out = _run(path, [REPLIES[0], "not json at all", REPLIES[1]])
        counts = R.ledger_from_log(L.SearchLog(path))
        assert counts["model_invocations"] == out["ledger"]["calls"] == 6
        assert counts["replies_returned"] == 6 and counts["accepted_replies"] == out["ledger"]["accepted_replies"] == 4 and counts["malformed_replies"] == out["ledger"]["malformed_replies"] == 2
        assert counts["failed_calls"] == counts["rejected_replies"] == counts["unknown_status_calls"] == 0


def test_a_clean_log_passes_the_audit_and_a_tampered_one_does_not():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.jsonl"
        _run(path, REPLIES)
        before = path.read_bytes()
        report = _audit(path, fitness_recompute=FITNESS)
        failed = [c for c in report["checks"] if not c["passed"]]
        assert not failed, failed
        assert path.read_bytes() == before                                           # the audit changes nothing
        assert report["ledger"]["calls"] == 6 and report["rebuilt_ledger"]["unique_valid"] == report["ledger"]["unique_valid"]
        wrong = _audit(path, fitness_recompute=lambda e: FITNESS(e) + 1e-6)
        assert [c["check"] for c in wrong["checks"] if not c["passed"]] == ["the training-split fitness recomputed from the training windows reproduces every logged fitness"]
    with tempfile.TemporaryDirectory() as d:                                          # another model in the call metadata
        path = Path(d) / "log.jsonl"
        _run(path, REPLIES, model="other-model")
        assert any(not c["passed"] and "pinned model" in c["check"] for c in _audit(path)["checks"])
    with tempfile.TemporaryDirectory() as d:                                          # overage in a rate-limit event
        path = Path(d) / "log.jsonl"
        _run(path, REPLIES, rate_limit_events=[{"status": "allowed", "isUsingOverage": True}])
        assert any(not c["passed"] and "overage" in c["check"] for c in _audit(path)["checks"])
    with tempfile.TemporaryDirectory() as d:                                          # an edited log record
        path = Path(d) / "log.jsonl"
        _run(path, REPLIES)
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            record = json.loads(line)
            if record["type"] == "proposal" and record["status"] == "evaluated":
                record["status"] = "duplicate"
                lines[i] = json.dumps(record, sort_keys=True)
                break
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert any(not c["passed"] for c in _audit(path)["checks"])


def test_the_random_control_is_audited_against_the_sampler():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "random.jsonl"
        count = L.run_random_control(FITNESS, "A", 25, 11, L.SearchLog(path))
        report = R.validate_random_log(path, "A", 11, 25, fitness_recompute=FITNESS)
        assert all(c["passed"] for c in report["checks"]), report["checks"]
        assert report["ledger"]["draws"] == count["draws"]
        assert not all(c["passed"] for c in R.validate_random_log(path, "A", 12, 25)["checks"])           # another seed is not the same draw sequence


def test_the_pool_orders_by_training_fitness_and_ties_by_first_appearance():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.jsonl"
        _run(path, REPLIES)
        table = R.llm_proposals(path)
        pool = R.candidate_pool(table, size=4)
        assert pool["canonical"].is_unique and pool["in_pool"].sum() == 4 and list(pool["fitness_rank"]) == list(range(1, len(pool) + 1))
        assert (pool["fitness"].diff().dropna() <= 0).all()
        ties = pool[pool["fitness"].duplicated(keep=False)]
        for _, group in ties.groupby("fitness"):
            assert list(group["log_seq"]) == sorted(group["log_seq"])
        assert set(pool["canonical"]) == set(table.loc[table["status"].eq("evaluated"), "canonical"])


def test_traits_flag_the_subject_level_variable_and_the_categories():
    t = R.expression_traits('interact(conv_gamma("lunch", "carbs", 30, 2), bmi())')
    assert t["uses_bmi"] and t["categories"] == "BMI (subject-level) x meal" and t["top_operator"] == "interact"
    u = R.expression_traits('ratio(ts_mean("hr", 30), native("last"))')
    assert not u["uses_bmi"] and u["uses_native_state"] and u["categories"] == "activity x native glucose state" and u["reads_back_minutes"] >= 74


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print("PASS", name)
        except Exception as error:      # noqa: BLE001
            import traceback
            failures += 1
            print("FAIL", name, type(error).__name__, error)
            traceback.print_exc()
    sys.exit(1 if failures else 0)
