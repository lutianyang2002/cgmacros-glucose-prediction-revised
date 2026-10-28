"""Checks of the Claude Code command-line adapter against a fake `claude` program (no real call, no network, no credentials).

Run with `python tests/test_llm_cli_model.py` from the repository root.
"""

from test_llm_factor_search import ScriptedModel

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import llm_factor_search as L  # noqa: E402
from src import llm_cli_model as C  # noqa: E402

FAKE = r'''
import json, os, sys
scenario = os.environ.get("FAKE_SCENARIO", "ok")
if "auth" in sys.argv:
    print(json.dumps({"ok": {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": "pro"},
                      "api": {"loggedIn": True, "authMethod": "api_key", "apiProvider": "firstParty", "subscriptionType": None},
                      "max": {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty", "subscriptionType": "max"}}.get(os.environ.get("FAKE_AUTH", "ok"))))
    sys.exit(0)
prompt = sys.stdin.read()
if os.environ.get("FAKE_COUNTER"):
    with open(os.environ["FAKE_COUNTER"], "a") as handle:
        handle.write("call\n")
seen_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
if scenario == "crash":
    sys.stderr.write("boom"); sys.exit(1)
init = {"type": "system", "subtype": "init", "model": os.environ.get("FAKE_MODEL", "claude-sonnet-5"), "apiKeySource": os.environ.get("FAKE_SOURCE", "none"), "tools": [], "claude_code_version": "0.0.0"}
events = [init]
events.append({"type": "assistant", "message": {"model": os.environ.get("FAKE_ASSISTANT_MODEL", os.environ.get("FAKE_MODEL", "claude-sonnet-5"))}})
if scenario == "limit_event":
    events.append({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": int(os.environ.get("FAKE_RESETS", "1")), "rateLimitType": "five_hour", "isUsingOverage": False}})
if scenario == "overage":
    events.append({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "isUsingOverage": True, "overageStatus": "allowed"}})
if scenario == "warning":
    events.append({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed_warning", "isUsingOverage": False}})
if scenario == "limit_text":
    events.append({"type": "result", "subtype": "success", "is_error": True, "result": "5-hour limit reached - resets 3pm", "usage": {}})
elif scenario != "no_result":
    events.append({"type": "result", "subtype": "success", "is_error": False, "result": os.environ.get("FAKE_REPLY") or json.dumps({"saw_key": seen_key, "prompt_chars": len(prompt), "args": sys.argv[1:]}), "total_cost_usd": 0.01,
                   "usage": {"input_tokens": 10, "output_tokens": 5}})
for e in events:
    print(json.dumps(e))
'''


def _model(directory, **kw):
    fake = Path(directory) / "fake_claude.py"
    fake.write_text(FAKE, encoding="utf-8")
    return C.ClaudeCliModel(command=[sys.executable, str(fake)], timeout_seconds=60, **kw)


class _Env:
    def __init__(self, **values):
        self.values, self.old = values, {}

    def __enter__(self):
        for k, v in self.values.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


CLEAN = {name: None for name in (*C.FORBIDDEN_ENVIRONMENT, *C.NESTED_MARKERS)}


def _expect_pause(model, reason, **env):
    with _Env(**{**CLEAN, **env}):
        try:
            model({"request_id": "x", "prompt": "hello"})
        except L.PauseSearch as pause:
            assert pause.reason == reason, (pause.reason, reason)
            return
    raise AssertionError(f"no pause for {reason}")


def test_a_normal_call_returns_the_reply_and_records_metadata_without_credentials():
    with tempfile.TemporaryDirectory() as d, _Env(**CLEAN, FAKE_SCENARIO="ok"):
        model = _model(d)
        text = model({"request_id": "x", "prompt": "hello world"})
        reply = json.loads(text)
        assert reply["prompt_chars"] == 11 and reply["saw_key"] is False
        args = reply["args"]
        assert "--bare" not in args and "-p" in args and "--no-session-persistence" in args and "--safe-mode" in args
        assert args[args.index("--tools") + 1] == "" and args[args.index("--model") + 1] == "claude-sonnet-5" and "--tools" in args
        assert not {"--fallback-model", "--max-budget-usd", "--betas", "--bare"} & set(args)
        record = model.last
        assert record["model"] == "claude-sonnet-5" and record["assistant_models"] == ["claude-sonnet-5"] and record["api_key_source"] == "none" and record["input_tokens"] == 10 and record["estimated_cost_usd"] == 0.01
        assert "prompt" not in json.dumps(record) and "key" not in json.dumps({k: v for k, v in record.items() if k != "api_key_source"}).lower()


def test_an_api_key_in_the_environment_stops_before_any_call_and_never_reaches_the_child():
    with tempfile.TemporaryDirectory() as d:
        model = _model(d)
        _expect_pause(model, "billing_variable_present", ANTHROPIC_API_KEY="sk-test-not-real", FAKE_SCENARIO="ok")
        with _Env(ANTHROPIC_API_KEY="sk-test-not-real"):
            scrubbed = C.scrubbed_environment()
        assert "ANTHROPIC_API_KEY" not in scrubbed
    with _Env(CLAUDE_CODE_USE_BEDROCK="1", ANTHROPIC_BASE_URL="http://example.invalid"):
        found = C.check_environment()["billing_variables_set"]
    assert found == ["ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK"]


def test_a_nested_claude_code_session_is_refused():
    with tempfile.TemporaryDirectory() as d:
        _expect_pause(_model(d), "nested_session", CLAUDECODE="1", FAKE_SCENARIO="ok")


def test_usage_limit_overage_and_wrong_login_source_pause_without_a_reply():
    with tempfile.TemporaryDirectory() as d:
        model = _model(d)
        _expect_pause(model, "usage_limit_reached", FAKE_SCENARIO="limit_event")
        _expect_pause(model, "usage_limit_reached", FAKE_SCENARIO="limit_text")
        _expect_pause(model, "extra_usage_detected", FAKE_SCENARIO="overage")
        _expect_pause(model, "api_key_source", FAKE_SCENARIO="ok", FAKE_SOURCE="ANTHROPIC_API_KEY")
        with _Env(**CLEAN, FAKE_SCENARIO="warning"):
            assert json.loads(model({"request_id": "x", "prompt": "p"}))["prompt_chars"] == 1        # a warning is not a stop; it is recorded
            assert model.last["rate_limit_events"][0]["status"] == "allowed_warning"
        pinging = _model(d, allowed_key_sources=None)
        with _Env(**CLEAN, FAKE_SCENARIO="ok", FAKE_SOURCE="whatever"):
            pinging({"request_id": "x", "prompt": "p"})
            assert pinging.last["api_key_source"] == "whatever"                                      # the ping only reports an unknown source


def test_a_different_model_pauses_the_search():
    with tempfile.TemporaryDirectory() as d:
        model = _model(d)
        _expect_pause(model, "model_changed", FAKE_SCENARIO="ok", FAKE_MODEL="claude-haiku-other")
        _expect_pause(model, "model_changed", FAKE_SCENARIO="ok", FAKE_ASSISTANT_MODEL="claude-opus-other")
        with _Env(**CLEAN, FAKE_SCENARIO="ok", FAKE_MODEL="claude-haiku-other"):
            free = C.ClaudeCliModel("claude-sonnet-5", command=model.command, pin_model=False)         # the ping only reports
            free({"request_id": "x", "prompt": "p"})
            assert free.last["model"] == "claude-haiku-other"


def test_other_failures_are_ordinary_exceptions_that_the_budget_counts():
    with tempfile.TemporaryDirectory() as d:
        model = _model(d)
        for scenario in ("crash", "no_result"):
            with _Env(**CLEAN, FAKE_SCENARIO=scenario):
                try:
                    model({"request_id": "x", "prompt": "p"})
                except L.PauseSearch:
                    raise AssertionError(f"{scenario} must not be a pause")
                except RuntimeError:
                    continue
                raise AssertionError(f"{scenario} was not an error")


def test_the_login_audit_accepts_only_a_subscription_of_the_expected_type():
    with tempfile.TemporaryDirectory() as d:
        fake = Path(d) / "fake_claude.py"
        fake.write_text(FAKE, encoding="utf-8")
        command = [sys.executable, str(fake)]
        with _Env(FAKE_AUTH="ok"):
            assert C.check_login(command, "pro")["subscriptionType"] == "pro"
        for variant in ("api", "max"):
            with _Env(FAKE_AUTH=variant):
                try:
                    C.check_login(command, "pro")
                except L.PauseSearch as pause:
                    assert pause.reason == "login_not_a_subscription"
                    continue
                raise AssertionError(f"{variant} accepted")


def test_a_pause_stops_the_search_without_counting_and_the_search_resumes_from_the_log():
    memory = L.MemoryConfig(n_islands=2, top_k=3, n_examples=2, temperature=0.05)
    limits = L.BudgetLimits(max_calls=20, proposals_per_call=3, max_proposals=100, max_evaluations=100)
    policy = L.CountingPolicy(failed_call_counts_as_call=True, max_retries_per_call=1, invalid_proposal_counts_as_proposal=True)
    replies = [json.dumps([{"expression": e, "hypothesis": "h"} for e in group]) for group in (['ts_mean("hr", 30)', 'ts_mean("hr", 60)', 'bmi()'], ['native("last")', 'exp_decay("hr", 30)', 'ts_max("hr", 15)'],
                                                                                                     ['ts_mean("calories", 30)', 'bmi()', 'native("diff")'])]
    fitness = lambda e: (sum(map(ord, e)) % 97) / 1000

    class Limited:
        def __init__(self, allowed):
            self.allowed, self.n = allowed, 0

        def __call__(self, request):
            if self.n >= self.allowed:
                raise L.PauseSearch("usage_limit_reached", "test")
            self.n += 1
            return replies[(self.n - 1) % 3]

    with tempfile.TemporaryDirectory() as d:
        full = L.run_search(ScriptedModel(replies), fitness, "A", 3, memory, limits, policy, L.SearchLog(Path(d) / "full.jsonl"), 7, "m", None, "digest")
        path = Path(d) / "part.jsonl"
        first = L.run_search(Limited(2), fitness, "A", 3, memory, limits, policy, L.SearchLog(path), 7, "m", None, "digest")
        assert first["stopped_by"] == "paused:usage_limit_reached" and first["ledger"]["calls"] == 2
        assert [r["type"] for r in L.SearchLog(path).records].count("pause") == 1
        resumed = L.run_search(ScriptedModel(replies[2:] + replies[:2]), fitness, "A", 3, memory, limits, policy, L.SearchLog(path), 7, "m", None, "digest")
        assert resumed["islands"] == full["islands"] and resumed["ledger"] == full["ledger"]           # the same search as without the pause
        try:
            L.run_search(ScriptedModel(replies), fitness, "A", 3, memory, limits, policy, L.SearchLog(path), 7, "m", None, "another digest")
        except RuntimeError:
            return
        raise AssertionError("a changed configuration was accepted on resume")


VALID_REPLY = json.dumps([{"expression": e, "hypothesis": "h"} for e in ('ts_mean("hr", 30)', 'ts_mean("hr", 60)', 'bmi()')])
POLICY = L.CountingPolicy(failed_call_counts_as_call=True, max_retries_per_call=1, invalid_proposal_counts_as_proposal=True)


def _search(model, path, max_calls=6, clock=None, rounds=3, fitness=None):
    memory = L.MemoryConfig(n_islands=2, top_k=3, n_examples=2, temperature=0.05)
    limits = L.BudgetLimits(max_calls=max_calls, proposals_per_call=3, max_proposals=100, max_evaluations=100)
    kwargs = {"clock": clock} if clock else {}
    return L.run_search(model, fitness or (lambda e: (sum(map(ord, e)) % 97) / 1000), "A", rounds, memory, limits, POLICY, L.SearchLog(path), 7, "m", None, "digest", **kwargs)


def _invocations(counter):
    return len(counter.read_text().split()) if counter.exists() else 0


def _identity(ledger):
    """Every call is exactly one of: accepted (parsed), malformed, failed, rejected after the invocation, unknown status."""
    assert ledger["calls"] == ledger["accepted_replies"] + ledger["malformed_replies"] + ledger["failed_calls"] + ledger["rejected_replies"] + ledger["unknown_status_calls"], ledger


def test_a_wrong_model_found_after_the_call_counts_as_a_call_and_its_reply_is_not_used():
    with tempfile.TemporaryDirectory() as d:
        counter, path = Path(d) / "calls.txt", Path(d) / "log.jsonl"
        used = []
        with _Env(**CLEAN, FAKE_SCENARIO="ok", FAKE_MODEL="claude-haiku-other", FAKE_REPLY=VALID_REPLY, FAKE_COUNTER=str(counter)):
            out = _search(_model(d), path, fitness=lambda e: used.append(e) or 0.1)
        ledger = out["ledger"]
        assert out["stopped_by"] == "paused:model_changed" and _invocations(counter) == 1
        assert ledger["calls"] == 1 and ledger["rejected_replies"] == 1 and ledger["accepted_replies"] == 0 and ledger["proposals"] == 0 and ledger["evaluations"] == 0 and not used
        _identity(ledger)
        log = L.SearchLog(path)
        response, pause = log.of("response")[0], log.of("pause")[0]
        assert response["status"] == "rejected" and response["reason"] == "model_changed" and "ts_mean" in response["rejected_text"] and response["call"]["model"] == "claude-haiku-other"
        assert pause["invoked"] is True and not log.of("proposal")                      # the refused text was kept for audit and never judged


def test_overage_found_after_the_call_counts_as_a_call_and_its_reply_is_not_used():
    with tempfile.TemporaryDirectory() as d:
        counter, path = Path(d) / "calls.txt", Path(d) / "log.jsonl"
        with _Env(**CLEAN, FAKE_SCENARIO="overage", FAKE_REPLY=VALID_REPLY, FAKE_COUNTER=str(counter)):
            out = _search(_model(d), path)
        assert out["stopped_by"] == "paused:extra_usage_detected" and _invocations(counter) == 1
        assert out["ledger"]["calls"] == 1 and out["ledger"]["rejected_replies"] == 1 and out["ledger"]["evaluations"] == 0
        _identity(out["ledger"])
        assert L.SearchLog(path).of("pause")[0]["invoked"] is True


def test_repeated_pause_and_resume_cannot_exceed_the_call_limit():
    for scenario, extra in (("ok", {"FAKE_MODEL": "claude-haiku-other"}), ("overage", {})):
        with tempfile.TemporaryDirectory() as d:
            counter, path = Path(d) / "calls.txt", Path(d) / "log.jsonl"
            with _Env(**CLEAN, FAKE_SCENARIO=scenario, FAKE_REPLY=VALID_REPLY, FAKE_COUNTER=str(counter), **extra):
                outcomes = [_search(_model(d), path, max_calls=4) for _ in range(12)]          # twelve fresh processes against the same log
            assert _invocations(counter) == 4, (scenario, _invocations(counter))
            assert outcomes[-1]["stopped_by"] == "max_calls" and outcomes[-1]["ledger"]["calls"] == 4 and outcomes[-1]["ledger"]["rejected_replies"] == 4
            assert outcomes[-1]["ledger"]["evaluations"] == 0
            _identity(outcomes[-1]["ledger"])


def test_an_interrupted_call_is_counted_as_unknown_and_never_sent_again():
    class Interrupted:
        def __call__(self, request):
            raise KeyboardInterrupt

    class Counting:
        def __init__(self):
            self.ids = []

        def __call__(self, request):
            self.ids.append(request["request_id"])
            return VALID_REPLY

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.jsonl"
        try:
            _search(Interrupted(), path)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("the interruption was swallowed")
        log = L.SearchLog(path)
        assert len(log.of("request")) == 1 and not log.of("response")                  # written before the call, no outcome
        interrupted = log.of("request")[0]["request_id"]
        again = Counting()
        out = _search(again, path, max_calls=6)
        assert interrupted not in again.ids                                               # not repeated
        assert out["ledger"]["unknown_status_calls"] == 1 and out["ledger"]["calls"] == len(again.ids) + 1 <= 6
        _identity(out["ledger"])
        assert [r.get("status") for r in L.SearchLog(path).of("response")].count("unknown") == 1
        third = Counting()
        replay = _search(third, path, max_calls=6)                                       # a second resume does not call again and changes nothing
        assert replay["ledger"] == out["ledger"] and not third.ids
        # interruptions in a loop cannot enlarge the budget either
        path2 = Path(d) / "loop.jsonl"
        invocations = []
        for _ in range(20):
            def interrupt(request):
                invocations.append(request["request_id"])
                raise KeyboardInterrupt
            try:
                _search(interrupt, path2, max_calls=5)
            except KeyboardInterrupt:
                pass
        assert len(invocations) == 5 and len(set(invocations)) == 5


def test_a_refusal_before_the_invocation_is_not_a_call():
    with tempfile.TemporaryDirectory() as d:
        counter, path = Path(d) / "calls.txt", Path(d) / "log.jsonl"
        with _Env(**{**CLEAN, "CLAUDECODE": "1"}, FAKE_SCENARIO="ok", FAKE_REPLY=VALID_REPLY, FAKE_COUNTER=str(counter)):
            out = _search(_model(d), path)
        assert out["stopped_by"] == "paused:nested_session" and _invocations(counter) == 0 and out["ledger"]["calls"] == 0
        assert not L.SearchLog(path).of("request") and L.SearchLog(path).of("pause")[0]["invoked"] is False
        with _Env(**CLEAN, FAKE_SCENARIO="ok", FAKE_REPLY=VALID_REPLY, FAKE_COUNTER=str(counter)):
            out = _search(_model(d), path, max_calls=6)                                  # after the problem is gone the search runs its full budget
        assert out["stopped_by"] in ("rounds_completed", "max_calls") and out["ledger"]["calls"] == _invocations(counter) == 6 and out["ledger"]["accepted_replies"] == 6, (out["stopped_by"], out["ledger"], _invocations(counter))
        _identity(out["ledger"])

    class Plain:                                                                          # an adapter without check(): a pause before the call leaves a marker, the slot stays free
        def __init__(self, refuse):
            self.refuse, self.n = refuse, 0

        def __call__(self, request):
            if self.refuse:
                raise L.PauseSearch("login_not_a_subscription", "test")
            self.n += 1
            return VALID_REPLY

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.jsonl"
        for _ in range(5):
            assert _search(Plain(True), path)["ledger"]["calls"] == 0
        ok = Plain(False)
        out = _search(ok, path, max_calls=6)
        assert ok.n == out["ledger"]["calls"] == 6


def test_after_a_usage_limit_the_search_waits_for_the_reset_time():
    now = [1000.0]
    with tempfile.TemporaryDirectory() as d:
        counter, path = Path(d) / "calls.txt", Path(d) / "log.jsonl"
        with _Env(**CLEAN, FAKE_SCENARIO="limit_event", FAKE_RESETS="5000", FAKE_REPLY=VALID_REPLY, FAKE_COUNTER=str(counter)):
            first = _search(_model(d), path, clock=lambda: now[0])
            assert first["stopped_by"] == "paused:usage_limit_reached" and _invocations(counter) == 1 and first["ledger"]["calls"] == 1 and first["ledger"]["rejected_replies"] == 1
            for _ in range(5):
                again = _search(_model(d), path, clock=lambda: now[0])
                assert again["stopped_by"] == "paused:limit_not_reset" and _invocations(counter) == 1       # waiting does not call the model
        assert L.SearchLog(path).of("pause")[0]["resets_at"] == 5000
        now[0] = 5001.0
        with _Env(**CLEAN, FAKE_SCENARIO="ok", FAKE_REPLY=VALID_REPLY, FAKE_COUNTER=str(counter)):
            done = _search(_model(d), path, max_calls=6, clock=lambda: now[0])
        assert done["stopped_by"] in ("rounds_completed", "max_calls") and done["ledger"]["calls"] == _invocations(counter) == 6 and done["ledger"]["rejected_replies"] == 1, (done["stopped_by"], done["ledger"], _invocations(counter))
        _identity(done["ledger"])


def test_a_policy_that_would_not_count_failed_calls_is_refused():
    with tempfile.TemporaryDirectory() as d:
        bad = L.CountingPolicy(failed_call_counts_as_call=False, max_retries_per_call=1, invalid_proposal_counts_as_proposal=True)
        memory = L.MemoryConfig(n_islands=1, top_k=3, n_examples=2, temperature=0.05)
        limits = L.BudgetLimits(max_calls=3, proposals_per_call=3, max_proposals=10, max_evaluations=10)
        try:
            L.run_search(ScriptedModel([VALID_REPLY]), lambda e: 0.1, "A", 1, memory, limits, bad, L.SearchLog(Path(d) / "log.jsonl"), 7, "m")
        except ValueError:
            return
        raise AssertionError("the policy was accepted")


def test_the_command_line_holds_a_post_invocation_pause_for_review():
    from src import llm_search_cli as cli

    with tempfile.TemporaryDirectory() as d:
        log = L.SearchLog(Path(d) / "log.jsonl")
        assert cli.needs_review(log) is None
        log.append("pause", request_id="r", reason="model_changed", detail="", invoked=True, resets_at=None)
        assert cli.needs_review(log) == "model_changed"
        log.append("review", after="model_changed")
        assert cli.needs_review(log) is None
        log.append("pause", request_id="r2", reason="usage_limit_reached", detail="", invoked=True, resets_at=123.0)
        assert cli.needs_review(log) is None                                              # waits for the reset time instead
        log.append("pause", request_id="r3", reason="nested_session", detail="", invoked=False, resets_at=None)
        assert cli.needs_review(log) is None                                              # nothing was called


def test_cli_refusals_precede_context_fitness_and_provider_setup():
    from unittest import TestCase
    from unittest.mock import patch
    from src import llm_search_cli as cli
    case = TestCase()
    scenarios = [
        ("run", False, [], [], 2),
        ("run", True, [{"type": "pause", "reason": "wrong_model", "invoked": True}], [], 3),
        ("run", True, [{"type": "stop", "evaluations": 2}], [], 0),
        ("run", True, [{"type": "request", "request_id": str(i)} for i in range(30)], [], 0),
        ("random", False, [{"type": "stop", "evaluations": 2}], [{"type": "random_stop"}], 2),
        ("random", False, [], [], 2),
    ]
    for command, authorized, records, random_records, expected in scenarios:
        settings = json.loads(cli.SETTINGS_PATH.read_text(encoding="utf-8"))
        settings["authorized"] = authorized
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for kind, rows in (("llm", records), ("random", random_records)):
                (folder / f"{kind}.jsonl").write_text("\n".join(json.dumps({"seq": i, **r}) for i, r in enumerate(rows)))
            with patch.object(cli, "load_settings", return_value=settings), patch.object(cli, "log_path", side_effect=lambda settings, tag, kind="llm": folder / f"{kind}.jsonl"), \
                 patch.object(cli, "load_context", side_effect=AssertionError("context loaded")), patch.object(cli, "build_fitness", side_effect=AssertionError("fitness built")), \
                 patch.object(C, "ClaudeCliModel", side_effect=AssertionError("provider initialized")):
                case.assertEqual(cli.main([command, "--protocol", "A"]), expected)


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
