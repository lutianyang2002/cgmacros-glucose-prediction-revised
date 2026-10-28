"""Command line of the LLM factor search (run from an independent terminal, never from inside a Claude Code session).

    python -m src.llm_search_cli check-login              login and environment checks, no model call
    python -m src.llm_search_cli ping                     exactly one real call through the adapter (reports the login source, model, tokens, rate-limit events)
    python -m src.llm_search_cli verify [--protocol A]    settings invariants (budget, pinned model, no fallback) and, with --protocol, the boundary checks of the inner splits; no model call
    python -m src.llm_search_cli check-data --protocol A  builds the training-split fitness and scores three hand-written expressions; no model call
    python -m src.llm_search_cli run --protocol A         the search under the budget of the settings file; requires "authorized": true
    python -m src.llm_search_cli status --protocol A      ledger of the log; tells whether the search can be resumed
    python -m src.llm_search_cli random --protocol A      Random-DSL control with the number of evaluations of the finished model search; no model call

The search resumes from its append-only log. It pauses (exit code 3) when the usage limit is reached, when usage beyond the plan is reported, when the login source or the model is not the
pinned one, or when the login is not the subscription login. A pause that was found only after the model had been invoked (wrong model, overage, login source, usage limit) has already
been counted as a call; a pause before the invocation (nested session, billing variable, login) was not. A request that was written to the log but has no outcome (interrupted run) is
counted as a call on resume and is never sent again. After a usage limit the search waits for the reset time; after any other post-invocation pause it needs `--resume-after-review`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src import llm_factor_search as dsl
from src.factor_data import PROTOCOLS, load_context
from src.paths import PROJECT_ROOT

FACTOR_DIR = PROJECT_ROOT / "results" / "factor_discovery"
SETTINGS_PATH = FACTOR_DIR / "llm_search_settings.json"
CHECK_EXPRESSIONS = ['ts_mean("hr", 60)', 'interact(conv_gamma("lunch", "carbs", 30, 2), bmi())', 'native("slope_1h")']


def load_settings(path: Path = SETTINGS_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def settings_digest(settings: dict) -> str:
    """Digest of the fields that change what the search does (not of the authorization flag or of notes)."""
    keys = ("model", "budget", "counting_policy", "memory", "seed", "fitness")
    return hashlib.sha256(json.dumps({k: settings[k] for k in keys}, sort_keys=True).encode()).hexdigest()


def search_objects(settings: dict) -> tuple:
    b, m, c = settings["budget"], settings["memory"], settings["counting_policy"]
    return (dsl.MemoryConfig(n_islands=b["n_islands"], top_k=m["top_k"], n_examples=m["n_examples"], temperature=m["temperature"]),
            dsl.BudgetLimits(max_calls=b["max_calls"], proposals_per_call=b["proposals_per_call"], max_proposals=b["max_proposals"], max_evaluations=b["max_evaluations"]),
            dsl.CountingPolicy(failed_call_counts_as_call=c["failed_call_counts_as_call"], max_retries_per_call=c["max_retries_per_call"],
                               invalid_proposal_counts_as_proposal=c["invalid_proposal_counts_as_proposal"]))


def build_fitness(ns: dict, name: str, settings: dict):
    """Fitness over the training split of one protocol: inner splits, per-split baselines and residuals, DSL contexts (nothing beyond the training period is evaluated)."""
    from src.llm_fitness import ExpressionFitness, build_contexts, group_splits, time_split

    fit = settings["fitness"]
    frame = ns["tables"][name]
    frame = frame[frame["split"].eq("train")].reset_index(drop=True)
    flag = pd.read_parquet(ns.get("factor_dir", FACTOR_DIR) / ("primitives_A.parquet" if name == PROTOCOLS["A"] else "primitives_B.parquet"), columns=[*ns["WINDOW_KEY"], "split", "native_missing"])
    frame = frame.merge(flag, on=[*ns["WINDOW_KEY"], "split"], validate="one_to_one")
    lags = ns["lag_columns"][name]
    _, minutes, events = ns["get_inputs"](name)
    no_events = pd.DataFrame({"timestamp": pd.to_datetime([]), **{c: [] for c in ns["EVENT_COLUMNS"]}})
    if name == PROTOCOLS["A"]:
        scope = pd.read_csv(ns["DATA_QUALITY_DIR"] / "split_scope_v1.csv", parse_dates=["start_time", "protocol_a_train_end_time"]).set_index("subject_id")
        splits = [time_split(frame, scope["start_time"], scope["protocol_a_train_end_time"], lags, fit["protocol_A_share"])]
        contexts = build_contexts(frame, minutes, events, ns["BMI"], no_events, scope["protocol_a_train_end_time"])
    else:
        splits = group_splits(frame, lags, fit["protocol_B_folds"], fit["protocol_B_inner_folds"])
        contexts = build_contexts(frame, minutes, events, ns["BMI"], no_events)
    return ExpressionFitness(frame, splits, contexts, fit["alpha"], fit["min_defined"], fit["min_held_windows"]), frame


def log_path(settings: dict, tag: str, kind: str = "llm", factor_dir=None) -> Path:
    return (FACTOR_DIR if factor_dir is None else Path(factor_dir)) / settings["paths"][f"{kind}_log_{tag}"]


def verify_settings(settings: dict) -> list:
    """Invariants of the settings file that protect the budget and the billing path; every entry is (check, passed)."""
    b, c, m = settings["budget"], settings["counting_policy"], settings["model"]
    from src.llm_cli_model import ClaudeCliModel

    command = ClaudeCliModel(m["id"], command=["claude"]).build_command()
    return [("model id is pinned to claude-sonnet-5", m["id"] == "claude-sonnet-5"), ("only the subscription login source is accepted", m["allowed_key_sources"] == ["none"]),
            ("expected subscription is pro", m["expected_subscription"] == "pro"),
            ("command has no --bare, --fallback-model, --max-budget-usd or --betas", not {"--bare", "--fallback-model", "--max-budget-usd", "--betas"} & set(command)),
            ("command disables all tools and session persistence", command[command.index("--tools") + 1] == "" and "--no-session-persistence" in command),
            ("calls per protocol = islands x rounds = max_calls = 30", b["n_islands"] * b["rounds_per_island"] == b["max_calls"] == 30),
            ("proposals = calls x proposals per call = max_proposals = max_evaluations = 90", b["max_calls"] * b["proposals_per_call"] == b["max_proposals"] == b["max_evaluations"] == 90),
            ("both protocols together stay within 60 calls", 2 * b["max_calls"] <= b["total_calls_both_protocols"] == 60),
            ("a failed call counts as a call and shares the call limit", c["failed_call_counts_as_call"] is True and c["max_retries_per_call"] <= 1),
            ("memory: top 5, two examples, temperature 0.01", settings["memory"] == {"top_k": 5, "n_examples": 2, "temperature": 0.01}),
            ("seed 42 for the search and the control", settings["seed"] == 42 and settings["random_dsl"]["seed"] == 42),
            ("fitness: alpha 1.0, 75/25 time split in A, 3-fold GroupKFold in B", settings["fitness"]["alpha"] == 1.0 and settings["fitness"]["protocol_A_share"] == 0.75 and settings["fitness"]["protocol_B_folds"] == 3)]


class Progress:
    """Prints one line per real call (request, model, tokens, plan status); the prompt and the reply are only in the log."""

    def __init__(self, model):
        self.model = model

    @property
    def last(self):
        return self.model.last

    def check(self):
        self.model.check()

    def _line(self, request, outcome):
        r = self.model.last or {}
        status = (r.get("rate_limit_events") or [{}])[-1]
        print(f"{time.strftime('%H:%M:%S')} {request['request_id']}: {outcome} {r.get('model')} in={r.get('input_tokens')} out={r.get('output_tokens')} {r.get('seconds')} s "
              f"plan={status.get('status')} overage={status.get('isUsingOverage')}", flush=True)

    def __call__(self, request):
        try:
            text = self.model(request)
        except dsl.PauseSearch as pause:
            if pause.invoked:
                self._line(request, f"REJECTED({pause.reason}), counted as a call,")
            raise
        except Exception as error:      # noqa: BLE001
            self._line(request, f"FAILED({type(error).__name__}), counted as a call,")
            raise
        self._line(request, "ok")
        return text


def needs_review(log) -> str | None:
    """Reason of a post-invocation pause that is waiting for a human decision (None if the search may continue). A usage limit with a known reset time waits for that time instead."""
    marks = [r for r in log.records if r["type"] in ("pause", "review", "response", "stop")]
    for r in reversed(marks):
        if r["type"] == "review" or r["type"] == "stop":
            return None
        if r["type"] == "pause":
            if r.get("invoked") and not (r["reason"] == "usage_limit_reached" and r.get("resets_at")):
                return r["reason"]
            return None
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check-login", "ping", "verify", "check-data", "run", "status", "random"])
    parser.add_argument("--protocol", choices=["A", "B"])
    parser.add_argument("--settings", default=str(SETTINGS_PATH))
    parser.add_argument("--resume-after-review", action="store_true", help="continue after a pause that was found after a model call (wrong model, overage, login source); the call stays counted")
    args = parser.parse_args(argv)
    settings = load_settings(Path(args.settings))
    from src.llm_cli_model import ClaudeCliModel, check_environment, check_login, resolve_claude

    if args.command == "verify":
        checks = verify_settings(settings)
        for text, ok in checks:
            print(("PASS " if ok else "FAIL ") + text)
        print(f"authorized flag: {settings.get('authorized')}")
        failed = [t for t, ok in checks if not ok]
        if not args.protocol:
            return 1 if failed else 0
        from src.llm_fitness import boundary_report_groups, boundary_report_time

        ns = load_context(include_factors=False)
        name = PROTOCOLS[args.protocol]
        fitness, frame = build_fitness(ns, name, settings)
        if args.protocol == "A":
            scope = pd.read_csv(ns["DATA_QUALITY_DIR"] / "split_scope_v1.csv", parse_dates=["start_time", "protocol_a_train_end_time"]).set_index("subject_id")
            report = boundary_report_time(frame, fitness.splits[0], scope["start_time"], scope["protocol_a_train_end_time"], ns["lag_columns"][name], settings["fitness"]["protocol_A_share"])
            bad = [k for k in ("rows_in_both", "fit_rows_with_label_after_cut", "held_rows_with_history_before_cut", "held_rows_with_label_before_cut", "held_rows_with_label_after_training_end",
                               "overlapping_supports_between_oof_blocks") if report[k]]
        else:
            scope = pd.read_csv(ns["DATA_QUALITY_DIR"] / "split_scope_v1.csv")
            training = set(scope.loc[scope["protocol_b_group"].eq("train"), "subject_id"])
            report = boundary_report_groups(frame, fitness.splits, training)
            bad = [k for k in ("subjects_outside_the_training_subjects",) if report[k]] + (["rows_in_both"] if any(report["rows_in_both"]) else []) +\
                  (["fit_subjects_overlapping_held_subjects"] if any(report["fit_subjects_overlapping_held_subjects"]) else []) + ([] if report["held_subjects_disjoint_across_folds"] else ["folds"])
        print(json.dumps(report, indent=1, default=str))
        print("BOUNDARY CHECK " + ("FAILED: " + ", ".join(bad) if bad or failed else "PASSED"))
        return 1 if bad or failed else 0
    if args.command == "check-login":
        command = resolve_claude()
        print(json.dumps({"claude_command": command[0], "environment": check_environment(), "login": check_login(command, settings["model"]["expected_subscription"])}, indent=1))
        return 0
    if args.command == "ping":
        model = ClaudeCliModel(settings["model"]["id"], timeout_seconds=settings["model"]["timeout_seconds"], expected_subscription=settings["model"]["expected_subscription"], allowed_key_sources=None,
                               pin_model=False)
        pre = model.preflight()
        text = model({"request_id": "ping", "prompt": "Reply with exactly one word: OK"})
        record = {"preflight": pre, "call": dict(model.last), "reply": text[:80], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        (FACTOR_DIR / "llm_search").mkdir(exist_ok=True)
        (FACTOR_DIR / "llm_search" / "ping.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        print(json.dumps(record, indent=1, default=str))
        return 0
    if not args.protocol:
        parser.error("--protocol is required")
    name = PROTOCOLS[args.protocol]
    memory, limits, policy = search_objects(settings)
    if args.command == "status":
        path = log_path(settings, args.protocol)
        if not path.exists():
            print("no log: the search has not started")
            return 0
        log = dsl.SearchLog(path)
        last = [r for r in log.records if r["type"] in ("stop", "pause", "budget")]
        ledger = {k: v for k, v in (last[-1] if last else {}).items() if k not in ("seq",)}
        from src.llm_results import ledger_from_log

        counts = ledger_from_log(log)
        print(json.dumps({"model_invocations_in_log": counts["model_invocations"], "replies_returned_by_the_model": counts["replies_returned"],
                          "accepted_replies_parsed": counts["accepted_replies"], "malformed_replies": counts["malformed_replies"], "failed_calls": counts["failed_calls"],
                          "rejected_replies": counts["rejected_replies"], "unknown_status_calls": counts["unknown_status_calls"], "last_record": ledger,
                          "waiting_for_review": needs_review(log), "can_start_another_call": counts["model_invocations"] < limits.max_calls}, indent=1))
        return 0
    if args.command == "run":
        if not settings.get("authorized"):
            print("The settings file has \"authorized\": false. The search needs your explicit authorization.")
            return 2
        log = dsl.SearchLog(log_path(settings, args.protocol))
        from src.llm_results import ledger_from_log
        counts = ledger_from_log(log)
        proposals = [r for r in log.of("proposal") if r.get("index", -1) >= 0]
        if (log.of("stop") or counts["model_invocations"] >= limits.max_calls or len(proposals) >= limits.max_proposals
                or sum(r["status"] in ("evaluated", "unevaluable") for r in proposals) >= limits.max_evaluations):
            print("Search is complete or its budget is exhausted; no calculation started.")
            return 0
        if dsl._limit_wait(log, time.time) is not None:
            print("Search is waiting for the recorded usage-limit reset; no calculation started.")
            return 3
        waiting = needs_review(log)
        if waiting and not args.resume_after_review:
            print(f"NOT STARTED: the search paused on {waiting!r} after a model call (the call is counted). Look at the log, then run again with --resume-after-review.")
            return 3
        started = log.of("config")
        if started and started[0]["digest"] != settings_digest(settings):
            raise ValueError("settings differ from the recorded search")
        model = ClaudeCliModel(settings["model"]["id"], timeout_seconds=settings["model"]["timeout_seconds"], expected_subscription=settings["model"]["expected_subscription"],
                               allowed_key_sources=tuple(settings["model"]["allowed_key_sources"]), pin_model=True)
        try:
            model.preflight()
        except dsl.PauseSearch as pause:
            print(f"NOT STARTED: {pause}")
            return 3
        if waiting:
            log.append("review", after=waiting, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    if args.command == "random":
        llm = dsl.SearchLog(log_path(settings, args.protocol))
        stops = llm.of("stop")
        if not stops or llm.records[-1]["type"] != "stop":
            print("The model search has not finished: the number of evaluations to match is not fixed yet.")
            return 2
        target = stops[-1]["evaluations"]
        random_log = dsl.SearchLog(log_path(settings, args.protocol, "random"))
        if random_log.records:
            print("Random-DSL log is not empty; choose a new output path in the settings.")
            return 2
    ns = load_context(include_factors=False)
    fitness, frame = build_fitness(ns, name, settings)
    print(f"{name}: {len(frame)} training windows, {frame['subject'].nunique()} subjects, {len(fitness.splits)} inner split(s), "
          f"{[(len(s.train), len(s.held)) for s in fitness.splits]} fit/held-out rows")
    if args.command == "check-data":
        for e in CHECK_EXPRESSIONS:
            t = time.time()
            print(f"{e}: gain {fitness(e)} ({time.time() - t:.2f} s)")
        return 0
    if args.command == "run":
        out = dsl.run_search(Progress(model), fitness, name, settings["budget"]["rounds_per_island"], memory, limits, policy, log, settings["seed"],
                             f"claude-cli:{settings['model']['id']}", None, settings_digest(settings))
        print(json.dumps({"stopped_by": out["stopped_by"], "ledger": out["ledger"]}, indent=1))
        return 3 if out["stopped_by"].startswith("paused") else 0
    if args.command == "random":
        count = dsl.run_random_control(fitness, name, target, settings["random_dsl"]["seed"], random_log)
        print(dsl.budget_report(stops[-1], count).to_string(index=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
