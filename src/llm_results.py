"""Offline summaries, candidate pools and materialized values from recorded LLM/Random-DSL searches."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src import llm_factor_search as dsl

# What a leaf operator reads (used for the category of an expression and for the identity-proxy disclosure).
OPERATOR_CATEGORY = {
    "ts_mean": "activity", "ts_max": "activity", "exp_decay": "activity",
    "conv_gamma": "meal", "last_meal": "meal", "last_meal_value": "meal", "minutes_since_meal": "meal", "nutrient_ratio": "meal",
    "native": "native glucose state", "bmi": "BMI (subject-level)",
}


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ counts
def status_of(response: dict) -> str:
    return response.get("status") or ("failed" if response.get("failed") else "accepted")


def ledger_from_log(log: dsl.SearchLog) -> dict:
    """Counts of a log read from its records alone, keeping apart the model invocations and the replies that were parsed.

    `model_invocations`  every request that reached the service: responses other than `not_invoked`, plus requests without an outcome (unknown status);
    `replies_returned`   invocations whose reply the adapter accepted (right model, plan, login), parsable or not;
    `accepted_replies`   of those, the replies that parsed into proposals; `malformed_replies` the ones that did not.
    """
    responses = log.of("response")
    answered = {r["request_id"] for r in responses}
    orphans = {r["request_id"] for r in log.of("request")} - answered
    statuses = [status_of(r) for r in responses]
    returned = [r for r in responses if status_of(r) == "accepted"]
    malformed = {r["request_id"] for r in log.of("proposal") if r.get("status") == "malformed_response"}
    return {"model_invocations": len(responses) - statuses.count("not_invoked") + len(orphans), "replies_returned": len(returned), "accepted_replies": len(returned) - len(malformed & {r["request_id"] for r in returned}),
            "malformed_replies": len(malformed), "failed_calls": statuses.count("failed"), "rejected_replies": statuses.count("rejected"), "unknown_status_calls": statuses.count("unknown") + len(orphans)}


# ------------------------------------------------------------------ tables from the logs
def llm_proposals(path) -> pd.DataFrame:
    """One row per logged proposal of the model search, with the call it came from."""
    log = dsl.SearchLog(path)
    requests = {r["request_id"]: r for r in log.of("request")}
    order = {rid: i for i, rid in enumerate(requests)}
    rows = []
    for r in log.of("proposal"):
        if r.get("index", 0) < 0:
            continue
        q = requests[r["request_id"]]
        rows.append({"search": "LLM-Guided", "call_number": order[r["request_id"]] + 1, "request_id": r["request_id"], "round": q["round"], "island": q["island"], "index": r["index"],
                     "log_seq": r["seq"], "raw": r.get("raw", ""), "canonical": r.get("canonical", ""), "status": r["status"], "kind": r.get("kind", ""), "fitness": r.get("fitness", np.nan),
                     "hypothesis": r.get("hypothesis", "")})
    return pd.DataFrame(rows)


def random_draws(path) -> pd.DataFrame:
    log = dsl.SearchLog(path)
    rows = [{"search": "Random-DSL", "call_number": np.nan, "request_id": "", "round": np.nan, "island": np.nan, "index": i, "log_seq": r["seq"], "raw": r.get("raw", ""), "canonical": r.get("canonical", ""),
             "status": r["status"], "kind": r.get("kind", ""), "fitness": r.get("fitness", np.nan), "hypothesis": ""} for i, r in enumerate(log.of("random_draw"))]
    return pd.DataFrame(rows)


def candidate_pool(proposals: pd.DataFrame, size: int = 25) -> pd.DataFrame:
    """Unique valid expressions with a finite training fitness, best first (ties by first appearance), the first `size` of them. Only the training-split fitness of the search is used."""
    evaluated = proposals[proposals["status"].eq("evaluated") & np.isfinite(proposals["fitness"].astype(float))]
    unique = evaluated.sort_values("log_seq").drop_duplicates("canonical", keep="first")
    ranked = unique.sort_values(["fitness", "log_seq"], ascending=[False, True]).reset_index(drop=True)
    ranked.insert(0, "fitness_rank", np.arange(1, len(ranked) + 1))
    ranked["in_pool"] = ranked["fitness_rank"] <= size
    return ranked


# ------------------------------------------------------------------ search-log validation
def validate_llm_log(path, protocol: str, settings: dict, digest: str, memory, limits, policy, fitness_recompute=None) -> dict:
    """Independent checks of one recorded model-search log. Returns {"checks": [(name, passed, detail)], "ledger": {...}}; the log file is not modified."""
    path = Path(path)
    before = sha256_of(path)
    log = dsl.SearchLog(path)
    checks, observations = [], []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    model_id, key_sources = settings["model"]["id"], set(settings["model"]["allowed_key_sources"])
    recs = log.records
    check("sequence numbers are 0..n-1 without gaps", [r["seq"] for r in recs] == list(range(len(recs))), f"{len(recs)} records")
    config = log.of("config")
    check("one config record whose digest equals the configured settings digest", len(config) == 1 and config[0]["digest"] == digest and config[0]["protocol"] == protocol and config[0]["seed"] == settings["seed"],
          config[0]["digest"][:12] if config else "")
    requests, responses = log.of("request"), log.of("response")
    check("one request and one response per call, none without an outcome", len(requests) == len(responses) == len({r["request_id"] for r in requests}) and
          {r["request_id"] for r in requests} == {r["request_id"] for r in responses}, f"{len(requests)} requests, {len(responses)} responses")
    check("every response was accepted by the adapter (no rejected, failed or unknown call)", all(status_of(r) == "accepted" for r in responses))
    check("request prompts hash to the logged digest and pass the data guard", all(hashlib.sha256(q["prompt"].encode()).hexdigest() == q["prompt_sha256"] and bool(dsl.prompt_guard(q["prompt"])) for q in requests))
    calls = [r.get("call", {}) for r in responses]
    check(f"every call reports the pinned model {model_id} in the init event and in every assistant message (the reply text comes from it)",
          all(c.get("model") == model_id and set(c.get("assistant_models", [])) == {model_id} for c in calls))
    extra = sorted({m for c in calls for m in c.get("models_used", [])} - {model_id})
    observations.append(f"the result's modelUsage of every call also lists {extra}: an auxiliary model that Claude Code itself uses; no assistant message and no reply text came from it, and its token use was not recorded"
                        if extra else "no model other than the pinned one appears in modelUsage")
    check("login source is the subscription login (apiKeySource none) on every call", all(c.get("api_key_source") in key_sources for c in calls), str({c.get("api_key_source") for c in calls}))
    events = [e for c in calls for e in c.get("rate_limit_events", [])]
    check("no call used overage (isUsingOverage false in every rate-limit event) and none was rejected",
          all(e.get("isUsingOverage") in (False, None) for e in events) and all(str(e.get("status")).lower() not in ("rejected", "blocked", "limit_reached") for e in events),
          f"{len(events)} rate-limit events, overageStatus {sorted({str(e.get('overageStatus')) for e in events})}")
    check("every call ended with exit code 0, no error flag and result subtype success", all(c.get("exit_code") == 0 and not c.get("is_error") and c.get("result_subtype") == "success" for c in calls))
    check("no tools were available to the model", all(not c.get("tools") for c in calls))
    check("the call count does not exceed the configured cap", len(requests) <= limits.max_calls, f"{len(requests)} <= {limits.max_calls}")

    # re-judge every reply from its text, independently of the fitness
    seen, rebuilt, ignored_extra = {}, [], 0
    counts = {"accepted_replies": 0, "malformed_replies": 0, "proposals": 0, "valid": 0, "invalid": 0, "duplicates": 0, "unique_valid": 0}
    logged = {(r["request_id"], r["index"]): r for r in log.of("proposal")}
    mismatches = []
    for q in requests:
        text = next(r for r in responses if r["request_id"] == q["request_id"])["text"]
        try:
            items = dsl.parse_response(text, limits.proposals_per_call)
            raw_count = len(json.loads(text[text.find("["):text.rfind("]") + 1]))
            ignored_extra += max(0, raw_count - limits.proposals_per_call)
            counts["accepted_replies"] += 1
        except dsl.ResponseError:
            counts["malformed_replies"] += 1
            if (q["request_id"], -1) not in logged:
                mismatches.append((q["request_id"], "malformed reply without record"))
            continue
        for index, item in enumerate(items):
            counts["proposals"] += 1
            record = logged.get((q["request_id"], index))
            try:
                node = dsl.parse_expression(item["expression"])
                dsl.check_causal(node)
                expr = dsl.canonical(node)
            except dsl.DslError:
                counts["invalid"] += 1
                verdict = ("rejected", "")
            else:
                counts["valid"] += 1
                if expr in seen:
                    counts["duplicates"] += 1
                    verdict = ("duplicate", expr)
                else:
                    seen[expr] = q["request_id"]
                    counts["unique_valid"] += 1
                    verdict = ("evaluated", expr)
            if record is None or record["status"] not in (verdict[0], "unevaluable") or (verdict[1] and record.get("canonical") != verdict[1]):
                mismatches.append((q["request_id"], index, verdict, None if record is None else record["status"]))
            rebuilt.append((q["request_id"], index, verdict))
    check("every logged proposal verdict (valid, duplicate, evaluated) is reproduced from the reply text alone", not mismatches and len(logged) == len(rebuilt) + counts["malformed_replies"], str(mismatches[:3]))
    evaluated = [r for r in log.of("proposal") if r["status"] == "evaluated"]
    check("no canonical expression was evaluated twice", len({r["canonical"] for r in evaluated}) == len(evaluated), f"{len(evaluated)} evaluations")
    check("every evaluated expression has a finite fitness", all(np.isfinite(r["fitness"]) for r in evaluated))
    stop = log.of("stop")
    ledger = {k: stop[-1][k] for k in dsl.Ledger.FIELDS} if stop else {}
    rebuilt_ledger = {"calls": len(requests), **counts, "evaluations": len(evaluated)}
    check("the final ledger equals the counts rebuilt from the log", bool(stop) and all(ledger[k] == v for k, v in rebuilt_ledger.items() if k in ledger) and ledger["failed_calls"] == ledger["rejected_replies"] == ledger["unknown_status_calls"] == 0,
          str({k: (ledger.get(k), v) for k, v in rebuilt_ledger.items() if ledger.get(k) != v}))
    check("stop reason", bool(stop) and stop[-1]["reason"] == "max_calls", stop[-1]["reason"] if stop else "no stop record")
    check("proposals within the approved budget", counts["proposals"] <= limits.max_proposals and len(evaluated) <= limits.max_evaluations)

    # a replay needs neither a model nor a fitness and appends nothing to the log (done on a copy)
    with tempfile.TemporaryDirectory() as folder:
        copy = Path(folder) / path.name
        shutil.copyfile(path, copy)
        def no_fitness(expr):
            raise AssertionError("the replay evaluated an expression")
        replay = dsl.run_search(dsl.ReplayModel(dsl.SearchLog(copy)), no_fitness, protocol, settings["budget"]["rounds_per_island"], memory, limits, policy, dsl.SearchLog(copy), settings["seed"],
                                config[0]["model_id"] if config else "replay", None, digest)
        unchanged = sha256_of(copy) == before
    check("replay from the log alone (no model, no fitness) reproduces the ledger and appends no record", unchanged and replay["ledger"] == ledger, replay["stopped_by"])
    if fitness_recompute is not None:
        diffs = np.array([abs(fitness_recompute(r["canonical"]) - r["fitness"]) for r in evaluated])
        check("the training-split fitness recomputed from the training windows reproduces every logged fitness", diffs.max() < 1e-9, f"max |difference| {diffs.max():.2e} over {len(diffs)} expressions")
    return {"protocol": protocol, "checks": checks, "observations": observations, "ledger": ledger, "rebuilt_ledger": rebuilt_ledger, "extra_proposals_ignored_by_the_parser": ignored_extra, "log_sha256": before,
            "tokens": {"input": int(sum(c.get("input_tokens", 0) for c in calls)), "output": int(sum(c.get("output_tokens", 0) for c in calls))},
            "seconds": float(sum(c.get("seconds", 0) for c in calls)), "claude_code_versions": sorted({str(c.get("claude_code_version")) for c in calls})}


def validate_random_log(path, protocol: str, seed: int, llm_evaluations: int, fitness_recompute=None) -> dict:
    """The control's log against the sampler: same seed, same draw order, same duplicate rule, as many evaluations as the model search."""
    path = Path(path)
    before = sha256_of(path)
    log = dsl.SearchLog(path)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    sampler, seen, expected, draws, duplicates = dsl.RandomDsl(seed), set(), [], 0, 0
    while len(expected) < llm_evaluations:
        draws += 1
        node = dsl.parse_expression(sampler.sample())
        dsl.check_causal(node)
        expr = dsl.canonical(node)
        if expr in seen:
            duplicates += 1
            continue
        seen.add(expr)
        expected.append(expr)
    drawn = log.of("random_draw")
    stop = log.of("random_stop")[-1]
    check("sequence numbers are 0..n-1 without gaps", [r["seq"] for r in log.records] == list(range(len(log.records))))
    check("the logged expressions are exactly the sampler's output for the seed, in the same order", [r.get("canonical") for r in drawn] == expected, f"seed {seed}")
    check("evaluations equal the model search's evaluations", stop["evaluations"] == llm_evaluations == len(drawn), f"{stop['evaluations']} vs {llm_evaluations}")
    check("draws, duplicates and the unique count equal the independent re-count", (stop["draws"], stop["duplicates"], stop["unique_valid"]) == (draws, duplicates, len(expected)),
          f"draws {stop['draws']}/{draws}, duplicates {stop['duplicates']}/{duplicates}")
    check("no canonical expression was evaluated twice", len({r["canonical"] for r in drawn}) == len(drawn))
    check("every evaluated expression has a finite fitness", all(r["status"] == "evaluated" and np.isfinite(r["fitness"]) for r in drawn))
    if fitness_recompute is not None:
        diffs = np.array([abs(fitness_recompute(r["canonical"]) - r["fitness"]) for r in drawn])
        check("the training-split fitness recomputed from the training windows reproduces every logged fitness", diffs.max() < 1e-9, f"max |difference| {diffs.max():.2e} over {len(diffs)} expressions")
    return {"protocol": protocol, "checks": checks, "ledger": {k: stop[k] for k in ("draws", "invalid", "duplicates", "unique_valid", "evaluations", "unevaluable")}, "log_sha256": before}


# ------------------------------------------------------------------ expressions: traits and values
def expression_traits(expression: str) -> dict:
    node = dsl.parse_expression(expression)
    leaves, stack = [], [node]
    while stack:
        name, args = stack.pop()
        if name in ("interact", "ratio"):
            stack.extend(a for a in args if isinstance(a, tuple))
        else:
            leaves.append(name)
    categories = sorted({OPERATOR_CATEGORY[n] for n in leaves})
    low, high = dsl.check_causal(node)
    return {"top_operator": node[0], "leaf_operators": " | ".join(sorted(leaves)), "categories": " x ".join(categories), "uses_bmi": "bmi" in leaves, "uses_native_state": dsl.uses_native(node),
            "reads_back_minutes": -low}


def materialize(expressions: list, table: pd.DataFrame, minutes: dict, events: dict, bmi: pd.Series, no_events: pd.DataFrame) -> pd.DataFrame:
    """Values of DSL expressions at every window of `table` (training and validation windows of one protocol) with the causal operators; minutes and events must end at each subject's last admitted origin."""
    from src.llm_fitness import build_contexts

    contexts = build_contexts(table, minutes, events, bmi, no_events)
    nodes = {e: dsl.parse_expression(e) for e in expressions}
    out = {e: np.full(len(table), np.nan) for e in expressions}
    for rows, context in contexts.values():
        for e, node in nodes.items():
            out[e][rows] = dsl.evaluate_expression(node, context)
    return pd.DataFrame(out, index=table.index)
