"""Offline checks of the constrained LLM factor search: DSL validation, causality, prompt isolation, budget, log replay, Random-DSL (no model, no network, no project data).

Run with `python tests/test_llm_factor_search.py` from the repository root.
"""

import ast
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import llm_factor_search as L  # noqa: E402
from src.factor_construction import CandidateContext, conv_gamma, ts_mean  # noqa: E402

VALID = {
    "single": 'ts_mean("hr", 30)',
    "interaction": 'interact(conv_gamma("lunch", "carbs", 30, 2), bmi())',
    "historical": 'exp_decay("calories", 60)',
    "ratio": 'ratio(last_meal("fiber"), last_meal("carbs"))',
    "native": 'interact(minutes_since_meal("any"), native("slope_1h"))',
}


def _kind(text):
    try:
        L.check_causal(L.parse_expression(text))
    except L.DslError as error:
        return error.kind
    return None


def test_valid_expressions_are_accepted_with_offsets_inside_the_history():
    for text in VALID.values():
        node = L.parse_expression(text)
        lo, hi = L.check_causal(node)
        assert hi == 0 and -120 <= lo <= 0
    assert L.source_offsets(L.parse_expression('ts_mean("hr", 30)')) == (-30, 0)
    assert L.uses_native(L.parse_expression(VALID["native"])) and not L.uses_native(L.parse_expression(VALID["single"]))


def test_unsafe_or_out_of_grammar_proposals_are_rejected_with_a_reason():
    cases = {
        'ts_mean("hr", 240)': "parameter_out_of_range",                       # beyond the 120-minute history
        'ts_mean("hr", -30)': "parameter_out_of_range",                       # a negative window would look forward
        'ts_mean("glucose", 30)': "parameter_out_of_range",                   # field outside the whitelist
        'ts_mean("target", 30)': "forbidden_term",
        'df.rolling(30, center=True).mean()': "forbidden_term",
        'shift(ts_mean("hr", 30), -5)': "forbidden_term",
        'groupby("subject")': "forbidden_term",
        '__import__("os").system("dir")': "forbidden_term",
        'open("x.csv")': "forbidden_term",
        'lambda x: x': "forbidden_term",
        'np.mean(1)': "forbidden_construct",
        'ts_mean("hr", 30) + ts_mean("hr", 60)': "forbidden_construct",
        '[ts_mean("hr", 30) for _ in range(3)]': "forbidden_construct",
        'log(ts_mean("hr", 30))': "unknown_operator",
        'ts_mean("hr")': "wrong_arity",
        'ts_mean(signal="hr", window=30)': "keyword_arguments_not_allowed",
        'conv_gamma("lunch", "carbs", 30, 5)': "parameter_out_of_range",
        'ts_mean(30, "hr")': "bad_argument_type",
        'interact(ts_mean("hr", 30), ts_mean("hr", 30))': "degenerate",
        'ts_mean("hr", 30': "syntax_error",
        "": "empty_expression",
    }
    for text, kind in cases.items():
        assert _kind(text) == kind, (text, _kind(text), kind)
    deep = 'interact(interact(interact(bmi(), native("last")), native("diff")), native("curv_1h"))'
    assert _kind(deep) == "too_deep"
    assert _kind("x" * 400) == "too_long"


def test_duplicates_are_found_by_the_canonical_form():
    a = L.canonical(L.parse_expression('interact(bmi(), ts_mean("hr", 30))'))
    b = L.canonical(L.parse_expression("interact( ts_mean('hr', 30.0) , bmi() )"))
    assert a == b                                                           # commutative product, quote style and integer-valued floats normalized
    assert L.canonical(L.parse_expression('ratio(bmi(), ts_mean("hr", 30))')) != L.canonical(L.parse_expression('ratio(ts_mean("hr", 30), bmi())'))


# ---- synthetic subject for the evaluator
def _subject(n=700, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2023-01-01")
    ts = t0 + pd.to_timedelta(np.arange(n), unit="min")
    native = np.cumsum(rng.normal(0, 4, n // 15 + 2)) + 110
    libre = np.interp(np.arange(n), np.arange(len(native)) * 15, native)
    minutes = pd.DataFrame({"timestamp": ts, "libre_gl": libre, "heart_rate_bpm": 70 + 10 * rng.random(n), "activity_calories_per_minute": rng.random(n) * 3,
                            "is_published_source_row": True, "is_inserted_grid_row": False, "source_segment_id": 1, "activity_met": rng.random(n)})
    minutes.loc[200:215, "heart_rate_bpm"] = np.nan
    events = pd.DataFrame({"timestamp": t0 + pd.to_timedelta([100, 260, 330, 520], unit="min"), "meal_type_normalized": ["Breakfast", "Lunch", "Lunch", "Dinner"],
                           "carbs": [40.0, 55.0, 0.0, 70.0], "protein": [10.0, 20.0, 5.0, 25.0], "fat": [8.0, 12.0, 0.0, 30.0], "fiber": [5.0, 6.0, 0.0, 9.0], "meal_calories": [300.0, 450.0, 20.0, 700.0]})
    origins = pd.Series(ts[300:640:5])
    return minutes, events, origins


def _context(minutes, events, origins):
    return CandidateContext(minutes, events, origins, bmi=27.5)


def test_evaluator_calls_the_fixed_operators():
    minutes, events, origins = _subject()
    ctx = _context(minutes, events, origins)
    pos = ctx.position
    got = L.evaluate_expression(L.parse_expression('ts_mean("hr", 30)'), ctx)
    assert np.allclose(got, ts_mean(ctx.activity["hr"], 30, name="hr").values[pos], equal_nan=True)
    got = L.evaluate_expression(L.parse_expression('conv_gamma("lunch", "carbs", 30, 2)'), ctx)
    assert np.allclose(got, conv_gamma(np.nan_to_num(ctx.series["lunch"]["carbs"]), 30, 2).values[pos])
    assert np.allclose(L.evaluate_expression(L.parse_expression("bmi()"), ctx), 27.5)


def test_division_by_zero_and_missing_values_give_missing_not_infinite():
    minutes, events, origins = _subject()
    ctx = _context(minutes, events, origins)
    value = L.evaluate_expression(L.parse_expression('ratio(last_meal_value("lunch", "carbs"), last_meal_value("lunch", "fat"))'), ctx)
    assert not np.isinf(value).any() and np.isnan(value).any()               # the second lunch has zero fat and the early windows have no lunch
    inside = _context(minutes, events, pd.Series([minutes["timestamp"].iloc[220]]))          # the window [190, 220] holds the 16 missing minutes 200-215
    got = L.evaluate_expression(L.parse_expression('ts_mean("hr", 30)'), inside)[0]
    assert np.isclose(got, np.nanmean(minutes["heart_rate_bpm"].to_numpy()[190:221]))      # missing minutes are skipped, never read as zero
    only_gap = minutes.copy()
    only_gap.loc[:, "heart_rate_bpm"] = np.nan
    assert np.isnan(L.evaluate_expression(L.parse_expression('ts_mean("hr", 30)'), _context(only_gap, events, origins))).all()


def test_values_at_t_do_not_change_when_everything_after_t_is_replaced():
    minutes, events, origins = _subject()
    expressions = [L.parse_expression(t) for t in [*VALID.values(), 'ts_max("calories", 120)', 'conv_gamma("any", "protein", 45, 3)', 'last_meal_value("dinner", "fiber")',
                                                  'nutrient_ratio("carb_share")', 'minutes_since_meal("lunch")', 'ratio(native("last"), bmi())', 'interact(native("diff"), native("curv_1h"))']]
    base = _context(minutes, events, origins)
    reference = {L.canonical(e): L.evaluate_expression(e, base) for e in expressions}
    rng = np.random.default_rng(3)
    for j in (10, 40, 60):
        origin = origins.iloc[j]
        later = minutes["timestamp"] > origin
        changed = minutes.copy()
        for c in ("libre_gl", "heart_rate_bpm", "activity_calories_per_minute"):
            changed.loc[later, c] = rng.normal(500, 100, later.sum())
        ev = events.copy()
        after = ev["timestamp"] > origin
        ev.loc[after, ["carbs", "protein", "fat", "fiber", "meal_calories"]] = 999.0
        extra = pd.DataFrame({"timestamp": [origin + pd.Timedelta(minutes=5)], "meal_type_normalized": ["Snack"], "carbs": [80.0], "protein": [1.0], "fat": [1.0], "fiber": [1.0], "meal_calories": [300.0]})
        ev = pd.concat([ev, extra], ignore_index=True)
        ctx = _context(changed, ev, origins)
        for e in expressions:
            now = L.evaluate_expression(e, ctx)
            before = reference[L.canonical(e)]
            assert np.array_equal(np.asarray(now)[j:j + 1], np.asarray(before)[j:j + 1], equal_nan=True), (L.canonical(e), j)


def test_the_perturbation_check_detects_a_leaky_operator():
    """Negative control: a centerd rolling mean (what the DSL forbids) changes when the minutes after t change, so the check above can fail."""
    minutes, _, origins = _subject()
    j = 20
    t = minutes["timestamp"] == origins.iloc[j]
    position = int(np.flatnonzero(t)[0])
    hr = minutes["heart_rate_bpm"].to_numpy().copy()
    leaky = pd.Series(hr).rolling(31, center=True, min_periods=1).mean().to_numpy()[position]
    hr[position + 1:] = 500.0
    assert not np.isclose(leaky, pd.Series(hr).rolling(31, center=True, min_periods=1).mean().to_numpy()[position])


# ---- prompt, reply, search
def test_prompt_is_built_from_fixed_text_and_aggregate_feedback_only():
    prompt = L.build_prompt(3, [('ts_mean("hr", 30)', 0.0123, "recent heart rate")], [("parameter_out_of_range", "ts_mean argument 2")])
    for name in L.OPERATORS:
        assert name + "(" in prompt
    for bad in ("2023-01-01 10:00:00", "subject 12", "values: 101.2, 99.5, 98.1, 97.7, 100.3, 104.4, 111.0"):
        try:
            L.prompt_guard(prompt + "\n" + bad)
        except L.PromptDataError:
            continue
        raise AssertionError(f"prompt guard accepted {bad!r}")


def test_reply_parser_demands_the_structure():
    ok = L.parse_response('Here you go: [{"expression": "bmi()", "hypothesis": "baseline"}, {"expression": "native(\\"last\\")", "hypothesis": "level", "motivation": "x"}]', expected=3)
    assert [i["expression"] for i in ok] == ["bmi()", 'native("last")']
    for bad in ("no json", '[{"expression": 3, "hypothesis": "x"}]', '[{"expression": "bmi()"}]', "[{"):
        try:
            L.parse_response(bad)
        except L.ResponseError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def _reply(expressions):
    return json.dumps([{"expression": e, "hypothesis": f"hypothesis {i}"} for i, e in enumerate(expressions)])


MEMORY = L.MemoryConfig(n_islands=2, top_k=3, n_examples=2, temperature=0.05)
POLICY = L.CountingPolicy(failed_call_counts_as_call=True, max_retries_per_call=1, invalid_proposal_counts_as_proposal=True)


def _fitness(expr):
    if "bmi" in expr and "ratio" in expr:
        return None                                                              # not evaluable
    return (sum(map(ord, expr)) % 97) / 1000.0


SCRIPT = [
    _reply(['ts_mean("hr", 30)', 'ts_mean("hr", 240)', 'interact(bmi(), ts_mean("hr", 30))']),
    _reply(['interact(ts_mean("hr", 30), bmi())', 'df.rolling(5, center=True).mean()', 'native("last")']),
    _reply(['ratio(native("last"), bmi())', 'exp_decay("hr", 30)', 'ts_max("calories", 15)']),
    "I cannot answer in JSON.",
]


def _run(path, model, rounds=3, limits=None, policy=POLICY):
    limits = limits or L.BudgetLimits(max_calls=20, proposals_per_call=3, max_proposals=100, max_evaluations=100)
    return L.run_search(model, _fitness, "A", rounds, MEMORY, limits, policy, L.SearchLog(path), seed=7, model_id="scripted", temperature=0.0)


def test_budget_counts_invalid_duplicate_and_unevaluable_proposals():
    with tempfile.TemporaryDirectory() as d:
        out = _run(Path(d) / "log.jsonl", ScriptedModel(SCRIPT), rounds=2)
        c = out["ledger"]
        # replies 0-3 go to island 0, island 1, island 0, island 1; reply 3 is not JSON, so 3 + 3 + 3 proposals
        # rejected: ts_mean 240 and rolling; duplicate: the second interact; not evaluable: ratio(native, bmi)
        assert c["calls"] == 4 and c["proposals"] == 9 and c["invalid"] == 2 and c["valid"] == 7 and c["duplicates"] == 1
        assert c["unique_valid"] == 6 and c["evaluations"] == 6 and c["unevaluable"] == 1
        free = L.CountingPolicy(failed_call_counts_as_call=True, max_retries_per_call=1, invalid_proposal_counts_as_proposal=False)
        out2 = _run(Path(d) / "log2.jsonl", ScriptedModel(SCRIPT), rounds=2, policy=free)
        assert out2["ledger"]["proposals"] == 9 - out2["ledger"]["invalid"] == 7


def test_limits_stop_the_search_and_failed_calls_follow_the_policy():
    with tempfile.TemporaryDirectory() as d:
        out = _run(Path(d) / "log.jsonl", ScriptedModel(SCRIPT), rounds=10, limits=L.BudgetLimits(max_calls=3, proposals_per_call=3, max_proposals=100, max_evaluations=100))
        assert out["stopped_by"] == "max_calls" and out["ledger"]["calls"] == 3
        model_calls = {"n": 0}

        def broken(request):
            model_calls["n"] += 1
            raise RuntimeError("service unavailable")

        out = _run(Path(d) / "log3.jsonl", broken, rounds=2, limits=L.BudgetLimits(max_calls=5, proposals_per_call=3, max_proposals=100, max_evaluations=100))
        assert out["ledger"]["failed_calls"] == model_calls["n"] and out["ledger"]["evaluations"] == 0 and out["stopped_by"] == "max_calls"
        assert model_calls["n"] <= 5                                           # a failing service cannot exceed the call limit, retries included


def test_replay_and_resume_reproduce_the_search_without_a_model():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.jsonl"
        scripted = ScriptedModel(SCRIPT)
        full = _run(path, scripted, rounds=3)
        n_calls = scripted.calls
        assert n_calls == full["ledger"]["calls"] > 0
        replay = _run(path, L.ReplayModel(L.SearchLog(path)), rounds=3)                       # same log, replay model that cannot answer anything new
        assert replay["islands"] == full["islands"] and replay["ledger"] == full["ledger"]
        def no_fitness(expression):
            raise AssertionError("a replay evaluated an expression")

        again = L.run_search(L.ReplayModel(L.SearchLog(path)), no_fitness, "A", 3, MEMORY, L.BudgetLimits(max_calls=20, proposals_per_call=3, max_proposals=100, max_evaluations=100), POLICY,
                             L.SearchLog(path), seed=7, model_id="scripted", temperature=0.0)
        assert again["islands"] == full["islands"]                                            # the replay needs neither the model nor the data
        rebuilt = llm_proposals(path).query("status == 'evaluated'")
        accepted = [e["expression"] for isl in full["islands"] for e in isl]
        assert set(accepted) <= set(rebuilt["canonical"]) and rebuilt["canonical"].is_unique
        # an interrupted run (two rounds), resumed from its log with more rounds, equals the uninterrupted one
        part = Path(d) / "part.jsonl"
        _run(part, ScriptedModel(SCRIPT), rounds=2)
        resumed = _run(part, ScriptedModel(SCRIPT), rounds=3)
        assert resumed["islands"] == full["islands"]
        try:
            _run(Path(d) / "empty.jsonl", L.ReplayModel(L.SearchLog(Path(d) / "empty.jsonl")), rounds=1)
        except KeyError:
            pass
        else:
            raise AssertionError("a replay model answered a request that is not in the log")


def test_validation_scores_never_reach_the_memory():
    """The fitness callback is the only source of scores; the search function has no validation input at all."""
    import inspect

    assert list(inspect.signature(L.run_search).parameters)[:3] == ["model", "fitness", "protocol"]
    assert not any("valid" in p for p in inspect.signature(L.run_search).parameters)


def test_island_memory_keeps_the_highest_gains_and_samples_them_more_often():
    """Fitness is a gain: a larger value is better, so the memory keeps the largest values and the Boltzmann sample prefers them."""
    island = L.Island(0, L.MemoryConfig(n_islands=1, top_k=3, n_examples=1, temperature=0.01))
    actions = [island.add(f"e{i}", g, 0, "h") for i, g in enumerate([0.02, -0.30, 0.05, 0.04, 0.01])]
    assert [e["fitness"] for e in island.entries] == [0.05, 0.04, 0.02]
    assert actions == ["entered", "entered", "entered", "entered", "rejected_below_top_k"] or actions[-1] == "rejected_below_top_k"
    rng = np.random.default_rng(0)
    picks = [island.sample(rng)[0]["fitness"] for _ in range(400)]
    assert picks.count(0.05) > picks.count(0.04) > picks.count(0.02)
    assert picks.count(0.05) > 250


def test_memory_parameters_have_no_defaults():
    for cls in (L.MemoryConfig, L.CountingPolicy, L.BudgetLimits):
        try:
            cls()
        except TypeError:
            continue
        raise AssertionError(f"{cls.__name__} has defaults")


def test_random_dsl_is_valid_seeded_and_budget_matched():
    s1, s2 = L.RandomDsl(5), L.RandomDsl(5)
    drawn = [s1.sample() for _ in range(300)]
    assert drawn == [s2.sample() for _ in range(300)]
    for text in drawn:
        L.check_causal(L.parse_expression(text))
    with tempfile.TemporaryDirectory() as d:
        llm = _run(Path(d) / "llm.jsonl", ScriptedModel(SCRIPT), rounds=3)
        log = L.SearchLog(Path(d) / "rand.jsonl")
        rnd = L.run_random_control(_fitness, "A", llm["ledger"]["evaluations"], seed=11, log=log)
        report = L.budget_report(llm["ledger"], rnd)
        assert bool(report.set_index("quantity").loc["fitness evaluations", "equal"])
        assert rnd["evaluations"] == llm["ledger"]["evaluations"] and rnd["unique_valid"] == rnd["evaluations"]




def test_the_module_has_no_network_or_dynamic_execution():
    source = Path(L.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not imported & {"requests", "urllib", "http", "socket", "openai", "anthropic", "subprocess", "httpx", "aiohttp"}
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"eval", "exec", "compile", "__import__"}


from src.llm_results import llm_proposals


def test_random_control_refuses_existing_log_without_fitness_or_append():
    from unittest import TestCase
    case = TestCase()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "random.jsonl"
        L.run_random_control(lambda expression: 0.1, "A", 2, 7, L.SearchLog(path))
        before = path.read_bytes()
        def forbidden(expression):
            raise AssertionError("fitness was called on an existing log")
        with case.assertRaises(ValueError):
            L.run_random_control(forbidden, "A", 2, 7, L.SearchLog(path))
        case.assertEqual(path.read_bytes(), before)


class ScriptedModel:
    """Offline stand-in for a model: replies from a list of strings (or a function of the request); used by synthetic tests."""

    def __init__(self, replies):
        self.replies, self.calls = replies, 0

    def __call__(self, request: dict) -> str:
        self.calls += 1
        if callable(self.replies):
            return self.replies(request)
        return self.replies[(self.calls - 1) % len(self.replies)]


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
