"""Constrained LLM-guided factor search and its Random-DSL control (offline machinery; no network code).

The language model may only propose expressions of a small DSL whose operators are the configured causal operators (`src/factor_construction.py`).
An expression is parsed with `ast` (never executed), checked against a whitelist of operators, fields and parameter grids, canonicalized for the duplicate test
and evaluated by calling the causal operators. The search keeps island memories, counts a budget, and writes an append-only JSONL log from which every accepted
candidate can be rebuilt without calling a model.

Nothing here decides a cost-relevant setting: the memory parameters, the counting policy and the limits are required arguments (no defaults), and the module contains
no client for any external service. A model is any callable `model(request) -> str`; `ReplayModel` answers only from a log.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.factor_construction import (ABSORPTION_KS, ABSORPTION_NUTRIENTS, ABSORPTION_TAUS, ACTIVITY_WINDOWS, EXP_DECAY_TAUS, HISTORY_MINUTES, MEAL_CLASSES, conv_gamma, exp_decay, interact,
                                     last_event_value, macronutrient_ratios, ratio, time_since_event, ts_max, ts_mean)

__all__ = ["DslError", "PauseSearch", "ResponseError", "PromptDataError", "OPERATORS", "parse_expression", "canonical", "uses_native", "source_offsets", "check_causal", "evaluate_expression", "dsl_specification",
           "build_prompt", "prompt_guard", "parse_response", "MemoryConfig", "CountingPolicy", "BudgetLimits", "Ledger", "SearchLog", "ReplayModel", "ReplayMiss", "run_search",
           "RandomDsl", "run_random_control", "budget_report"]

MEALS = tuple(MEAL_CLASSES.values())
MEALS_OR_ANY = MEALS + ("any",)
SIGNALS = ("hr", "calories")
NATIVE_NAMES = ("last", "diff", "slope_1h", "curv_1h")
RATIO_KINDS = ("carb_share", "fiber_to_carbs", "protein_x_carbs")
MAX_CHARS, MAX_DEPTH = 300, 3
FORBIDDEN_TERMS = ("target", "label", "residual", "subject", "groupby", "group_by", "rolling", "shift", "center", "lead", "future", "test", "import", "exec", "eval", "open", "read_csv",
                   "lambda", "__", "os.", "sys.", "http", "lag", "diff(", "pct_change", "expanding", "transform")
NATIVE_MIN_OFFSET = -74                     # native-reading values extend 74 minutes back; sampling-phase estimation uses a longer causal history


class DslError(ValueError):
    """A proposal that the DSL rejects; `kind` is a stable machine-readable class."""

    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind, self.detail = kind, detail


class PauseSearch(Exception):
    """Raised by a model adapter when the search must stop on a refusal.

    `invoked` is True when the refusal can only be known after the model call has been made (wrong model, usage beyond the plan, login source, usage limit): such a call is counted and
    its reply is not used. It is False for refusals before any invocation (nested session, billing variable, login check), which are not calls. `resets_at` (epoch seconds) is the end
    of a usage limit, if known; `text` is the refused reply, kept in the log for audit only."""

    def __init__(self, reason: str, detail: str = "", invoked: bool = False, resets_at: float | None = None, text: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail, self.invoked, self.resets_at, self.text = reason, detail, invoked, resets_at, text


class ResponseError(ValueError):
    """The model reply is not the requested JSON structure."""


class PromptDataError(ValueError):
    """A prompt that contains something that could be project data."""


# ------------------------------------------------------------------ the DSL
@dataclass(frozen=True)
class Operator:
    name: str
    choices: tuple          # one tuple of allowed values per literal parameter
    n_expr: int             # number of sub-expressions
    describe: str
    min_offset: int         # earliest source minute relative to t of the operator itself
    meaning: str


def _window_ops():
    return {
        "ts_mean": Operator("ts_mean", (SIGNALS, ACTIVITY_WINDOWS), 0, "ts_mean(signal, window)", 0, "mean of the observed minutes of heart rate or activity calories in the last `window` minutes"),
        "ts_max": Operator("ts_max", (SIGNALS, ACTIVITY_WINDOWS), 0, "ts_max(signal, window)", 0, "maximum of the observed minutes in the last `window` minutes"),
        "exp_decay": Operator("exp_decay", (SIGNALS, EXP_DECAY_TAUS), 0, "exp_decay(signal, tau)", -HISTORY_MINUTES, "exponentially weighted mean of the observed minutes over the last 120 minutes"),
        "conv_gamma": Operator("conv_gamma", (MEALS_OR_ANY, ABSORPTION_NUTRIENTS, ABSORPTION_TAUS, ABSORPTION_KS), 0, "conv_gamma(meal, nutrient, tau, k)", -HISTORY_MINUTES,
                               "logged meals of a class (or all classes) weighted by a unit-sum gamma kernel with time to peak `tau` minutes and shape `k` over the last 120 minutes"),
        "last_meal": Operator("last_meal", (ABSORPTION_NUTRIENTS,), 0, "last_meal(nutrient)", -HISTORY_MINUTES, "nutrient content of the most recent logged meal of any class, if it lies within 120 minutes"),
        "last_meal_value": Operator("last_meal_value", (MEALS, ABSORPTION_NUTRIENTS), 0, "last_meal_value(meal, nutrient)", -HISTORY_MINUTES,
                                    "nutrient content of the most recent logged meal of one class within 120 minutes"),
        "minutes_since_meal": Operator("minutes_since_meal", (MEALS_OR_ANY,), 0, "minutes_since_meal(meal)", -HISTORY_MINUTES, "minutes since the latest logged meal of a class (or any class) within 120 minutes"),
        "nutrient_ratio": Operator("nutrient_ratio", (RATIO_KINDS,), 0, "nutrient_ratio(kind)", -HISTORY_MINUTES,
                                   "carb_share = carbs/(carbs+protein+fat), fiber_to_carbs = fiber/carbs, protein_x_carbs = protein*carbs, of the most recent logged meal"),
        "native": Operator("native", (NATIVE_NAMES,), 0, "native(name)", NATIVE_MIN_OFFSET, "inferred native-reading state at or before t: last reading, last difference, 1-hour slope or curvature"),
        "bmi": Operator("bmi", (), 0, "bmi()", 0, "body-mass index of the subject at enrollment"),
        "interact": Operator("interact", (), 2, "interact(a, b)", 0, "product of two expressions"),
        "ratio": Operator("ratio", (), 2, "ratio(a, b)", 0, "a / b, undefined (missing) where b is not positive"),
    }


OPERATORS = _window_ops()


def _literal(node):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float))             and not isinstance(node.operand.value, bool):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool) and isinstance(node.value, (str, int, float)):
        return node.value
    raise DslError("forbidden_construct", f"argument of type {type(node).__name__}")


def _build(node, depth: int):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise DslError("forbidden_construct", type(node).__name__)
    if node.keywords:
        raise DslError("keyword_arguments_not_allowed", ",".join(str(k.arg) for k in node.keywords))
    name = node.func.id
    if name not in OPERATORS:
        raise DslError("unknown_operator", name)
    op = OPERATORS[name]
    if depth > MAX_DEPTH:
        raise DslError("too_deep", f"depth above {MAX_DEPTH}")
    if len(node.args) != len(op.choices) + op.n_expr:
        raise DslError("wrong_arity", f"{name} takes {len(op.choices) + op.n_expr} arguments, got {len(node.args)}")
    args = []
    for position, allowed in enumerate(op.choices):
        value = _literal(node.args[position])
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(allowed[0], str) != isinstance(value, str):
            raise DslError("bad_argument_type", f"{name} argument {position + 1}")
        if value not in allowed:
            raise DslError("parameter_out_of_range", f"{name} argument {position + 1} = {value!r} not in {list(allowed)}")
        args.append(value)
    for sub in node.args[len(op.choices):]:
        args.append(_build(sub, depth + 1))
    return (name, tuple(args))


def parse_expression(text: str) -> tuple:
    """Parse a DSL expression into a tree `(operator, args)`; never executes anything. Raises `DslError` with a stable `kind`."""
    if not isinstance(text, str) or not text.strip():
        raise DslError("empty_expression")
    text = text.strip()
    if len(text) > MAX_CHARS:
        raise DslError("too_long", f"{len(text)} characters")
    lowered = text.lower()
    for term in FORBIDDEN_TERMS:
        if term in lowered:
            raise DslError("forbidden_term", term)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise DslError("syntax_error", str(error.msg)) from None
    root = _build(tree.body, 1)
    _check_degenerate(root)
    return root


def _check_degenerate(node):
    name, args = node
    if name in ("interact", "ratio"):
        if canonical(args[0]) == canonical(args[1]):
            raise DslError("degenerate", f"{name} of an expression with itself")
        for sub in args:
            _check_degenerate(sub)


def canonical(node) -> str:
    """Canonical text: arguments normalized; the product is commutative, so its arguments are sorted."""
    name, args = node
    parts = [canonical(a) if isinstance(a, tuple) else json.dumps(a) for a in args]
    if name == "interact":
        parts = sorted(parts)
    return f"{name}({', '.join(parts)})"


def uses_native(node) -> bool:
    name, args = node
    return name == "native" or any(isinstance(a, tuple) and uses_native(a) for a in args)


def source_offsets(node) -> tuple:
    """(earliest, latest) source minute relative to t that the expression reads."""
    name, args = node
    op = OPERATORS[name]
    if name in ("ts_mean", "ts_max"):
        return -args[1], 0
    lows, highs = [op.min_offset], [0]
    for a in args:
        if isinstance(a, tuple):
            lo, hi = source_offsets(a)
            lows.append(lo)
            highs.append(hi)
    return min(lows), max(highs)


def check_causal(node):
    lo, hi = source_offsets(node)
    if hi > 0:
        raise DslError("future_reference", f"reads minute {hi} after t")
    if lo < -HISTORY_MINUTES:
        raise DslError("history_too_long", f"reads {lo} minutes back")
    return lo, hi


def evaluate_expression(node, context) -> np.ndarray:
    """Values at the window origins of a `CandidateContext` by calling the causal operators; the context holds one subject's minutes, meal events, BMI and origins."""
    name, args = node
    pos = context.position
    if name in ("ts_mean", "ts_max"):
        f = (ts_mean if name == "ts_mean" else ts_max)(context.activity[args[0]], args[1], name=args[0])
        return f.values[pos]
    if name == "exp_decay":
        return exp_decay(context.activity[args[0]], args[1], name=args[0]).values[pos]
    if name == "conv_gamma":
        meal, nutrient, tau, k = args
        pulses = np.nan_to_num(context.series[meal][nutrient], nan=0.0)
        return conv_gamma(pulses, tau, k, delay=0, name=f"{meal}.{nutrient}").values[pos]
    if name == "last_meal":
        return context._last_meal()[args[0]][pos]
    if name == "last_meal_value":
        s = context.series[args[0]]
        return last_event_value(s["event"], s[args[1]], delay=0, name=args[0]).values[pos]
    if name == "minutes_since_meal":
        return time_since_event(context.series[args[0]]["event"], delay=0, name=args[0]).values[pos]
    if name == "nutrient_ratio":
        last = context._last_meal()
        return macronutrient_ratios(last["carbs"], last["protein"], last["fat"], last["fiber"])[args[0]].values[pos]
    if name == "native":
        return np.asarray(context.native()[f"native_{args[0]}"], float)
    if name == "bmi":
        return np.full(len(pos), float(context.bmi))
    if name in ("interact", "ratio"):
        a, b = evaluate_expression(args[0], context), evaluate_expression(args[1], context)
        return (interact if name == "interact" else ratio)(a, b, name=canonical(node)).values
    raise DslError("unknown_operator", name)


def dsl_specification() -> str:
    """The operator list as it is shown to the model (generated from the registry, so prompt and validator cannot drift apart)."""
    lines = []
    for op in OPERATORS.values():
        grid = "; ".join(", ".join(json.dumps(v) for v in c) for c in op.choices)
        lines.append(f"- {op.describe}: {op.meaning}." + (f" Allowed values: {grid}." if op.choices else ""))
    lines.append(f"Expressions are function calls only, at most {MAX_DEPTH} levels deep and {MAX_CHARS} characters; strings in double quotes; no other syntax.")
    return "\n".join(lines)


# ------------------------------------------------------------------ prompt and reply
TASK = ("Task: propose new explanatory factors for the 30-minute-ahead glucose forecast residual of a fixed linear baseline. A factor is a number that can be computed at the forecast time t "
        "from the preceding 120 minutes of logged meals, heart rate, activity calories, the inferred native glucose readings and the body-mass index. The factors are used in a ridge regression "
        "on the baseline's residual; their value is judged by the gain on held-out parts of the training data only.")
CONSTRAINTS = ("Rules: use only the operators listed; every factor must be computable from information available at or before t; do not refer to the target, the residual, subjects, test data or "
               "any file; do not write code. Each proposal needs one sentence of hypothesis (a motivation, not a finding).")
OUTPUT_FORMAT = ("Reply with a JSON array of objects and nothing else: [{\"expression\": \"<DSL expression>\", \"hypothesis\": \"<one sentence>\", \"motivation\": \"<optional, short>\"}]. "
                 "Give exactly {n} objects.")
_PROMPT_PATTERNS = (re.compile(r"\d{4}-\d{2}-\d{2}"), re.compile(r"\d{1,2}:\d{2}:\d{2}"), re.compile(r"subject[\s_-]*\d", re.I), re.compile(r"\bsub[\s_-]?\d{1,2}\b", re.I),
                    re.compile(r"\b\d+\.\d{5,}\b"), re.compile(r"(?:\d+\.?\d*\s*,\s*){6,}\d"))


def prompt_guard(text: str) -> str:
    """Reject a prompt that looks like it carries data: timestamps, subject identifiers, long decimals, runs of numbers, or any number other than a four-decimal gain or a small integer."""
    for pattern in _PROMPT_PATTERNS:
        if pattern.search(text):
            raise PromptDataError(f"pattern {pattern.pattern!r} found in the prompt")
    rest = re.sub(r"gain=-?\d+\.\d{4}", "", text)
    for token in re.findall(r"\d+\.\d+|\d+", rest):
        if "." in token or int(token) > MAX_CHARS:
            raise PromptDataError(f"number {token} in the prompt is neither a gain nor a parameter")
    return text


def _clean(text: str, limit: int = 200) -> str:
    """One line of free text without decimals or large numbers (so that nothing that looks like a data value can pass through a model reply into a later prompt)."""
    text = re.sub(r"\d+[.,]\d+", "#", str(text))
    text = re.sub(r"\d{4,}", "#", text)
    return re.sub(r"\s+", " ", text)[:limit]


def build_prompt(n_proposals: int, examples: list, failures: list) -> str:
    """Prompt from fixed text, the DSL list, accepted expressions with their rounded training-fold fitness, and short rejection reasons. No data row can enter.

    `examples`: list of (canonical expression, fitness float, hypothesis str); `failures`: list of (rejection kind, short detail)."""
    parts = [TASK, "Operators:\n" + dsl_specification(), CONSTRAINTS]
    if examples:
        parts.append("Expressions accepted so far with their cross-validated gain (higher is better):\n" + "\n".join(f"- {e}  gain={f:.4f}  ({_clean(h, 120)})" for e, f, h in examples))
    if failures:
        parts.append("Recent rejections (do not repeat):\n" + "\n".join(f"- {k}: {_clean(d, 80)}" for k, d in failures))
    parts.append(OUTPUT_FORMAT.replace("{n}", str(n_proposals)))
    return prompt_guard("\n\n".join(parts))


def parse_response(text: str, expected: int | None = None) -> list:
    """The JSON array of the reply: list of dicts with `expression`, `hypothesis` and optional `motivation`."""
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ResponseError("no JSON array in the reply")
    try:
        items = json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        raise ResponseError(f"invalid JSON: {error.msg}") from None
    if not isinstance(items, list) or not all(isinstance(i, dict) and isinstance(i.get("expression"), str) and isinstance(i.get("hypothesis"), str) for i in items):
        raise ResponseError("every item needs string fields `expression` and `hypothesis`")
    if expected is not None and len(items) > expected:
        items = items[:expected]                         # extra proposals are ignored, not evaluated
    return [{"expression": i["expression"], "hypothesis": _clean(i["hypothesis"]), "motivation": _clean(i.get("motivation", ""))} for i in items]


# ------------------------------------------------------------------ island memory, budget, log
@dataclass(frozen=True)
class MemoryConfig:
    """Explicit search-budget settings; every field is required."""
    n_islands: int
    top_k: int
    n_examples: int
    temperature: float


@dataclass(frozen=True)
class CountingPolicy:
    """How failures count against the budget (to be fixed before a real run)."""
    failed_call_counts_as_call: bool
    max_retries_per_call: int
    invalid_proposal_counts_as_proposal: bool


@dataclass(frozen=True)
class BudgetLimits:
    max_calls: int
    proposals_per_call: int
    max_proposals: int
    max_evaluations: int


class Ledger:
    """Budget counters. `exceeded()` names the limit that stops the search."""

    FIELDS = ("calls", "accepted_replies", "malformed_replies", "failed_calls", "rejected_replies", "unknown_status_calls", "retries", "proposals", "valid", "unique_valid", "invalid", "duplicates",
              "evaluations", "unevaluable")

    def __init__(self, limits: BudgetLimits, policy: CountingPolicy):
        self.limits, self.policy = limits, policy
        self.count = {f: 0 for f in self.FIELDS}

    def exceeded(self) -> str | None:
        c, lim = self.count, self.limits
        if c["calls"] >= lim.max_calls:
            return "max_calls"
        if c["proposals"] >= lim.max_proposals:
            return "max_proposals"
        if c["evaluations"] >= lim.max_evaluations:
            return "max_evaluations"
        return None

    def snapshot(self) -> dict:
        return dict(self.count)


class SearchLog:
    """Append-only JSONL log; every record carries a type and a sequence number."""

    def __init__(self, path):
        self.path = Path(path)
        self.records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()] if self.path.exists() else []

    def append(self, record_type: str, **fields):
        record = {"seq": len(self.records), "type": record_type, **fields}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.records.append(record)
        return record

    def of(self, record_type: str) -> list:
        return [r for r in self.records if r["type"] == record_type]


class ReplayMiss(KeyError):
    """A request that has no logged reply was sent to a replay model: the search needs a model call that a replay must not make."""


class ReplayModel:
    """Answers from the log only; a request that was never answered is an error, so a replay can never reach a model."""

    def __init__(self, log: SearchLog):
        self.answers = {r["request_id"]: r["text"] for r in log.of("response")}

    def __call__(self, request: dict) -> str:
        if request["request_id"] not in self.answers:
            raise ReplayMiss(f"no logged reply for {request['request_id']}")
        return self.answers[request["request_id"]]


class Island:
    def __init__(self, island_id: int, cfg: MemoryConfig):
        self.id, self.cfg, self.entries = island_id, cfg, []

    def add(self, expression: str, fitness: float, round_: int, hypothesis: str) -> str:
        """Keep the `top_k` best by training-fold fitness; returns what happened to the entry."""
        if any(e["expression"] == expression for e in self.entries):
            return "already_in_memory"
        self.entries.append({"expression": expression, "fitness": float(fitness), "round": round_, "hypothesis": hypothesis})
        self.entries.sort(key=lambda e: (-e["fitness"], e["round"], e["expression"]))
        if len(self.entries) > self.cfg.top_k:
            dropped = self.entries.pop()
            return "entered" if dropped["expression"] != expression else "rejected_below_top_k"
        return "entered"

    def sample(self, rng: np.random.Generator) -> list:
        """Boltzmann sample (without replacement) of in-context examples from the memory."""
        if not self.entries:
            return []
        fit = np.array([e["fitness"] for e in self.entries])
        p = np.exp((fit - fit.max()) / self.cfg.temperature)
        p /= p.sum()
        n = min(self.cfg.n_examples, len(self.entries))
        pick = rng.choice(len(self.entries), size=n, replace=False, p=p)
        return [self.entries[i] for i in sorted(pick)]


# ------------------------------------------------------------------ the search loop
def _request_id(protocol: str, round_: int, island: int, attempt: int) -> str:
    return f"{protocol}|r{round_}|i{island}|a{attempt}"


def _call_metadata(model) -> dict:
    """Metadata of the last call of an adapter (model, tokens, plan status, never credentials); empty for models that keep none."""
    last = getattr(model, "last", None)
    return json.loads(json.dumps(dict(last), default=str)) if last else {}


def _limit_wait(log: SearchLog, clock) -> float | None:
    """Reset time (epoch seconds) of the last usage-limit pause if it lies in the future and no call has been made since; the search does not invoke the model before it."""
    pauses = log.of("pause")
    if not pauses or not pauses[-1].get("resets_at"):
        return None
    later_call = any(r["seq"] > pauses[-1]["seq"] and r["type"] == "response" for r in log.records)
    return pauses[-1]["resets_at"] if not later_call and clock() < pauses[-1]["resets_at"] else None


def run_search(model, fitness, protocol: str, rounds: int, memory: MemoryConfig, limits: BudgetLimits, policy: CountingPolicy, log: SearchLog, seed: int, model_id: str,
               temperature: float | None = None, config_digest: str | None = None, clock=time.time) -> dict:
    """Run (or resume, or replay) the island search.

    Call accounting. A *call* is every real invocation of the model, whatever came back:
      accepted reply          parsed (`accepted_replies`) or not parsable (`malformed_replies`),
      failed call             error, time-out, bad exit (`failed_calls`),
      rejected reply          the invocation happened and its result was refused afterwards: wrong model, usage beyond the plan, wrong login source, usage limit (`rejected_replies`),
      unknown status          a request that was logged but has no outcome, e.g. after an interruption: it may have reached the service, so it is counted (`unknown_status_calls`),
                              never sent again and not retried.
    `calls` is their sum and is bounded by `limits.max_calls` for the whole life of the log, so pausing, resuming or retrying cannot enlarge the budget. Only a refusal *before* the
    invocation (nested session, billing variable, login check; `PauseSearch.invoked` False) is not a call. Rejected text is logged for audit but never parsed or used.

    `fitness(canonical_expression) -> float | None` is the training-data fitness (None: not evaluable). Replies and judged proposals already in the log are reused without calling the
    model or the fitness. Returns {"ledger", "stopped_by", "islands"}; `stopped_by` starts with "paused:" when the search stopped on a refusal or on a usage limit.
    """
    if not policy.failed_call_counts_as_call:
        raise ValueError("every real invocation counts as a call: failed_call_counts_as_call must be True")
    if config_digest is not None:                          # a resumed search must run under the configuration it started with
        started = log.of("config")
        if started and started[0]["digest"] != config_digest:
            raise RuntimeError("the configuration differs from the one recorded in the log")
        if not started:
            log.append("config", digest=config_digest, protocol=protocol, seed=seed, model_id=model_id)
    ledger = Ledger(limits, policy)
    islands = [Island(i, memory) for i in range(memory.n_islands)]
    seen = {}
    answered = {r["request_id"]: r for r in log.of("response")}
    logged = {(r["request_id"], r["index"]): r for r in log.of("proposal")}
    requested = {r["request_id"]: r["prompt_sha256"] for r in log.of("request")}
    failures = [[] for _ in islands]
    stopped = None

    def paused(reason: str) -> dict:
        return {"ledger": ledger.snapshot(), "stopped_by": f"paused:{reason}", "islands": [i.entries for i in islands]}

    for round_ in range(rounds):
        for isl in islands:
            stopped = ledger.exceeded()
            if stopped:
                break
            rng = np.random.default_rng([seed, round_, isl.id])
            examples = [(e["expression"], e["fitness"], e["hypothesis"]) for e in isl.sample(rng)]
            prompt = build_prompt(limits.proposals_per_call, examples, failures[isl.id][-5:])
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            text, attempt = None, 0
            while text is None:
                rid = _request_id(protocol, round_, isl.id, attempt)
                reply = answered.get(rid)
                if reply is not None and reply.get("status") == "not_invoked":      # a refusal before the invocation left a marker: the slot is still free
                    del answered[rid]
                    requested.pop(rid, None)
                    reply = None
                if reply is None and rid in requested:
                    if requested[rid] != digest:
                        raise RuntimeError(f"the prompt of {rid} differs from the logged request; the search state is not reproducible")
                    # logged request without an outcome (interruption): it may have reached the service, so it counts as a call and is not repeated
                    reply = log.append("response", request_id=rid, text="", status="unknown", failed=True, error="unknown_status")
                    answered[rid] = reply
                if reply is not None:
                    status = reply.get("status") or ("failed" if reply.get("failed") else "accepted")
                    ledger.count["calls"] += 1
                    if status == "accepted":
                        text = reply["text"]
                        break
                    ledger.count[{"failed": "failed_calls", "rejected": "rejected_replies", "unknown": "unknown_status_calls"}[status]] += 1
                    if status == "unknown":
                        break                                                   # the slot is given up: an interrupted call is neither repeated nor retried
                    attempt += 1
                    if attempt > policy.max_retries_per_call:
                        break
                    ledger.count["retries"] += 1
                    continue
                # a new invocation
                if ledger.count["calls"] >= limits.max_calls:
                    break                                                       # no new call, retries included, beyond the call limit
                waiting = _limit_wait(log, clock)
                if waiting:
                    return paused("limit_not_reset")
                try:
                    if hasattr(model, "check"):
                        model.check()                                           # refusals before any invocation: nothing is logged, nothing is counted
                except PauseSearch as pause:
                    log.append("pause", request_id=rid, reason=pause.reason, detail=pause.detail, invoked=False, resets_at=pause.resets_at, **ledger.snapshot())
                    return paused(pause.reason)
                request = {"request_id": rid, "protocol": protocol, "round": round_, "island": isl.id, "prompt": prompt, "model_id": model_id, "temperature": temperature, "seed": seed}
                log.append("request", **request, prompt_sha256=digest)             # written before the invocation: the write-ahead marker for an interrupted call
                requested[rid] = digest
                try:
                    reply_text = model(request)
                except ReplayMiss:
                    raise
                except PauseSearch as pause:
                    if not pause.invoked:
                        answered[rid] = log.append("response", request_id=rid, text="", status="not_invoked", failed=False, reason=pause.reason)
                        log.append("pause", request_id=rid, reason=pause.reason, detail=pause.detail, invoked=False, resets_at=pause.resets_at, **ledger.snapshot())
                        return paused(pause.reason)
                    answered[rid] = log.append("response", request_id=rid, text="", status="rejected", failed=True, reason=pause.reason, detail=pause.detail,
                                               rejected_text=(pause.text or "")[:4000], call=_call_metadata(model))
                    ledger.count["calls"] += 1
                    ledger.count["rejected_replies"] += 1
                    log.append("pause", request_id=rid, reason=pause.reason, detail=pause.detail, invoked=True, resets_at=pause.resets_at, **ledger.snapshot())
                    return paused(pause.reason)
                except Exception as error:                                       # a failed call is logged and counted
                    answered[rid] = log.append("response", request_id=rid, text="", status="failed", failed=True, error=type(error).__name__, call=_call_metadata(model))
                    ledger.count["calls"] += 1
                    ledger.count["failed_calls"] += 1
                    attempt += 1
                    if attempt > policy.max_retries_per_call:
                        break
                    ledger.count["retries"] += 1
                    continue
                answered[rid] = log.append("response", request_id=rid, text=reply_text, status="accepted", failed=False, call=_call_metadata(model))
                ledger.count["calls"] += 1
                text = reply_text
            if text is None:
                continue
            try:
                items = parse_response(text, limits.proposals_per_call)
            except ResponseError as error:
                ledger.count["malformed_replies"] += 1
                if (rid, -1) not in logged:
                    logged[(rid, -1)] = log.append("proposal", request_id=rid, index=-1, status="malformed_response", detail=str(error))
                failures[isl.id].append(("malformed_response", str(error)))
                continue
            ledger.count["accepted_replies"] += 1
            for index, item in enumerate(items):
                ledger.count["proposals"] += 1
                rec = logged.get((rid, index))
                if rec is None:
                    rec = _judge(item, rid, index, seen, ledger, fitness, isl, round_, log)
                    logged[(rid, index)] = rec
                status = rec["status"]
                if status == "rejected":
                    ledger.count["invalid"] += 1
                    ledger.count["proposals"] -= int(not policy.invalid_proposal_counts_as_proposal)
                    failures[isl.id].append((rec["kind"], rec["detail"]))
                    continue
                ledger.count["valid"] += 1
                if status == "duplicate":
                    ledger.count["duplicates"] += 1
                    failures[isl.id].append(("duplicate", rec["canonical"]))
                    continue
                seen.setdefault(rec["canonical"], rid)
                ledger.count["unique_valid"] += 1
                if status == "not_evaluated_budget":
                    break
                ledger.count["evaluations"] += 1
                if status == "unevaluable":
                    ledger.count["unevaluable"] += 1
                    failures[isl.id].append(("unevaluable", rec["canonical"]))
                    continue
                isl.add(rec["canonical"], rec["fitness"], round_, rec["hypothesis"])
            if not any(r["type"] == "budget" and r["request_id"] == rid for r in log.records):
                log.append("budget", request_id=rid, **ledger.snapshot())
        if stopped:
            break
    stopped = stopped or ledger.exceeded()
    last = log.of("stop")[-1] if log.of("stop") else None
    if last is None or any(last.get(k) != v for k, v in ledger.snapshot().items()) or last["reason"] != (stopped or "rounds_completed"):
        log.append("stop", reason=stopped or "rounds_completed", **ledger.snapshot())
    return {"ledger": ledger.snapshot(), "stopped_by": stopped or "rounds_completed", "islands": [i.entries for i in islands]}


def _judge(item: dict, rid: str, index: int, seen: dict, ledger: Ledger, fitness, island: Island, round_: int, log: SearchLog) -> dict:
    """Validate one proposal, evaluate it if it is new and valid, and write its log record."""
    try:
        node = parse_expression(item["expression"])
        check_causal(node)
        expr = canonical(node)
    except DslError as error:
        return log.append("proposal", request_id=rid, index=index, raw=item["expression"], status="rejected", kind=error.kind, detail=error.detail)
    if expr in seen:
        return log.append("proposal", request_id=rid, index=index, raw=item["expression"], canonical=expr, status="duplicate", first_seen=seen[expr])
    if ledger.count["evaluations"] >= ledger.limits.max_evaluations:
        return log.append("proposal", request_id=rid, index=index, canonical=expr, status="not_evaluated_budget")
    value = fitness(expr)
    if value is None or not np.isfinite(value):
        return log.append("proposal", request_id=rid, index=index, raw=item["expression"], canonical=expr, status="unevaluable", hypothesis=item["hypothesis"])
    return log.append("proposal", request_id=rid, index=index, raw=item["expression"], canonical=expr, status="evaluated", fitness=float(value), hypothesis=item["hypothesis"],
                      motivation=item["motivation"], island=island.id, round=round_)


# ------------------------------------------------------------------ Random-DSL control
class RandomDsl:
    """Seeded sampler over the same grammar and parameter grids: a single operator or a product / ratio of two operators (same validator and duplicate rule as the model's proposals)."""

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.leaves = [n for n, o in OPERATORS.items() if o.n_expr == 0]

    def _leaf(self) -> str:
        name = self.leaves[self.rng.integers(len(self.leaves))]
        args = [json.dumps(c[self.rng.integers(len(c))]) if isinstance(c[0], str) else str(c[self.rng.integers(len(c))]) for c in OPERATORS[name].choices]
        return f"{name}({', '.join(args)})"

    def sample(self) -> str:
        if self.rng.random() < 1 / 3:
            return self._leaf()
        first = self._leaf()
        second = self._leaf()
        while second == first:                       # a product or ratio of an expression with itself is rejected by the DSL, so it is not drawn
            second = self._leaf()
        return f"{('interact', 'ratio')[self.rng.integers(2)]}({first}, {second})"


def run_random_control(fitness, protocol: str, target_evaluations: int, seed: int, log: SearchLog, max_draws: int = 100000) -> dict:
    """Draw expressions until `target_evaluations` unique valid ones have been evaluated (the evaluation count of the model search); duplicates and unevaluable draws are counted."""
    if log.records:
        raise ValueError("Random-DSL output log must be empty; choose a new output path")
    sampler, seen = RandomDsl(seed), set()
    count = {"draws": 0, "invalid": 0, "duplicates": 0, "unique_valid": 0, "evaluations": 0, "unevaluable": 0}
    while count["evaluations"] < target_evaluations and count["draws"] < max_draws:
        text = sampler.sample()
        count["draws"] += 1
        try:
            node = parse_expression(text)
            check_causal(node)
            expr = canonical(node)
        except DslError as error:
            count["invalid"] += 1
            log.append("random_draw", protocol=protocol, raw=text, status="rejected", kind=error.kind)
            continue
        if expr in seen:
            count["duplicates"] += 1
            continue
        seen.add(expr)
        count["unique_valid"] += 1
        value = fitness(expr)
        count["evaluations"] += 1
        if value is None or not np.isfinite(value):
            count["unevaluable"] += 1
            log.append("random_draw", protocol=protocol, canonical=expr, status="unevaluable")
            continue
        log.append("random_draw", protocol=protocol, canonical=expr, status="evaluated", fitness=float(value))
    log.append("random_stop", protocol=protocol, seed=seed, **count)
    return count


def budget_report(llm_ledger: dict, random_count: dict) -> pd.DataFrame:
    """The four budget quantities of the equal-budget definition side by side."""
    rows = [("total proposals", llm_ledger["proposals"], random_count["draws"]), ("valid proposals", llm_ledger["valid"], random_count["unique_valid"] + random_count["duplicates"]),
            ("unique valid expressions", llm_ledger["unique_valid"], random_count["unique_valid"]), ("fitness evaluations", llm_ledger["evaluations"], random_count["evaluations"])]
    out = pd.DataFrame(rows, columns=["quantity", "llm", "random_dsl"])
    out["equal"] = out["llm"] == out["random_dsl"]
    return out


# ------------------------------------------------------------------ training-split fitness (building blocks; the residual definition is fixed by the caller)


