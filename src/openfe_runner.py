"""OpenFE comparator for the residual factor search.

OpenFE 0.0.12 runs in its own environment (`.venv-openfe`, scikit-learn 1.5.2). This module has two parts:

* a formula evaluator that needs neither OpenFE nor any fitted state: the selected features are applied to any window table by project code, row by row, with the semantics of the
  OpenFE operators (`parse_formula`, `evaluate_formula`);
* the search itself (`candidate_formulas`, `run_search`, command line entry), which imports OpenFE and is only called from `.venv-openfe`.

Only operators that act on a single row are allowed (the protocol's `openfe.operators_allowed`). The operators that use statistics across rows (`freq`, `GroupByThen*`, `Combine*`) are never
generated; `candidate_formulas` refuses to return them.
"""

from __future__ import annotations

import json
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:                                                      # OpenFE is installed in .venv-openfe only
    from openfe import OpenFE
except ImportError:                                       # pragma: no cover - the main environment
    OpenFE = None

__all__ = ["ROW_WISE_OPERATORS", "GLOBAL_OPERATORS", "parse_formula", "formula_columns", "evaluate_formula", "candidate_formulas", "run_search"]

UNARY_OPERATORS = ("abs", "log", "sqrt", "square", "sigmoid", "round", "residual")
BINARY_FUNCTIONS = ("min", "max")
BINARY_SYMBOLS = ("+", "-", "*", "/")
ROW_WISE_OPERATORS = UNARY_OPERATORS + BINARY_FUNCTIONS + BINARY_SYMBOLS
GLOBAL_OPERATORS = ("freq", "GroupByThenMin", "GroupByThenMax", "GroupByThenMean", "GroupByThenMedian", "GroupByThenStd", "GroupByThenRank", "GroupByThenNUnique", "Combine", "CombineThenFreq")
_DELIMITERS = "(),+-*/"


# ------------------------------------------------------------------ formulas: parsing and evaluation without OpenFE
def parse_formula(text: str):
    """Parse an OpenFE formula string into a tree: a column name, or (operator, [children]).

    Binary arithmetic is written `(a+b)`, every other operator `name(arg)` or `name(arg1,arg2)`, as in OpenFE's `tree_to_formula`. Column names must not contain `()+-*/,`.
    """
    tree, end = _parse(text, 0)
    if end != len(text):
        raise ValueError(f"trailing characters in formula {text!r}")
    return tree


def _parse(text: str, i: int):
    if text[i] == "(":
        left, i = _parse(text, i + 1)
        operator = text[i]
        if operator not in BINARY_SYMBOLS:
            raise ValueError(f"expected an arithmetic operator at position {i} of {text!r}")
        right, i = _parse(text, i + 1)
        if text[i] != ")":
            raise ValueError(f"expected ')' at position {i} of {text!r}")
        return (operator, [left, right]), i + 1
    j = i
    while j < len(text) and text[j] not in _DELIMITERS:
        j += 1
    name = text[i:j]
    if not name:
        raise ValueError(f"empty name at position {i} of {text!r}")
    if j < len(text) and text[j] == "(":
        children, j = [], j + 1
        while True:
            child, j = _parse(text, j)
            children.append(child)
            if text[j] == ",":
                j += 1
            elif text[j] == ")":
                return (name, children), j + 1
            else:
                raise ValueError(f"expected ',' or ')' at position {j} of {text!r}")
    return name, j


def formula_columns(text: str) -> list:
    """Input columns that a formula reads, in order of first appearance."""
    columns = []

    def walk(node):
        if isinstance(node, str):
            if node not in columns:
                columns.append(node)
        else:
            for child in node[1]:
                walk(child)

    walk(parse_formula(text))
    return columns


def _apply(node, frame: pd.DataFrame) -> np.ndarray:
    if isinstance(node, str):
        return frame[node].to_numpy(dtype=float)
    operator, children = node
    values = [_apply(child, frame) for child in children]
    with np.errstate(all="ignore"):
        if operator in UNARY_OPERATORS and len(values) == 1:
            d = values[0]
            if operator == "abs":
                return np.abs(d)
            if operator == "log":
                return np.log(np.abs(np.where(d == 0, np.nan, d)))
            if operator == "sqrt":
                return np.sqrt(np.abs(d))
            if operator == "square":
                return np.square(d)
            if operator == "sigmoid":
                return 1 / (1 + np.exp(-d))
            if operator == "round":
                return np.floor(d)
            return d - np.floor(d)                                                       # residual
        if operator in BINARY_FUNCTIONS + BINARY_SYMBOLS and len(values) == 2:
            a, b = values
            if operator == "max":
                return np.maximum(a, b)
            if operator == "min":
                return np.minimum(a, b)
            if operator == "+":
                return a + b
            if operator == "-":
                return a - b
            if operator == "*":
                return a * b
            return a / np.where(b == 0, np.nan, b)
    raise ValueError(f"operator {operator!r} with {len(values)} argument(s) is not a row-wise OpenFE operator")


def evaluate_formula(text: str, frame: pd.DataFrame) -> pd.Series:
    """Value of a formula for every row of `frame`, computed from that row alone (infinite values become NaN, as in OpenFE's `transform`)."""
    values = _apply(parse_formula(text), frame)
    values[~np.isfinite(values)] = np.nan
    return pd.Series(values, index=frame.index, name=text)


# ------------------------------------------------------------------ the search (needs OpenFE)
def candidate_formulas(columns: list) -> list:
    """Order-1 OpenFE candidates over numeric columns with the row-wise operators only: 7 unary operators and 6 binary operators on every pair of different columns."""
    from openfe import get_candidate_features
    from openfe.utils import tree_to_formula

    nodes = get_candidate_features(numerical_features=list(columns), categorical_features=[], ordinal_features=[], order=1)
    formulas = [tree_to_formula(node) for node in nodes if node.name != "freq"]          # `freq` counts values across rows; it is the only global operator generated without categorical columns
    for text in formulas:
        operators = _operators_of(parse_formula(text))
        if not set(operators) <= set(ROW_WISE_OPERATORS):
            raise ValueError(f"candidate {text!r} uses a global operator")
    return formulas


def _operators_of(node) -> list:
    if isinstance(node, str):
        return []
    return [node[0]] + [op for child in node[1] for op in _operators_of(child)]


if OpenFE is not None:
    class RecordingOpenFE(OpenFE):
        """OpenFE with a log of the stage-one rounds (candidates entering, scores of the candidates). The search itself is unchanged."""

        def _calculate_and_evaluate(self, candidate_features, train_idx, val_idx):
            from openfe.utils import tree_to_formula

            results = super()._calculate_and_evaluate(candidate_features, train_idx, val_idx)
            self.round_log = getattr(self, "round_log", [])
            self.round_log.append({"round": len(self.round_log) + 1, "candidates": len(results), "train_rows": len(train_idx), "validation_rows": len(val_idx),
                                   "scores": {tree_to_formula(node): float(score) for node, score in results}})
            return results


def run_search(config: dict) -> dict:
    """Run OpenFE on one prepared input table and write the ranked features.

    config keys: input (parquet with `openfe_index`, `role` in {internal_train, internal_validation}, `label`, `init_score`, and the input columns), columns, output_dir, tag, seed, n_jobs,
    tmp_dir, n_data_blocks, min_candidate_features, max_candidates (optional, smoke tests only).
    """
    out = Path(config["output_dir"]).resolve()
    if out.exists():
        raise ValueError("OpenFE output must be a new directory")
    temporary = (Path(config["tmp_dir"]) / f"openfe_tmp_{config['tag']}.feather").resolve()
    root = Path(__file__).resolve().parents[1]
    if any(path.is_relative_to(root / folder) for path in (out, temporary) for folder in ("results", "history", "data", "report")):
        raise ValueError("OpenFE outputs must be outside saved scientific inputs and historical evidence")
    if temporary.exists():
        raise ValueError("OpenFE temporary output already exists; choose a new temporary path")
    expected = config.get("input_sha256")
    if expected is not None:
        with Path(config["input"]).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("OpenFE input_sha256 does not match the search input")
    table = pd.read_parquet(config["input"]).set_index("openfe_index")
    columns = list(config["columns"])
    data = table[columns].astype(float)
    label = table[["label"]].astype(float)
    init_scores = table[["init_score"]].astype(float)
    train_index = table.index[table["role"].eq("internal_train")]
    val_index = table.index[table["role"].eq("internal_validation")]
    if not train_index.is_unique or not val_index.is_unique or len(train_index.intersection(val_index)):
        raise ValueError("OpenFE internal splits must have unique disjoint row indices")
    if not np.isfinite(init_scores["init_score"]).all() or not np.isfinite(label["label"]).all():
        raise ValueError("OpenFE labels and initial scores must be finite")
    candidates = candidate_formulas(columns)
    if config.get("max_candidates"):
        rng = np.random.default_rng(config["seed"])
        candidates = [candidates[i] for i in sorted(rng.choice(len(candidates), size=min(config["max_candidates"], len(candidates)), replace=False))]
    from openfe.utils import formula_to_tree, tree_to_formula

    nodes = [formula_to_tree(text) for text in candidates]
    assert [tree_to_formula(n) for n in nodes] == candidates
    out.mkdir(parents=True)
    started = time.time()
    ofe = RecordingOpenFE()
    ofe.fit(data=data, label=label, task="regression", train_index=train_index, val_index=val_index, candidate_features_list=nodes, init_scores=init_scores, categorical_features=[],
            n_data_blocks=config.get("n_data_blocks", 8), min_candidate_features=config.get("min_candidate_features", 2000), feature_boosting=True, stage1_metric="predictive",
            stage2_metric="gain_importance", is_stage1=True, n_jobs=config.get("n_jobs", 1), seed=config.get("seed", 42), tmp_save_path=str(temporary),
            verbose=config.get("verbose", True))
    ranked = [(tree_to_formula(node), float(score)) for node, score in ofe.new_features_scores_list]
    last_round = ofe.round_log[-1]["scores"] if getattr(ofe, "round_log", None) else {}
    frame = pd.DataFrame({"rank": np.arange(1, len(ranked) + 1), "formula": [f for f, _ in ranked], "stage2_gain_importance": [s for _, s in ranked]})
    frame["stage1_last_round_score"] = frame["formula"].map(last_round)
    frame["input_columns"] = frame["formula"].map(lambda f: json.dumps(formula_columns(f)))
    frame["operators"] = frame["formula"].map(lambda f: json.dumps(sorted(set(_operators_of(parse_formula(f))))))
    frame.to_csv(out / f"ranking_{config['tag']}.csv", index=False)
    rounds = [{k: v for k, v in r.items() if k != "scores"} for r in getattr(ofe, "round_log", [])]
    import lightgbm
    import openfe
    import sklearn

    summary = {"tag": config["tag"], "candidates_generated": len(candidates), "stage1_rounds": rounds, "ranked_candidates": len(ranked), "seconds": round(time.time() - started, 1),
               "internal_train_rows": int(len(train_index)), "internal_validation_rows": int(len(val_index)), "seed": config.get("seed", 42), "n_jobs": config.get("n_jobs", 1),
               "n_data_blocks": config.get("n_data_blocks", 8), "min_candidate_features": config.get("min_candidate_features", 2000), "input_sha256": config.get("input_sha256"),
               "input_columns": columns, "versions": {"python": sys.version.split()[0], "openfe": openfe.__version__, "lightgbm": lightgbm.__version__, "scikit_learn": sklearn.__version__,
                                                       "numpy": np.__version__, "pandas": pd.__version__}}
    (out / f"summary_{config['tag']}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


if __name__ == "__main__":                                     # required on Windows: OpenFE starts worker processes
    print(json.dumps(run_search(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))))
