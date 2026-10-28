"""Synthetic checks of the validation reconstruction / authorized test evaluation execution control (`src/test_execution.py`) and of the delivered notebooks' read-only defaults.

Only a simulated lock, simulated files, mock evaluation functions and the text of the notebook are used. No project data file is opened, no model is fitted, no test split is touched; a read recorder
confirms that nothing is opened by the refusals.

Run with `python tests/test_execution_control.py` from the repository root.
"""

import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import factor_evaluation as FE  # noqa: E402
from src import factor_validation as FG  # noqa: E402
from src import test_execution as PC  # noqa: E402
from src import window_data as W  # noqa: E402

NOW = lambda: "2026-01-01T00:00:00+00:00"       # noqa: E731
LOCK_PATH = ROOT / "results" / "factor_discovery" / "evaluation_plan.json"
AUTH_PATH = ROOT / "results" / "factor_discovery" / PC.AUTH_NAME


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Evaluate:
    """A mock of the evaluation entry: counts its calls, returns small tables, can be told to fail."""

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def __call__(self, lock, access):
        self.calls.append(access)
        if self.fail:
            raise ValueError("simulated failure inside the evaluation")
        contrasts = pd.DataFrame({"id": ["A:P1", "A:P1"], "stratum": ["all", "P30"], "d_rmse": [-0.1, -0.2]})
        summary = pd.DataFrame({"config": ["base"], "rmse_pooled": [1.0]})
        return {tag: {"contrasts": contrasts, "summary": summary} for tag in ("A", "B")}


def make_root(tmp):
    root = Path(tmp)
    fd = root / "results" / "factor_discovery"
    for d in (fd, root / "src", root / "notebooks", root / "results" / "03_x"):
        d.mkdir(parents=True, exist_ok=True)
    (root / "src" / "a.py").write_bytes(b"x = 1\n")
    (root / "src" / "b.py").write_bytes(b"y = 2\n")
    (fd / "factor_protocol.json").write_bytes(b'{"frozen": true}')
    (fd / "small.csv").write_bytes(b"a\n1\n")
    (root / "results" / "03_x" / "big.parquet").write_bytes(b"PAR1 synthetic PAR1")
    nb = {"cells": [{"cell_type": "code", "source": ["a = 1\n", "b = 2"], "outputs": []}, {"cell_type": "markdown", "source": ["text"]}, {"cell_type": "code", "source": ["c = a + b"], "outputs": []}]}
    (root / "notebooks" / "09_factor_evaluation.ipynb").write_text(json.dumps(nb), encoding="utf-8")
    lock = {"schema_version": 1, "status": "DRAFT awaiting lock approval; not in force", "approval": {"approved": False, "date": None, "note": "pending"}, "execute_test_default": False, "decisions": {"comparison_scope": {"choice": "plan A"}, "Q5": {"decision": "approved"}}, "test_split_status": "previously used test split (simulated)",
            "revision": {"number": 3}, "access_checks": {"status": "qualified"}, "protocol": {"file": "results/factor_discovery/factor_protocol.json", "sha256": sha(b'{"frozen": true}')},
            "comparisons": [{"id": "A:P1", "role": "confirmatory (Holm family of 4)"}], "configs": {"A": {"base": {"columns": [], "alpha": None}}}, "statistics": {"holm": "m = 4"},
            "provenance": {"inputs_sha256": {"small.csv": sha(b"a\n1\n"), "results/03_x/big.parquet": sha(b"PAR1 synthetic PAR1")}, "code_sha256": {"src/a.py": sha(b"x = 1\n"), "src/b.py": sha(b"y = 2\n")}}}
    lp = fd / "evaluation_plan.json"
    lp.write_bytes(json.dumps(lock, indent=1).replace("\n", "\r\n").encode("utf-8"))
    return root


def read_lock(root):
    return json.loads((Path(root) / "results" / "factor_discovery" / "evaluation_plan.json").read_text(encoding="utf-8"))


def run(root, execute_test, evaluate, out_dir=None):
    return PC.run_authorized_test(read_lock(root), root, execute_test, out_dir=out_dir or Path(root) / "results" / "factor_evaluation", evaluate=evaluate, now=NOW)


def refused(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except PC.ExecutionRefused as error:
        return str(error)
    raise AssertionError("not refused")


def approve(root, authorize_execution=True):
    # Fixture-only permission for files created under TemporaryDirectory.
    fd = root / "results" / "factor_discovery"
    lock = read_lock(root)
    lock.update(status="LOCKED (synthetic)", approval={"approved": True, "date": "2026-02-02", "note": "synthetic"})
    (fd / "evaluation_plan.json").write_text(json.dumps(lock), encoding="utf-8")
    auth = {"approval": {"approved": True, "date": "2026-02-02", "by": "fixture"},
            "execution": {"execution_authorized": authorize_execution, "date": "2026-02-03", "by": "fixture"},
            "lock_scientific_sha256": PC.scientific_sha256(lock), "protocol_sha256": lock["protocol"]["sha256"],
            "code_sha256": lock["provenance"]["code_sha256"], "inputs_sha256": lock["provenance"]["inputs_sha256"],
            "notebook_code_cells_sha256": PC.notebook_code_sha256(root / "notebooks/09_factor_evaluation.ipynb")}
    (fd / PC.AUTH_NAME).write_text(json.dumps(auth), encoding="utf-8")


# ------------------------------------------------------------------ 1. approval not approved
def test_unapproved_every_request_for_the_test_is_refused_before_any_read():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(tmp)
        (root / "results" / "factor_discovery" / PC.AUTH_NAME).write_text(json.dumps({"approval": {"approved": False}, "execution": {"execution_authorized": True}}), encoding="utf-8")
        for auth_present in (False, True):
            if not auth_present:
                (root / "results" / "factor_discovery" / PC.AUTH_NAME).unlink()
            for execute_test in (True, False):
                mock = Evaluate()
                with FG.ReadRecorder() as recorder:
                    refused(run, root, execute_test, mock)
                assert mock.calls == [] and recorder.log == []
                assert not (root / "results" / "factor_evaluation").exists()
        # without an evaluate argument the default entry (which would call the real evaluation) is never reached
        with FG.ReadRecorder() as recorder:
            refused(PC.run_authorized_test, read_lock(root), root, True, out_dir=root / "results" / "factor_evaluation")
        assert recorder.log == []
    # the Access combinations of the data layer, with the real loader
    for access in (W.Access(), W.Access(execute_test=True), W.Access(lock_approved=True)):
        with FG.ReadRecorder() as recorder:
            try:
                W.load_keys(W.PROTOCOL_A, ("train", "test"), access)
            except W.TestAccessDenied:
                pass
            else:
                raise AssertionError("the test split was not refused")
        assert recorder.log == []
    # the modes
    lock = {"approval": {"approved": False}}
    assert PC.resolve_mode(lock, None, False) == "validation" and PC.resolve_mode(lock, {"approval": {"approved": False}}, False) == "validation"
    refused(PC.resolve_mode, lock, None, True)
    refused(PC.resolve_mode, {"approval": {"approved": True}}, {"approval": {"approved": False}}, False)        # the two records must agree
    refused(PC.resolve_mode, {"approval": {"approved": False}}, {"approval": {"approved": True}}, True)


# ------------------------------------------------------------------ 2. approval approved, EXECUTE_TEST False
def test_approved_but_not_executing_runs_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(tmp)
        approve(root, authorize_execution=True)
        lock = read_lock(root)
        auth = PC.load_authorization(root / "results" / "factor_discovery" / PC.AUTH_NAME)
        assert PC.resolve_mode(lock, auth, False) == "approved_idle" and PC.resolve_mode(lock, auth, True) == "authorized_test"
        mock = Evaluate()
        with FG.ReadRecorder() as recorder:
            assert "EXECUTE_TEST" in refused(run, root, False, mock)
        assert mock.calls == [] and recorder.log == [] and not (root / "results" / "factor_evaluation").exists()
        # approval approved but no separate authorized test evaluation execution authorization
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(tmp)
        approve(root, authorize_execution=False)
        mock = Evaluate()
        assert "authorized test evaluation" in refused(run, root, True, mock)
        assert mock.calls == [] and not (root / "results" / "factor_evaluation").exists()


# ------------------------------------------------------------------ 3. everything consistent: the simulated test runs once
def test_a_consistent_permission_runs_the_simulated_test_once_without_changing_science_or_code():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(tmp)
        before = read_lock(root)
        code_before = {p.name: sha(p.read_bytes()) for p in (root / "src").glob("*.py")}
        approve(root)
        after = read_lock(root)
        assert PC.scientific_sha256(before) == PC.scientific_sha256(after)                         # approval is registered without touching the scientific content
        assert {k for k in after if after[k] != before[k]} == {"status", "approval"} and after["status"].startswith("LOCKED") and after["approval"]["approved"] is True
        assert code_before == {p.name: sha(p.read_bytes()) for p in (root / "src").glob("*.py")}   # nor any code file
        mock = Evaluate()
        out = run(root, True, mock)
        assert len(mock.calls) == 1 and mock.calls[0].permits_test() and out["state"]["status"] == "COMPLETED"
        out_dir = root / "results" / "factor_evaluation"
        assert sorted(p.name for p in out_dir.iterdir()) == sorted([*PC.OUTPUT_FILES, PC.STATE_NAME])
        gains = pd.read_csv(out_dir / "comparison_results.csv")
        strata = pd.read_csv(out_dir / "meal_strata.csv")
        assert set(gains["stratum"]) == {"all"} and set(strata["stratum"]) == {"P30"} and gains["status"].str.contains("previously used test split").all()
        state = json.loads((out_dir / PC.STATE_NAME).read_text(encoding="utf-8"))
        assert state["status"] == "COMPLETED" and set(state["output_sha256"]) == set(PC.OUTPUT_FILES)


# ------------------------------------------------------------------ 4. mismatches abort before the evaluation
def test_any_hash_mismatch_aborts_before_the_evaluation_and_before_a_state_file_exists():
    def tamper_lock(root):
        path = root / "results" / "factor_discovery" / "evaluation_plan.json"
        lock = json.loads(path.read_text(encoding="utf-8"))
        lock["comparisons"][0]["id"] = "A:P9"
        path.write_bytes(json.dumps(lock, indent=1).replace("\n", "\r\n").encode("utf-8"))

    def tamper_text(root):
        path = root / "results" / "factor_discovery" / "evaluation_plan.json"
        lock = json.loads(path.read_text(encoding="utf-8"))
        lock["statistics"]["holm"] = "m = 5"
        path.write_bytes(json.dumps(lock, indent=1).replace("\n", "\r\n").encode("utf-8"))

    cases = {
        "scientific content (comparison id)": (tamper_lock, "scientific content"),
        "scientific content (text of the statistics)": (tamper_text, "scientific content"),
        "protocol": (lambda r: (r / "results" / "factor_discovery" / "factor_protocol.json").write_bytes(b'{"frozen": false}'), "protocol"),
        "missing registered source": (lambda r: (r / "src" / "a.py").unlink(), "code version"),
        "code file": (lambda r: (r / "src" / "b.py").write_bytes(b"y = 3\n"), "code version"),
        "notebook code cell": (lambda r: (r / "notebooks" / "09_factor_evaluation.ipynb").write_text(json.dumps({"cells": [{"cell_type": "code", "source": ["a = 9"], "outputs": []}]}), encoding="utf-8"), "factor-evaluation notebook"),
        "small input": (lambda r: (r / "results" / "factor_discovery" / "small.csv").write_bytes(b"a\n2\n"), "input differs"),
        "large input": (lambda r: (r / "results" / "03_x" / "big.parquet").write_bytes(b"changed"), "input differs"),
        "lock marked draft while the record is approved": (lambda r: None, "disagree"),
    }
    for name, (tamper, expected) in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            root = make_root(tmp)
            approve(root)
            if name.startswith("lock marked"):
                path = root / "results" / "factor_discovery" / "evaluation_plan.json"
                lock = json.loads(path.read_text(encoding="utf-8"))
                lock["approval"]["approved"] = False
                path.write_bytes(json.dumps(lock, indent=1).replace("\n", "\r\n").encode("utf-8"))
            else:
                tamper(root)
            mock = Evaluate()
            with FG.ReadRecorder() as recorder:
                message = refused(run, root, True, mock)
            assert expected in message, (name, message)
            assert mock.calls == [] and recorder.log == [] and not (root / "results" / "factor_evaluation").exists(), name
    # a record whose hashes were edited does not match either
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(tmp)
        approve(root)
        path = root / "results" / "factor_discovery" / PC.AUTH_NAME
        auth = json.loads(path.read_text(encoding="utf-8"))
        auth["lock_scientific_sha256"] = "0" * 64
        path.write_text(json.dumps(auth), encoding="utf-8")
        mock = Evaluate()
        refused(run, root, True, mock)
        assert mock.calls == []


# ------------------------------------------------------------------ 5. one shot, no overwrite
def test_the_evaluation_entry_is_called_once_and_nothing_is_overwritten_or_repeated():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(tmp)
        approve(root)
        mock = Evaluate()
        run(root, True, mock)
        out_dir = root / "results" / "factor_evaluation"
        snapshot = {p.name: sha(p.read_bytes()) for p in out_dir.iterdir()}
        assert "already started" in refused(run, root, True, mock)
        assert len(mock.calls) == 1 and snapshot == {p.name: sha(p.read_bytes()) for p in out_dir.iterdir()}
        # removing only the state file does not allow an overwrite of the tables
        (out_dir / PC.STATE_NAME).unlink()
        assert "already started" in refused(run, root, True, mock)
        assert len(mock.calls) == 1
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(tmp)
        approve(root)
        dry = root / "results" / "factor_evaluation" / "dry_run"
        mock = Evaluate()
        assert "dry-run" in refused(run, root, True, mock, dry)
        assert mock.calls == [] and not dry.exists()


# ------------------------------------------------------------------ 6. an exception is never retried
def test_an_exception_inside_the_evaluation_is_not_retried_and_leaves_a_failed_state():
    with tempfile.TemporaryDirectory() as tmp:
        root = make_root(tmp)
        approve(root)
        mock = Evaluate(fail=True)
        try:
            run(root, True, mock)
        except ValueError as error:
            assert "simulated failure" in str(error)
        else:
            raise AssertionError("the exception was swallowed")
        out_dir = root / "results" / "factor_evaluation"
        state = json.loads((out_dir / PC.STATE_NAME).read_text(encoding="utf-8"))
        assert state["status"] == "FAILED" and "simulated failure" in state["error"] and [p.name for p in out_dir.iterdir()] == [PC.STATE_NAME]
        mock.fail = False
        assert "already started" in refused(run, root, True, mock)                                 # not even with a working evaluation
        assert len(mock.calls) == 1


# ------------------------------------------------------------------ 8. the interpretation columns
def test_inference_labels_only_add_columns_and_leave_every_value_unchanged():
    sys.path.insert(0, str(ROOT / "tests"))
    from test_factor_evaluation import CONFIGS, _table      # synthetic tables of the existing engine test

    table = _table()
    ev = FE.Evaluator(table, CONFIGS, [], {}, "validation")
    comparisons = [{"id": "A:P1", "evaluation_protocol": "p", "family": "primary", "role": "confirmatory (Holm family of 4)", "from_config": "base", "to_config": "base + X"},
                   {"id": "A:P2", "evaluation_protocol": "p", "family": "primary", "role": "confirmatory (Holm family of 4)", "from_config": "base", "to_config": "base + X (heavy penalty)"},
                   {"id": "A:E8-1", "evaluation_protocol": "p", "family": "E8 minimal", "role": "exploratory validity sensitivity", "from_config": "base", "to_config": "base + X"},
                   {"id": "A:E1-1", "evaluation_protocol": "p", "family": "E1", "role": "pre-specified secondary (descriptive)", "from_config": "base", "to_config": "base + X"}]
    labeled = FE.run_comparisons(ev, comparisons, None, 200, strata_ids=("P1",))
    original = FE.add_inference_labels
    try:
        FE.add_inference_labels = lambda frame: frame                                              # the function as it was before the columns were added
        plain = FE.run_comparisons(ev, comparisons, None, 200, strata_ids=("P1",))
    finally:
        FE.add_inference_labels = original
    assert list(labeled.columns) == list(plain.columns) + FE.INFERENCE_LABEL_COLUMNS
    pd.testing.assert_frame_equal(labeled[list(plain.columns)], plain, check_exact=True)       # not a single value, p-value or Holm value changed
    assert (labeled["nominal_fwer_control_verified"] == False).all() and (labeled["test_split_independent"] == False).all()      # noqa: E712
    roles = labeled.groupby("id")["inference_role"].first().to_dict()
    assert roles == {"A:P1": "primary descriptive", "A:P2": "primary descriptive", "A:E8-1": "exploratory sensitivity", "A:E1-1": "secondary descriptive"}
    has_p = labeled["p_bootstrap"].notna() | labeled["p_holm_primary_family"].notna()
    assert (labeled.loc[has_p, "p_value_method"] == "exploratory approximate subject-bootstrap").all() and (labeled.loc[~has_p, "p_value_method"] == "").all() and has_p.any()
    assert (labeled["role"] == plain["role"]).all()                                               # the program's role field is unchanged
    assert not len(FE.add_inference_labels(pd.DataFrame()))


# ------------------------------------------------------------------ 9. Static default paths and isolated publication logic
def test_delivery_defaults_have_no_active_computation_or_publication_calls():
    """Inspect current defaults without executing either notebook or its imports."""
    class Defaults(ast.NodeVisitor):
        def __init__(self):
            self.flags, self.calls = {}, set()

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        self.flags[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
            self.generic_visit(node)

        def visit_If(self, node):
            # Fold literal configuration only; inspect both arms of data-dependent checks.
            branches = node.body + node.orelse
            if not any(isinstance(n, ast.Call) for n in ast.walk(node.test)):
                try:
                    chosen = eval(compile(ast.Expression(node.test), "guard", "eval"), {"__builtins__": {}}, self.flags)
                    branches = node.body if chosen else node.orelse
                except NameError:
                    pass
            self.visit(node.test)
            for statement in branches:
                self.visit(statement)

        def visit_FunctionDef(self, node):
            pass  # definitions do not execute their bodies

        def visit_Call(self, node):
            self.calls.add(ast.unparse(node.func).split(".")[-1])
            self.generic_visit(node)

    forbidden = {"fit", "fit_transform", "predict", "run_authorized_test", "run_fixed_test", "run_search", "exec", "load_context", "load_namespace",
                 "assemble_protocol", "evaluate_protocol", "read_parquet", "dataset", "load", "mkdir", "to_csv", "to_parquet", "write_text", "write_bytes", "run"}
    for filename, selector, expected in [("08_factor_discovery.ipynb", "REBUILD_TARGET", None), ("09_factor_evaluation.ipynb", "REBUILD_VALIDATION", False)]:
        nb = nbformat.read(ROOT / "notebooks" / filename, as_version=4)
        nbformat.validate(nb)
        visitor = Defaults()
        for cell in nb.cells:
            if cell.cell_type == "code":
                visitor.visit(ast.parse(cell.source))
        assert selector in visitor.flags and visitor.flags[selector] is expected
        assert visitor.flags.get("EXECUTE_TEST", False) is False
        assert not visitor.calls & forbidden, visitor.calls & forbidden
        assert {"read_csv", "display"} <= visitor.calls


def fragment(filename, cell_ids, namespace):
    """Wrap selected branch bodies for synthetic fixtures; no notebook initialization."""
    cells = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))["cells"]
    by_id = {cell["id"]: cell for cell in cells}
    body = []
    for cid in cell_ids:
        for node in ast.parse("".join(by_id[cid]["source"])).body:
            body.extend(node.body if isinstance(node, ast.If) else [node])
    wrapper = ast.parse("def isolated():\n    pass")
    wrapper.body[0].body = body + ast.parse("return locals()").body
    exec(compile(ast.fix_missing_locations(wrapper), "synthetic_fragment", "exec"), namespace)
    return namespace["isolated"]()


def test_selected_factor_display_rejects_short_long_and_wrong_k_lists():
    for bad, length in [(None, 2), ("T", 2), ("O", 1), ("O", 3), ("L", 1), ("L", 3), ("R", 1), ("R", 3)]:
        shown = []
        def selections(path):
            method = path.stem[-1]
            expressions = [f"e{i}" for i in range(length if method == bad else 2)]
            return pd.DataFrame([{"evaluation_protocol": "A", "selected_set": '["f0", "f1"]',
                                  "k_selected": 3 if bad == method == "T" else 2,
                                  "selected_formulas": json.dumps(expressions), "selected_expressions": json.dumps(expressions)}])
        ns = dict(pd=pd, json=json, SAVED_DIR=Path("synthetic"), PROTOCOL_A="A", display=shown.append)
        with patch.object(pd, "read_csv", side_effect=selections):
            if bad:
                with TestCase().assertRaises(ValueError):
                    fragment("08_factor_discovery.ipynb", ["review-factors"], ns)
                assert not shown
            else:
                fragment("08_factor_discovery.ipynb", ["review-factors"], ns)
                assert shown[0]["candidate_id"].tolist() == ["f0", "f1"] * 4
                assert shown[0]["expression"].tolist() == ["f0", "f1"] + ["e0", "e1"] * 3


def test_openfe_pool_validates_all_formulas_and_both_protocols_before_writing():
    from src.openfe_runner import evaluate_formula, formula_columns

    for failure, error in [("columns", ValueError), ("syntax", ValueError), ("evaluation", KeyError), (None, None)]:
        events, published = [], {}
        def ranking(path):
            assert path.parent == Path("new_rankings")
            extra = {"columns": "forbidden", "syntax": "x)"}.get(failure, "x") if path.stem.endswith("B") else "x"
            return pd.DataFrame({"rank": [1, 26], "formula": ["(x+x)", extra]})
        def inputs(name):
            events.append(name)
            frame = pd.DataFrame({"subject": [1, 1], "origin_time": [0, 1], "task_id": ["t", "t"], "split": ["train", "validation"], "x": [1., 2.]})
            return frame.drop(columns="x") if failure == "evaluation" and name == "B" else frame
        def publish(frame, path, **kwargs):
            events.append("write")
            published[path.name] = frame.copy() if isinstance(frame, pd.DataFrame) else frame
        ns = dict(pd=pd, FACTOR_DIR=Path("synthetic"), OPENFE_RANKING_DIR=Path("new_rankings"), PROTOCOL_A="A", PROTOCOL_B="B", OPENFE_COLUMNS=["x"],
                  WINDOW_KEY=["subject", "origin_time", "task_id"], window_inputs=inputs, formula_columns=formula_columns, evaluate_formula=evaluate_formula)
        with patch.object(pd, "read_csv", side_effect=ranking), patch.object(pd.DataFrame, "to_csv", publish),\
             patch.object(pd.DataFrame, "to_parquet", publish), patch.object(Path, "write_text", side_effect=AssertionError("unexpected JSON write")):
            if error:
                with TestCase().assertRaises(error):
                    fragment("08_factor_discovery.ipynb", ["2a30b72a"], ns)
                assert not published and "write" not in events
            else:
                fragment("08_factor_discovery.ipynb", ["2a30b72a"], ns)
                assert events[:3] == ["A", "B", "write"]
                values = published["openfe_values.parquet"]
                assert set(published) == {"openfe_features.csv", "openfe_values.parquet"}
                assert values["evaluation_protocol"].tolist() == ["A", "A", "B", "B"]
                assert values["O_A_0001"].dtype == np.dtype("float32") and values["O_A_0001"].dropna().tolist() == [2., 4.]
                assert len(published["openfe_features.csv"]) == 4


def test_validation_csvs_are_published_only_after_reference_and_strata_checks():
    for failure in ("reference", "strata", None):
        published, events = {}, []
        def evaluate(tag, *args):
            row = {"id": tag + ":P1", "cohort": "core", "stratum": "all", "group": "all", "min_window_rule": "applied",
                   "d_rmse": -0.1, "d_rmse_low": -0.2, "d_rmse_high": 0.0}
            return pd.DataFrame([row]), pd.DataFrame({"config": ["base"]})
        def compare_validation_results(frame, *args):
            return pd.DataFrame({"status": ["MISMATCH" if failure == "reference" else "match"]})
        def strata(tag):
            events.append(tag)
            return {"one": pd.Series([True]), "two": pd.Series([failure == "strata" and tag == "B"])}
        def publish(frame, path, **kwargs):
            events.append("write")
            published[path.name] = frame.copy()
        ns = dict(pd=pd, FE=SimpleNamespace(evaluate_protocol=evaluate, strata_masks=strata, N_RESAMPLES=2000),
                  FG=SimpleNamespace(compare_validation_results=compare_validation_results), assembled={tag: {"table": tag} for tag in ("A", "B")}, evaluation_plan={}, FACTOR_DIR=Path("synthetic"),
                  out=Path("synthetic"), display=lambda frame: None)
        with patch.object(pd.DataFrame, "to_csv", publish):
            if failure:
                with TestCase().assertRaises(ValueError):
                    fragment("09_factor_evaluation.ipynb", ["bb8a2b5a", "5f4e3b19"], ns)
                assert not published
            else:
                fragment("09_factor_evaluation.ipynb", ["bb8a2b5a", "5f4e3b19"], ns)
                assert events == ["A", "B", "write", "write"]
                assert set(published) == {"validation_contrasts.csv", "validation_summary.csv"}
                assert published["validation_contrasts.csv"]["status"].eq("VALIDATION reconstruction; not a new TEST result").all()



def test_factor_description_marks_undefined_correlations_as_unavailable():
    nb = json.loads((ROOT / "notebooks/08_factor_discovery.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in nb["cells"] if c.get("id") == "771265ae")
    tree = ast.parse(source)
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"finite_rho", "partner", "correlation"} for t in n.targets)]
    fields = next(n for n in ast.walk(tree) if isinstance(n, ast.Dict) and any(isinstance(k, ast.Constant) and k.value == "most_correlated_existing_factor" for k in n.keys))
    flag = next(v for k, v in zip(fields.keys, fields.values) if isinstance(k, ast.Constant) and k.value == "near_duplicate_of_existing_factor")
    for rho, expected in [({}, None), ({"t": np.nan}, None), ({"t": 0.96}, "t")]:
        ns = dict(rho=rho, np=np, pd=pd, existing={"t": "T: t"})
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "synthetic_correlations", "exec"), ns)
        assert ns["partner"] == expected
        result = eval(compile(ast.Expression(flag), "synthetic_duplicate_flag", "eval"), ns)
        assert pd.isna(result) if expected is None else result is True


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
