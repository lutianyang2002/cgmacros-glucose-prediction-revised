"""Guard a single approved TEST evaluation and preserve its execution evidence.

Approval, version and input checks precede evaluation. Exclusive creation prevents
overwriting outputs; every attempted run keeps its COMPLETED or FAILED state.
Historical permissions are archived separately and do not authorize the current code."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import window_data as W

AUTH_NAME = "execution_permission.json"
STATE_NAME = "execution_state.json"
OUTPUT_FILES = ("comparison_results.csv", "configuration_metrics.csv", "meal_strata.csv")
NON_SCIENTIFIC_KEYS = ("status", "approval", "revision", "access_checks", "decisions", "provenance")        # authorization state, history and hashes; everything else is the approved scientific content
RESULT_STATUS = "TEST evaluation (previously used test split; exploratory approximate bootstrap p-values; not an independent hold-out)"


class ExecutionRefused(RuntimeError):
    """The requested state or action is not allowed; raised before any data file is read."""


# ------------------------------------------------------------------ hashes
def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_sha256(lock: dict) -> str:
    """SHA-256 of the canonical JSON of the lock without its authorization state, history and hashes."""
    content = {k: v for k, v in lock.items() if k not in NON_SCIENTIFIC_KEYS}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def notebook_code_sha256(path) -> str:
    """SHA-256 of the sources of the code cells of a notebook (outputs and markdown are not part of it, so the in-place execution does not change it)."""
    cells = json.loads(Path(path).read_text(encoding="utf-8"))["cells"]
    return hashlib.sha256(json.dumps(["".join(c["source"]) for c in cells if c["cell_type"] == "code"], ensure_ascii=False).encode("utf-8")).hexdigest()


def _input_path(root: Path, key: str) -> Path:
    return root / key if "/" in key else root / "results" / "factor_discovery" / key


# ------------------------------------------------------------------ state
def load_authorization(path) -> dict | None:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def resolve_mode(lock: dict, auth: dict | None, execute_test: bool) -> str:
    """Determine permitted operation from the supplied plan and permission dictionaries."""
    lock_approved = lock.get("approval", {}).get("approved") is True
    auth_approved = bool(auth) and auth.get("approval", {}).get("approved") is True
    if lock_approved != auth_approved:
        raise ExecutionRefused("the lock and the authorization record disagree about lock approval; nothing is executed")
    if not lock_approved:
        if execute_test:
            raise ExecutionRefused("EXECUTE_TEST is True but lock approval is not approved and recorded; nothing is read")
        return "validation"
    return "authorized_test" if execute_test else "approved_idle"


def verify_for_test(root, lock: dict, auth: dict | None, execute_test: bool, notebook_path, out_dir) -> dict:
    """Conditions of the approved TEST evaluation, in order. Raises `ExecutionRefused` at the first failure; the input files are hashed last (the only step that opens a data file, byte level)."""
    root, out_dir = Path(root), Path(out_dir)
    if not execute_test:
        raise ExecutionRefused("EXECUTE_TEST is False")
    if resolve_mode(lock, auth, execute_test) != "authorized_test":
        raise ExecutionRefused("lock approval is not approved and recorded")
    if not str(lock.get("status", "")).startswith("LOCKED"):
        raise ExecutionRefused("the lock is not marked as in force (status must start with LOCKED)")
    approval, execution = auth.get("approval", {}), auth.get("execution", {})
    if not (approval.get("date") and approval.get("by")):
        raise ExecutionRefused("the lock approval record has no date or approver")
    if execution.get("execution_authorized") is not True or not (execution.get("date") and execution.get("by")):
        raise ExecutionRefused("authorized test evaluation has no separate execution authorization")
    if lock.get("execute_test_default") is not False:
        raise ExecutionRefused("the lock must keep execute_test_default False")
    if scientific_sha256(lock) != auth.get("lock_scientific_sha256"):
        raise ExecutionRefused("the scientific content of the lock differs from the approved version")
    protocol = lock["protocol"]
    if not (sha256_file(root / protocol["file"]) == protocol["sha256"] == auth.get("protocol_sha256")):
        raise ExecutionRefused("the input protocol differs from the approved one")
    code = auth.get("code_sha256") or {}
    if not code or set(lock["provenance"]["code_sha256"]) - set(code):
        raise ExecutionRefused("the registered code versions do not cover the code files listed in the lock")
    for rel, digest in code.items():
        if not (root / rel).is_file() or sha256_file(root / rel) != digest or lock["provenance"]["code_sha256"].get(rel, digest) != digest:
            raise ExecutionRefused(f"code version differs from the registered one: {rel}")
    if notebook_code_sha256(notebook_path) != auth.get("notebook_code_cells_sha256"):
        raise ExecutionRefused("the code cells of factor-evaluation notebook differ from the registered ones")
    if (out_dir / STATE_NAME).exists() or any((out_dir / name).exists() for name in OUTPUT_FILES):
        raise ExecutionRefused("a TEST evaluation was already started or written; it is never repeated automatically")
    inputs = auth.get("inputs_sha256") or {}
    if inputs != lock["provenance"]["inputs_sha256"]:
        raise ExecutionRefused("the registered inputs differ from the lock")
    for rel, digest in inputs.items():
        if sha256_file(_input_path(root, rel)) != digest:
            raise ExecutionRefused(f"input differs from the registered one: {rel}")
    return {"lock_scientific_sha256": auth["lock_scientific_sha256"], "code_files_verified": len(code), "inputs_verified": len(inputs), "approval": approval, "execution": execution}


# ------------------------------------------------------------------ single-run TEST evaluation
def default_evaluate(lock: dict, access: W.Access) -> dict:
    from src import factor_evaluation as FE

    return FE.run_fixed_evaluation(lock, "test", access)


def write_outputs(result: dict, out_dir, lock: dict) -> dict:
    """The three tables of authorized test evaluation (never overwritten): rows of the whole-cohort contrasts with the HbA1c groups, the meal strata, and the configuration summary."""
    out_dir = Path(out_dir)
    contrasts = pd.concat([r["contrasts"].assign(protocol_tag=tag) for tag, r in result.items()], ignore_index=True)
    summary = pd.concat([r["summary"].assign(protocol_tag=tag) for tag, r in result.items()], ignore_index=True)
    tables = {"comparison_results.csv": contrasts[contrasts["stratum"].eq("all")], "meal_strata.csv": contrasts[~contrasts["stratum"].eq("all")], "configuration_metrics.csv": summary}
    hashes = {}
    for name, frame in tables.items():
        path = out_dir / name
        with open(path, "x", newline="", encoding="utf-8") as handle:
            frame.assign(status=RESULT_STATUS).to_csv(handle, index=False)
        hashes[name] = sha256_file(path)
    return hashes


def run_authorized_test(lock: dict, root, execute_test: bool, notebook_path=None, out_dir=None, auth_path=None, evaluate=None, now=None) -> dict:
    """Verify, then call the evaluation entry exactly once and write the tables. No retry: the state file is created exclusively before the evaluation and kept whatever happens."""
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root / "results" / "factor_evaluation"
    if "dry_run" in out_dir.parts:
        raise ExecutionRefused("the validation reconstruction dry-run records are never used as an output location")
    auth = load_authorization(auth_path or root / "results" / "factor_discovery" / AUTH_NAME)
    notebook_path = Path(notebook_path or root / "notebooks" / "09_factor_evaluation.ipynb")
    evidence = verify_for_test(root, lock, auth, execute_test, notebook_path, out_dir)
    stamp = (now or (lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")))
    access = W.Access(execute_test=True, lock_approved=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / STATE_NAME
    record = {"status": "STARTED", "started": stamp(), "evidence": evidence}
    with open(state_path, "x", encoding="utf-8") as handle:                               # a second attempt cannot start
        json.dump(record, handle, indent=1)
    try:
        result = (evaluate or default_evaluate)(lock, access)
        record["output_sha256"] = write_outputs(result, out_dir, lock)
    except BaseException as error:                                                         # noqa: BLE001  (never retried; the state file keeps the failure)
        record.update(status="FAILED", finished=stamp(), error=f"{type(error).__name__}: {error}")
        state_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        raise
    record.update(status="COMPLETED", finished=stamp())
    state_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return {"state": record, "result": result}
