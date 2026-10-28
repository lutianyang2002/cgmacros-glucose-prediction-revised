"""Compare reconstructed factors and validation statistics against saved references.

Feature values, missingness and window keys are checked independently of the
statistical comparison. Access restrictions are enforced by window_data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import window_data as W

TOL = 1e-9
WINDOW_KEY = W.WINDOW_KEY


# ------------------------------------------------------------------ factor values
def _compare(name: str, rebuilt: np.ndarray, saved: np.ndarray, tol: float) -> dict:
    a, b = np.asarray(rebuilt, float), np.asarray(saved, float)
    same_missing = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    both = ~np.isnan(a) & ~np.isnan(b)
    worst = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
    return {"check": name, "values": int(len(a)), "max_abs_difference": worst, "same_missing_pattern": same_missing, "tolerance": tol, "passed": bool(same_missing and worst <= tol)}


def compare_factor_values(tag: str, assembled: dict, lock: dict, art: Path, minutes_events: tuple | None = None) -> pd.DataFrame:
    """Rebuilt against saved features. `minutes_events` = (keys, minutes, events) lets the logged-time diet features be rebuilt as well."""
    name, table = assembled["name"], assembled["table"]
    keys = [*WINDOW_KEY, "split"]
    rows = []
    saved_resid = pd.read_parquet(art / "base_residuals.parquet", filters=[("evaluation_protocol", "==", name)])
    merged = table[keys + ["target_value", "base_prediction", "residual"]].merge(saved_resid[keys + ["target_value", "base_prediction", "residual"]], on=keys, how="left", suffixes=("", "_saved"), validate="one_to_one")
    rows.append({"check": "window keys and count match after key alignment", "values": len(table), "max_abs_difference": 0.0, "same_missing_pattern": True, "tolerance": 0,
                 "passed": bool(len(table) == len(saved_resid) and merged["target_value_saved"].notna().all())})
    for c in ("target_value", "base_prediction", "residual"):
        rows.append(_compare(f"{c} (base_residuals.parquet)", merged[c], merged[f"{c}_saved"], TOL))
    ids = lock["features"]["T"][tag]["ids_in_selection_order"]
    for part, label in (("candidates_batch1.parquet", "T candidates batch 1"), ("candidates_batch2.parquet", "T candidates batch 2")):
        saved = pd.read_parquet(art / part, filters=[("evaluation_protocol", "==", name)])
        wanted = [c for c in ids if c in saved.columns]
        if not wanted:
            continue
        m = table[keys].merge(saved[keys + wanted], on=keys, how="left", validate="one_to_one")
        for c in wanted:
            rows.append(_compare(f"{label}: {c}", table[c], m[c], TOL))
    o_ids = list(lock["features"]["O"][tag]["formulas"])
    openfe = pd.read_parquet(art / "openfe_values.parquet", filters=[("evaluation_protocol", "==", name)])
    m = table[keys].merge(openfe[keys + o_ids], on=keys, how="left", validate="one_to_one")
    for c in o_ids:
        rows.append(_compare(f"O_k (openfe_values.parquet): {c}", table[c], m[c], 1e-6))
    prim = pd.read_parquet(art / f"primitives_{tag}.parquet")
    columns = list(dict.fromkeys(["native_missing"] + [c for c in lock["features"]["openfe_columns_from_primitives"] if c in prim.columns]))
    m = table[keys].merge(prim[keys + columns], on=keys, how="left", validate="one_to_one")
    inputs = assembled["inputs"]
    for c in columns:
        values = inputs.loc[:, c] if c in inputs.columns else table[c]
        if values.ndim != 1:
            raise ValueError(f"column {c} appears twice in the OpenFE input frame")
        rows.append(_compare(f"primitive: {c}", values, m[c], TOL))
    if minutes_events is not None:
        k, minutes, events = minutes_events
        d_columns = assembled["groups"]["D"]
        diet = W.diet_matrix(k, minutes, events, d_columns, "logged")
        for c in d_columns:
            rows.append(_compare(f"logged-time diet feature rebuilt from the minutes and events: {c}", diet[c], table[c], TOL))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ validation references
def _reference(art: Path, ref: dict, protocol: str, frames: dict) -> pd.DataFrame | None:
    filename = ref["artifact"]
    if filename not in frames:
        frames[filename] = pd.read_csv(art / filename)
    frame = frames[filename]
    if "evaluation_protocol" in frame.columns:
        frame = frame[frame["evaluation_protocol"] == protocol]
    for key in ("from_config", "to_config", "contrast_id", "cohort", "family_removed", "record_type", "setting", "model", "contrast_type", "lags", "block", "from", "to"):
        if key in ref and key in frame.columns:
            frame = frame[frame[key] == ref[key]]
    return frame


def compare_validation_results(contrasts: pd.DataFrame, lock: dict, art: Path, references: dict | None = None) -> pd.DataFrame:
    """Compare core validation contrasts under both window rules.

    Reference numerical fields must be finite on both sides; missing required fields
    or different outcome labels are mismatches. Derived references compare points only.
    """
    references = {} if references is None else references
    ours = contrasts[(contrasts["cohort"] == "core") & (contrasts["stratum"] == "all") & (contrasts["group"] == "all")]
    out = []
    for c in lock["comparisons"]:
        if c["id"] not in set(contrasts["id"]):
            continue
        ref = c["validation_reference"]
        for rule in ("applied (>= 20 windows)", "not applied"):
            row = ours[(ours["id"] == c["id"]) & (ours["min_window_rule"] == rule)]
            if row.empty:
                out.append({"id": c["id"], "rule": rule, "status": "not computed"})
                continue
            row = row.iloc[0]
            record = {"id": c["id"], "rule": rule, "reference": ref["artifact"] if ref else None}
            if ref is None:
                record["status"] = "no reference"
            elif "derived" in ref:
                saved = _reference(art, ref, c["evaluation_protocol"], references)
                saved = saved[saved["min_window_rule"].astype(str).str.startswith("applied" if rule.startswith("applied") else "not")]
                expected = float(saved["rmse_to"].iloc[0]) - float(_reference(art, {"artifact": ref["artifact"], "from_config": "base", "to_config": "base + M"}, c["evaluation_protocol"], references).pipe(
                    lambda f: f[f["min_window_rule"].astype(str).str.startswith("applied" if rule.startswith("applied") else "not")]).iloc[0]["rmse_from"])
                record.update({"fields": "rmse_to - rmse_from (point estimate only; no saved interval)", "max_abs_difference": abs(expected - float(row["d_rmse"])), "status": "match (point estimate)" if abs(expected - float(row["d_rmse"])) < 1e-9 else "MISMATCH"})
            else:
                saved = _reference(art, ref, c["evaluation_protocol"], references)
                saved = saved[saved["min_window_rule"].astype(str).str.startswith("applied" if rule.startswith("applied") else "not")] if "min_window_rule" in saved.columns else saved
                if len(saved) != 1:
                    record["status"] = f"reference rows found: {len(saved)}"
                else:
                    saved = saved.iloc[0]
                    fields = ("subjects", "validation_windows", "rmse_from", "rmse_to", "d_rmse", "d_rmse_low", "d_rmse_high", "improved_share", "pooled_d_rmse", "d_mae")
                    diffs = {f: abs(float(row[f]) - float(saved[f])) if f in row.index and np.isfinite([row[f], saved[f]]).all() else np.inf
                             for f in fields if f in saved.index}
                    required = {"subjects", "rmse_from", "rmse_to", "d_rmse"}
                    worst = max(diffs.values(), default=np.inf) if required <= diffs.keys() else np.inf
                    same_label = "outcome" not in saved.index or (pd.notna(saved["outcome"]) and pd.notna(row.get("outcome")) and saved["outcome"] == row["outcome"])
                    record.update({"fields": ", ".join(diffs), "max_abs_difference": worst, "worst_field": max(diffs, key=diffs.get) if diffs else None, "same_label": bool(same_label),
                                   "status": "match" if worst < 1e-8 and same_label else "MISMATCH"})
            out.append(record)
    return pd.DataFrame(out)


class ReadRecorder:
    """Record pandas read_csv/read_parquet and Arrow dataset calls; other reads are outside coverage."""

    def __enter__(self):
        import pandas as pd_
        import pyarrow.dataset as ds_

        self.log, self._orig = [], {}
        recorder = self

        def wrap(module, name, label):
            original = getattr(module, name)
            recorder._orig[(module, name)] = original

            def wrapped(*args, **kwargs):
                path = args[0] if args else kwargs.get("path") or kwargs.get("path_or_buf") or kwargs.get("source")
                recorder.log.append({"call": label, "path": str(path), "filters": str(kwargs.get("filters"))[:200]})
                return original(*args, **kwargs)
            setattr(module, name, wrapped)

        wrap(pd_, "read_parquet", "pandas.read_parquet")
        wrap(pd_, "read_csv", "pandas.read_csv")
        wrap(ds_, "dataset", "pyarrow.dataset.dataset")
        return self

    def __exit__(self, *exc):
        for (module, name), original in self._orig.items():
            setattr(module, name, original)
        return False


def data_range_check(tag: str, assembled: dict, root: Path, eval_split: str = "validation") -> dict:
    """Splits present, and for every subject the last minute and meal event that were loaded against the last admitted origin and (Protocol A) the end of the validation period."""
    table, minutes, events = assembled["table"], assembled["minutes"], assembled["events"]
    out = {"splits_present": sorted(table["split"].unique()), "only_allowed_splits": bool(set(table["split"]) <= {"train", eval_split}), "subjects": int(table["subject"].nunique())}
    last_origin, last_minute = assembled["last_origin"], assembled["last_minute_loaded"]
    last_event = pd.Series({s: e["timestamp"].max() for s, e in events.items() if len(e)})
    out["minutes_loaded_not_later_than_the_last_origin"] = bool((last_minute.reindex(last_origin.index) <= last_origin).all())
    out["meal_events_loaded_not_later_than_the_last_origin"] = bool((last_event <= last_origin.reindex(last_event.index)).all())
    if tag == "A":
        scope = pd.read_csv(root / "results" / "data_quality" / "split_scope_v1.csv", parse_dates=["protocol_a_val_end_time"]).set_index("subject_id")
        out["last_origin_not_later_than_the_validation_end_of_protocol_A"] = bool((last_origin <= scope["protocol_a_val_end_time"].reindex(last_origin.index)).all())
    else:
        scope = pd.read_csv(root / "results" / "data_quality" / "split_scope_v1.csv")
        out["no_protocol_B_test_subject_loaded"] = bool(not set(table["subject"]) & set(scope.loc[scope["protocol_b_group"] == "test", "subject_id"]))
    return out
