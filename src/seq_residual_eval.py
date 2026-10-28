"""Fixed-epoch sequence fitting and fixed-factor transfer on development data.

Historical checkpoint paths retain their original names. Existing configuration
and checkpoint identity checks prevent mixing incompatible training runs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from src import seq_models as seq
from src.residuals import (RIDGE_ALPHAS, best_alpha, contrast_row, grouped_folds,
                           ridge_path_predictions, subject_table, time_block_folds)
from src.window_data import load_keys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/sequence_transfer"
INPUTS = ROOT / "results/sequence_transfer"
KEYS = seq.KEYS
LOCK = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    writer(temporary)
    os.replace(temporary, path)


def save_json(path, value):
    with LOCK:
        atomic(path, lambda p: p.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8"))


def save_frame(path, frame):
    if not set(frame.get("split", pd.Series(dtype=str))) <= {"train", "validation"}:
        raise ValueError("unauthorized output split")
    atomic(path, lambda p: frame.to_parquet(p, index=False))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def development_parquet(path, columns, protocol=None):
    """Certify split metadata before loading these saved development-only files."""
    footer = pq.ParquetFile(path)
    j = footer.schema_arrow.get_field_index("split")
    if j < 0:
        raise ValueError(f"no split registry in {path}")
    for r in range(footer.num_row_groups):
        s = footer.metadata.row_group(r).column(j).statistics
        if s is None or s.min not in {"train", "validation"} or s.max not in {"train", "validation"}:
            raise ValueError(f"not certified development-only: {path}")
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
    if protocol is not None:
        frame["evaluation_protocol"] = protocol
    seq.require_development(frame)
    return frame


def load_inputs(config):
    windows = pd.read_parquet(INPUTS / "window_keys_targets_trainval.parquet")
    seq.require_development(windows)
    history = np.load(INPUTS / "glucose_history_trainval.npy", allow_pickle=False)
    if history.shape != (len(windows), 25) or not np.isfinite(history).all():
        raise ValueError("invalid cached sequence matrix")
    windows["row_id"] = np.arange(len(windows))
    fresh = pd.concat([load_keys(p, ("train", "validation")) for p in seq.PROTOCOLS.values()], ignore_index=True)
    for col in KEYS + ["split", "history_start_time", "target_time"]:
        if not windows[col].reset_index(drop=True).equals(fresh[col].reset_index(drop=True)):
            raise ValueError(f"cached window mismatch: {col}")
    if not np.array_equal(config["alpha_grid"], RIDGE_ALPHAS):
        raise ValueError("alpha grid differs from existing implementation")
    manual = config["factor_columns"]["A"]["M"]
    reference_path = ROOT / "results/laboratory_interactions/reference_trainval.parquet"
    # Read-only reuse of the isolated M50/cohort copy; no C model is run.
    reference = development_parquet(reference_path, KEYS + ["split", "target_value", "base_prediction", "residual", "oof_fold", "cohort_core"] + manual)
    if len(reference) != len(windows):
        raise ValueError("reference window count differs")
    merged = windows.merge(reference.rename(columns={"target_value": "reference_target"}), on=KEYS + ["split"],
                           how="left", sort=False, validate="one_to_one")
    if not np.array_equal(merged.target_value.to_numpy(), merged.reference_target.to_numpy()):
        raise ValueError("reference labels do not match cached labels")
    needed = set(c for p in ["A", "B"] for k in ["T", "O"] for c in config["factor_columns"][p][k])
    for name in ["candidates_batch1.parquet", "candidates_batch2.parquet", "openfe_values.parquet"]:
        path = ROOT / "results/factor_discovery" / name
        selected = [c for c in pq.ParquetFile(path).schema_arrow.names if c in needed and c not in merged.columns]
        if selected:
            values = development_parquet(path, KEYS + ["split"] + selected)
            merged = merged.merge(values, on=KEYS + ["split"], how="left", sort=False, validate="one_to_one")
    native = []
    for p in ["A", "B"]:
        native.append(development_parquet(ROOT / f"results/factor_discovery/primitives_{p}.parquet",
                      ["subject", "origin_time", "task_id", "split", "native_missing"], seq.PROTOCOLS[p]))
    merged = merged.merge(pd.concat(native, ignore_index=True), on=KEYS + ["split"], how="left", sort=False, validate="one_to_one")
    if len(merged) != len(windows) or not np.array_equal(merged.row_id.to_numpy(), windows.row_id.to_numpy()):
        raise ValueError("factor joins changed sequence row order")
    if merged.cohort_core.isna().any() or merged.native_missing.isna().any():
        raise ValueError("missing cohort/native flags")
    for p in ["A", "B"]:
        for columns in config["factor_columns"][p].values():
            if len(set(columns)) != len(columns) or set(columns) - set(merged.columns):
                raise ValueError("selected feature columns incomplete or duplicated")
    return merged.drop(columns="reference_target"), history


def assert_no_support_overlap(train, heldout):
    for subject, part in heldout.groupby("subject"):
        other = train.loc[train.subject.eq(subject)]
        if not ((other.target_time < part.history_start_time.min()) |
                (other.history_start_time > part.target_time.max())).all():
            raise ValueError("overlapping 150-minute support between fit and heldout")


def make_fold_plan(frame, config):
    scope = pd.read_csv(ROOT / "results/data_quality/split_scope_v1.csv").set_index("subject_id")
    bounds = pd.DataFrame({"lo": pd.to_datetime(scope.start_time), "hi": pd.to_datetime(scope.protocol_a_train_end_time)})
    records, audit = [], []
    for p in ["A", "B"]:
        part = frame.loc[frame.evaluation_protocol.eq(seq.PROTOCOLS[p])].copy()
        tr = part.split.eq("train")
        train = part.loc[tr]
        folds = time_block_folds(train, bounds) if p == "A" else grouped_folds(train.subject)
        if not np.array_equal(folds, train.oof_fold.to_numpy()):
            raise ValueError(f"{p} folds differ from historical OOF registry")
        if p == "A":
            fixed_edges = config["oof_specification"]["protocol_A"]["block_edges"]
            for subject in train.subject.unique():
                lo, hi = bounds.loc[subject, ["lo", "hi"]]
                edges = [lo + (hi - lo) * k / 5 for k in range(6)]
                if [pd.Timestamp(value) for value in fixed_edges[str(subject)]] != edges:
                    raise ValueError("A block edges differ from configured specification")
        fold_column = np.full(len(part), -2, dtype=int)
        fold_column[tr.to_numpy()] = folds
        part["sequence_fold"] = fold_column
        for f in range(5):
            fit = part.loc[part.split.eq("train") & part.sequence_fold.ge(0) & part.sequence_fold.ne(f)]
            held = part.loc[part.split.eq("train") & part.sequence_fold.eq(f)]
            if set(fit.row_id) & set(held.row_id) or not len(fit) or not len(held):
                raise ValueError("invalid OOF fit/heldout indices")
            if p == "A":
                assert_no_support_overlap(fit, held)
            else:
                expected = config["oof_specification"]["protocol_B"]["fold_membership_subjects"][f"fold_{f}"]
                if sorted(held.subject.unique().tolist()) != expected or set(fit.subject) & set(held.subject):
                    raise ValueError("B subject membership/isolation mismatch")
            audit.append({"protocol": p, "fold": f, "train_windows": len(fit), "heldout_windows": len(held),
                          "train_subjects": sorted(fit.subject.unique().tolist()), "heldout_subjects": sorted(held.subject.unique().tolist()),
                          "passed": True})
        records.append(part)
        audit.append({"protocol": p, "total_train": int(tr.sum()), "oof_covered": int((folds >= 0).sum()),
                      "excluded_boundary_windows": int((folds < 0).sum()), "oof_coverage_fraction": float((folds >= 0).mean()), "passed": True})
    plan = pd.concat(records).sort_values("row_id").reset_index(drop=True)
    save_frame(OUT / "window_fold_registry_trainval.parquet", plan[KEYS + ["split", "row_id", "history_start_time", "target_time", "cohort_core", "sequence_fold"]])
    save_json(OUT / "fold_isolation_preflight.json", audit)
    return plan


def core_rows(frame):
    return (frame.loc[frame.split.eq("train") & frame.cohort_core & frame.residual.notna()].copy(),
            frame.loc[frame.split.eq("validation") & frame.cohort_core].copy())


def fit_factors(train, validation, columns):
    if columns:
        path = ridge_path_predictions(train[columns], train.residual.to_numpy(), validation[columns])
        best, alpha, table = best_alpha(path, validation.base_prediction.to_numpy(), validation.target_value.to_numpy(), validation.subject.to_numpy())
        prediction = validation.base_prediction.to_numpy() + path[:, best]
    else:
        alpha = None
        prediction = validation.base_prediction.to_numpy().copy()
        table = subject_table(validation.subject, prediction - validation.target_value.to_numpy())
    return alpha, table, prediction


def key_digest(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame[KEYS + ["split", "row_id"]], index=False).to_numpy().tobytes()).hexdigest()


def task_paths(family, protocol, fold, seed):
    tag = f"{family}_{protocol}_{'full' if fold == -1 else 'fold' + str(fold)}_seed{seed}"
    directory = OUT / "shards" / family / protocol
    return tag, directory / f"{tag}_predictions_trainval.parquet", directory / f"{tag}_weights.pt", directory / f"{tag}.json"


def verify_shard(meta_path, config_hash, eval_frame, train_frame):
    """Return validated metadata and predictions from one shard read."""
    meta = read_json(meta_path)
    if meta["status"] != "complete" or meta["config_sha256"] != config_hash:
        raise ValueError("shard completion/config mismatch")
    for name, expected in meta["output_sha256"].items():
        if sha(OUT / name) != expected:
            raise ValueError("shard hash mismatch")
    if meta["fit_keys_sha256"] != key_digest(train_frame) or meta["prediction_keys_sha256"] != key_digest(eval_frame):
        raise ValueError("shard sample keys mismatch")
    pred = pd.read_parquet(OUT / meta["prediction_file"])
    if not np.array_equal(pred.row_id.to_numpy(), eval_frame.row_id.to_numpy()) or not np.isfinite(pred.prediction).all():
        raise ValueError("shard row alignment/nonfinite predictions")
    for col in KEYS + ["split"]:
        if not pred[col].reset_index(drop=True).equals(eval_frame[col].reset_index(drop=True)):
            raise ValueError("shard split/key mismatch")
    if pred.duplicated(KEYS).any():
        raise ValueError("duplicate shard predictions")
    return meta, pred


def train_task(family, protocol, fold, seed, frame, history, config, config_hash, progress):
    part = frame.loc[frame.evaluation_protocol.eq(seq.PROTOCOLS[protocol])]
    if fold >= 0:
        train = part.loc[part.split.eq("train") & part.sequence_fold.ge(0) & part.sequence_fold.ne(fold)]
        evaluation = part.loc[part.split.eq("train") & part.sequence_fold.eq(fold)]
    else:
        train = part.loc[part.split.eq("train")]
        evaluation = part.loc[part.split.eq("validation")]
    train, evaluation = train.reset_index(drop=True), evaluation.reset_index(drop=True)
    tag, pred_path, weights_path, meta_path = task_paths(family, protocol, fold, seed)
    if meta_path.exists():
        meta, _ = verify_shard(meta_path, config_hash, evaluation, train)
        print(f"REUSE {tag}", flush=True)
        return meta, True
    progress.update(current_task=tag, current_task_started_utc=now(), current_task_start_perf=time.perf_counter())
    save_json(OUT / "progress.json", progress)
    print(f"START {tag} epochs={config['epochs'][family][protocol]} rows={len(train)}", flush=True)
    for attempt in range(config["max_retries_per_task"] + 1):
        started = now()
        try:
            r, prediction, state = seq.train_one(family, train, evaluation,
                history[train.row_id.to_numpy()], history[evaluation.row_id.to_numpy()], seed=seed,
                device=config["devices"][family], epochs=config["epochs"][family][protocol], early_stopping=False, deadline=None)
            if r["epochs_run"] != config["epochs"][family][protocol] or r["early_stopping"] or not np.isfinite(prediction).all():
                raise ValueError("training did not follow training rule")
            pred = evaluation[KEYS + ["split", "row_id", "target_value"]].assign(
                prediction=prediction, model_family=family, protocol=protocol, seed=seed, fold=fold,
                prediction_role="OOF" if fold >= 0 else "validation_full_train", config_sha256=config_hash)
            save_frame(pred_path, pred)
            atomic(weights_path, lambda p: torch.save({"state_dict": state, "statistics": r["statistics"], "config_sha256": config_hash}, p))
            meta = {"status": "complete", "tag": tag, "family": family, "protocol": protocol, "fold": fold, "seed": seed,
                    "attempt": attempt + 1, "started_utc": started, "finished_utc": now(), "config_sha256": config_hash,
                    "fit_keys_sha256": key_digest(train), "prediction_keys_sha256": key_digest(evaluation),
                    "fit_subjects": sorted(train.subject.unique().tolist()), "prediction_subjects": sorted(evaluation.subject.unique().tolist()),
                    "fold_registry": "window_fold_registry_trainval.parquet", "fold_registry_sha256": sha(OUT / "window_fold_registry_trainval.parquet"),
                    "prediction_file": str(pred_path.relative_to(OUT)), "record": r,
                    "output_sha256": {str(pred_path.relative_to(OUT)): sha(pred_path), str(weights_path.relative_to(OUT)): sha(weights_path)}}
            save_json(meta_path, meta)  # completion marker is committed last
            print(f"DONE {tag} seconds={r['total_seconds']:.2f}", flush=True)
            return meta, False
        except (RuntimeError, MemoryError) as error:
            failure = {"tag": tag, "attempt": attempt + 1, "started_utc": started, "finished_utc": now(), "error": str(error), "traceback": traceback.format_exc()}
            save_json(OUT / "failures" / f"{tag}_attempt{attempt+1}.json", failure)
            recoverable = isinstance(error, MemoryError) or any(x in str(error).lower() for x in ["out of memory", "alloc", "cuda error"])
            if not recoverable or attempt >= config["max_retries_per_task"]:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"RETRY {tag}: identical seed/config after memory cleanup", flush=True)


def combine_protocol(family, protocol, frame, config, config_hash):
    part = frame.loc[frame.evaluation_protocol.eq(seq.PROTOCOLS[protocol])]
    expected = part.loc[(part.split.eq("train") & part.sequence_fold.ge(0)) | part.split.eq("validation")].sort_values("row_id")
    seed_frames, metadata = [], []
    for seed in config["seeds"]:
        pieces = []
        for fold in list(range(5)) + [-1]:
            if fold >= 0:
                fit = part.loc[part.split.eq("train") & part.sequence_fold.ge(0) & part.sequence_fold.ne(fold)]
                ev = part.loc[part.split.eq("train") & part.sequence_fold.eq(fold)]
            else:
                fit, ev = part.loc[part.split.eq("train")], part.loc[part.split.eq("validation")]
            _, pred_path, _, meta_path = task_paths(family, protocol, fold, seed)
            meta, predictions = verify_shard(meta_path, config_hash, ev.reset_index(drop=True), fit.reset_index(drop=True))
            metadata.append(meta)
            pieces.append(predictions)
        combined = pd.concat(pieces).sort_values("row_id").reset_index(drop=True)
        if not np.array_equal(combined.row_id.to_numpy(), expected.row_id.to_numpy()) or combined.duplicated(KEYS).any():
            raise ValueError("OOF/validation coverage is not exactly once per seed")
        if set(combined.seed) != {seed} or set(combined.config_sha256) != {config_hash}:
            raise ValueError("seed/config provenance mismatch")
        seed_frames.append(combined)
    seeds = pd.concat(seed_frames, ignore_index=True)
    ensemble = seed_frames[0][KEYS + ["split", "row_id", "target_value", "fold", "prediction_role"]].copy()
    ensemble["prediction"] = np.mean(np.stack([s.prediction.to_numpy() for s in seed_frames]), axis=0)
    ensemble["residual"] = ensemble.target_value - ensemble.prediction
    ensemble["model_family"], ensemble["protocol"], ensemble["config_sha256"] = family, protocol, config_hash
    save_frame(OUT / f"{family}_{protocol}_seeds_predictions_trainval.parquet", seeds)
    save_frame(OUT / f"{family}_{protocol}_ensemble_predictions_trainval.parquet", ensemble)
    checks = {"family": family, "protocol": protocol, "passed": True, "fits": len(metadata), "seeds": config["seeds"],
            "oof_windows_per_seed": int(expected.split.eq("train").sum()), "validation_windows_per_seed": int(expected.split.eq("validation").sum()),
            "train_total": int(part.split.eq("train").sum()), "excluded_boundary_windows": int((part.split.eq("train") & part.sequence_fold.lt(0)).sum()),
            "all_preprocessing_estimated_from_exact_fit_rows": True, "no_early_stopping": True,
            "output_sha256": {f"{family}_{protocol}_ensemble_predictions_trainval.parquet": sha(OUT / f"{family}_{protocol}_ensemble_predictions_trainval.parquet")}}
    save_json(OUT / f"{family}_{protocol}_ensemble_checks.json", checks)
    return ensemble, seeds


def evaluate_protocol(family, protocol, frame, ensemble, seeds, config):
    part = frame.loc[frame.evaluation_protocol.eq(seq.PROTOCOLS[protocol])].copy()
    joined = part.drop(columns=["residual"]).merge(ensemble[["row_id", "prediction", "residual"]], on="row_id", how="left", sort=False, validate="one_to_one")
    joined["lr_base_prediction"] = joined.base_prediction
    joined["base_prediction"] = joined.prediction
    train, validation = core_rows(joined)
    counts = validation.groupby("subject").size()
    eligible = counts.index[counts >= config["min_validation_windows_per_subject"]]
    keep = validation.subject.isin(eligible)
    report_valid = validation.loc[keep].reset_index(drop=True)
    lr_table = subject_table(report_valid.subject, report_valid.lr_base_prediction.to_numpy() - report_valid.target_value.to_numpy())
    _, base_all, base_prediction = fit_factors(train, validation, [])
    base = base_all.loc[eligible]
    fitted = {"LR": lr_table, "sequence": base}
    metrics, contrasts, predictions, seed_metrics, alpha_rows = [], [], [], [], []
    keyhash = key_digest(report_valid)
    for block in ["M", "T", "O"]:
        columns = config["factor_columns"][protocol][block]
        alpha, table_all, prediction = fit_factors(train, validation, columns)
        table = table_all.loc[eligible]
        fitted["sequence+" + block] = table
        contrasts.append({"family": family, "protocol": protocol, "comparison": f"sequence -> sequence+{block}", "factor_block": block,
                          "role": "primary exploratory transfer", "alpha": alpha, "training_windows": len(train), "sample_keys_sha256": keyhash,
                          **contrast_row(base, table, min_windows=20, n_resamples=2000, seed=42)})
        predictions.append(report_valid[KEYS + ["split", "row_id", "target_value"]].assign(model="sequence+" + block, prediction=prediction[keep.to_numpy()], family=family))
        alpha_rows.append({"family": family, "protocol": protocol, "block": block, "alpha": alpha, "columns": columns,
                           "train_windows": len(train), "alpha_selection_validation_windows": len(validation), "reported_validation_windows": len(report_valid),
                           "train_sample_keys_sha256": key_digest(train), "validation_sample_keys_sha256": key_digest(validation)})
    contrasts.append({"family": family, "protocol": protocol, "comparison": "LR -> sequence", "factor_block": "none", "role": "baseline comparison",
                      "sample_keys_sha256": keyhash, **contrast_row(lr_table, base, min_windows=20, n_resamples=2000, seed=42)})
    subjects = []
    for name, table in fitted.items():
        if not table.index.equals(base.index) or not table.n_windows.equals(base.n_windows):
            raise ValueError("models do not share exactly the same validation participants/windows")
        metrics.append({"family": family, "protocol": protocol, "model": name, "subjects": len(table), "windows": int(table.n_windows.sum()),
                        "subject_equal_rmse": float(table.rmse.mean()), "subject_equal_mae": float(table.mae.mean()),
                        "pooled_rmse": float(np.sqrt(table.sum_squared_error.sum() / table.n_windows.sum())), "sample_keys_sha256": keyhash})
        subjects.append(table.reset_index().assign(family=family, protocol=protocol, model=name))
    predictions.append(report_valid[KEYS + ["split", "row_id", "target_value"]].assign(model="sequence", prediction=report_valid.base_prediction, family=family))
    predictions.append(report_valid[KEYS + ["split", "row_id", "target_value"]].assign(model="LR", prediction=report_valid.lr_base_prediction, family=family))
    for seed in config["seeds"]:
        sv = seeds.loc[seeds.seed.eq(seed) & seeds.split.eq("validation"), ["row_id", "prediction"]]
        sv = report_valid.drop(columns="prediction").merge(sv, on="row_id", sort=False, validate="one_to_one")
        table = subject_table(sv.subject, sv.prediction.to_numpy() - sv.target_value.to_numpy())
        seed_metrics.append({"family": family, "protocol": protocol, "seed": seed, "subjects": len(table), "windows": len(sv),
                             "subject_equal_rmse": float(table.rmse.mean()), "subject_equal_mae": float(table.mae.mean()), "sample_keys_sha256": keyhash})
    result = {"metrics": metrics, "contrasts": contrasts, "seeds": seed_metrics, "alpha": alpha_rows}
    save_json(OUT / f"{family}_{protocol}_transfer_results.json", result)
    save_frame(OUT / f"{family}_{protocol}_reported_predictions_validation.parquet", pd.concat(predictions, ignore_index=True))
    pd.concat(subjects).to_csv(OUT / f"{family}_{protocol}_subject_metrics_validation.csv", index=False)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = read_json(OUT / "experiment_config.json")
    config_hash = sha(OUT / "experiment_config.json")
    if sha(ROOT / "configs/seq_factor_transfer.json") != config_hash:
        raise ValueError("active configuration differs from the checkpoint configuration")
    if torch.get_num_threads() != 14:
        torch.set_num_threads(14)
    code_paths = [ROOT / "src/seq_models.py", Path(__file__), ROOT / "src/residuals.py", ROOT / "src/evaluation.py"]
    versions = {str(p.relative_to(ROOT)): sha(p) for p in code_paths}
    if (OUT / "source_versions.json").exists():
        if read_json(OUT / "source_versions.json") != versions:
            raise ValueError("code changed since run freeze; inspect before resume")
    else:
        save_json(OUT / "source_versions.json", versions)
    inputs = [INPUTS / "window_keys_targets_trainval.parquet", INPUTS / "glucose_history_trainval.npy",
              ROOT / "results/laboratory_interactions/reference_trainval.parquet",
              ROOT / "results/data_quality/split_scope_v1.csv",
              ROOT / "results/forecast_windows/fold_assignments.parquet"]
    inputs += [ROOT / "results/factor_discovery" / name for name in [
        "factor_protocol.json", "evaluation_plan.json", "selection_T.csv", "selection_O.csv",
        "e2_O_validation.csv", "candidates_batch1.parquet", "candidates_batch2.parquet",
        "openfe_values.parquet", "primitives_A.parquet", "primitives_B.parquet"]]
    input_versions = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    if (OUT / "input_versions.json").exists():
        if read_json(OUT / "input_versions.json") != input_versions:
            raise ValueError("input hashes changed since freeze")
    else:
        save_json(OUT / "input_versions.json", input_versions)
    environment = {"recorded_utc": now(), "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "cpu_threads": torch.get_num_threads(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__, "pandas": pd.__version__}
    save_json(OUT / "environment.json", environment)
    state_path = OUT / "run_state.json"
    status = read_json(state_path) if state_path.exists() else {"started_utc": now(), "completed_tasks": 0, "reused_tasks": 0, "protocols": [], "failures": []}
    status.update(status="preflight", config_sha256=config_hash, cpu_threads=torch.get_num_threads(), torch=torch.__version__, python=platform.python_version(),
                  devices=config["devices"], early_stopping=False, analysis="fixed-epoch sequence factor transfer")
    progress = {"completed_tasks": 0, "total_tasks": 120, "current_task": "preflight"}
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(30):
            with LOCK:
                progress["heartbeat_utc"] = now()
                progress["current_task_elapsed_seconds"] = round(time.perf_counter() - progress.get("current_task_start_perf", time.perf_counter()), 1)
                save_json(OUT / "progress.json", progress)
                print("HEARTBEAT " + json.dumps(progress), flush=True)
    worker = threading.Thread(target=heartbeat, daemon=True)
    started = time.perf_counter()
    results = []
    try:
        frame, history = load_inputs(config)
        frame = make_fold_plan(frame, config)
        status["preflight_passed"] = True
        save_json(state_path, status)
        worker.start()
        for family, protocol in config["run_order"]:
            print(f"PROTOCOL START {family} {protocol}", flush=True)
            group_start = time.perf_counter()
            try:
                for fold in list(range(5)) + [-1]:
                    for seed in config["seeds"]:
                        meta, reused = train_task(family, protocol, fold, seed, frame, history, config, config_hash, progress)
                        progress["completed_tasks"] += 1
                        status["completed_tasks"] = progress["completed_tasks"]
                        status["reused_tasks"] += int(reused)
                        status["status"] = "training"
                        save_json(state_path, status)
                        if fold == 0 and seed == 0:
                            save_json(OUT / f"{family}_{protocol}_initial_fit_checks.json", {"passed": True, "task": meta["tag"], "counted_in_120_fits": True,
                                      "preprocessing_fit_rows_only": True, "finite_predictions": True, "config_sha256": config_hash})
                ensemble, seeds = combine_protocol(family, protocol, frame, config, config_hash)
                print(f"Ensemble completed {family} {protocol}", flush=True)
                result = evaluate_protocol(family, protocol, frame, ensemble, seeds, config)
                results.append(result)
                status["protocols"] = [r for r in status["protocols"] if (r["family"], r["protocol"]) != (family, protocol)]
                status["protocols"].append({"family": family, "protocol": protocol, "status": "sequence_transfer_complete", "seconds_this_invocation": time.perf_counter() - group_start})
                save_json(state_path, status)
                print(f"Factor transfer completed {family} {protocol}", flush=True)
            except (RuntimeError, MemoryError) as error:
                status["failures"].append({"family": family, "protocol": protocol, "error": str(error), "traceback": traceback.format_exc()})
                save_json(state_path, status)
                print(f"NETWORK/PROTOCOL STOP {family} {protocol}: {error}", flush=True)
        for kind, filename in [("metrics", "metrics_validation.csv"), ("contrasts", "factor_gains_validation.csv"), ("seeds", "single_seed_metrics_validation.csv")]:
            pd.DataFrame([r for result in results for r in result[kind]]).to_csv(OUT / filename, index=False)
        status["status"] = "sequence_transfer_complete" if len(results) == 4 else "partial_results"
    except Exception as error:
        status["status"] = "stopped_invalid_method_or_input"
        status["failures"].append({"error": str(error), "traceback": traceback.format_exc()})
        print(traceback.format_exc(), flush=True)
    finally:
        stop.set()
        if worker.is_alive():
            worker.join(timeout=2)
        status.update(finished_utc=now(), wall_seconds_this_invocation=time.perf_counter() - started,
                      config_sha256=config_hash)
        save_json(state_path, status)
        print(json.dumps(status), flush=True)
    if status["status"] != "sequence_transfer_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
