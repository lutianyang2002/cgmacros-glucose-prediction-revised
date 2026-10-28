"""Laboratory-state interaction runner for TRAIN/VALIDATION data.

Run with ``python -B -m src.lab_experiment --task all --output-dir reproductions/lab_run``. Core and selection tasks have separate hard timeouts. Existing completed fits
are reused only when config/code and exact train/validation inputs match.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.evaluation import every_fifteen_minutes, subject_ic
from src.factor_evaluation import strata_masks
from src.lab_interactions import Loader, KEY, FOLD_KEY, STATES, assert_trainval, check_bmi, lab_family_catalog, lab_interaction_columns, lab_state_parameters, sha
from src.residuals import RIDGE_ALPHAS, best_alpha, contrast_row, ridge_path_predictions, subject_table
from src.selection import one_standard_error_choice, redundancy_filter

ROOT = Path(__file__).resolve().parent.parent
OUT = None  # Set by the CLI to an isolated calculation directory.
CONFIG = ROOT / "configs/lab_interaction_sensitivity.json"
EVIDENCE = "exploratory, validation-stage, offline sensitivity analysis"


def write_json(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def csv(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, lineterminator="\n")


def record(**updates):
    p = OUT / "experiment_summary.json"
    current = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    current.update(updates)
    write_json("experiment_summary.json", current)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def kernel_map(config, tag):
    return {r["meal_class"]: r["kernel_column"] for r in config["kernel_source"]["parameters"] if r["protocol"] == tag}


def code_hash():
    modules = ["lab_interactions", "lab_experiment", "residuals", "selection", "evaluation", "factor_construction", "factor_evaluation", "baselines"]
    return {f"src/{m}.py": sha(ROOT / f"src/{m}.py") for m in modules}


def digest_frame(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


class Fitter:
    def __init__(self, train, valid, context, config, deadline):
        assert_trainval(train)
        assert_trainval(valid)
        if not train.split.eq("train").all() or not valid.split.eq("validation").all():
            raise ValueError("Fitting/evaluation partition violation")
        self.train, self.valid, self.context = train, valid, context
        self.config, self.deadline = config, deadline
        self.cache = {}
        self.computed, self.reused = 0, 0
        self.root = OUT / "model_checkpoints"
        self.root.mkdir(exist_ok=True)
        self.signature = {"config_sha256": sha(CONFIG), "code_sha256": code_hash(), "train_fingerprint": digest_frame(train), "validation_fingerprint": digest_frame(valid)}

    def check_budget(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("Worker time budget exceeded")

    def fit(self, columns):
        self.check_budget()
        columns = list(columns)
        if len(columns) != len(set(columns)):
            raise ValueError("Duplicate model column")
        key = tuple(columns)
        if key in self.cache:
            return self.cache[key]
        cid = hashlib.sha256(json.dumps([self.context, columns], sort_keys=True).encode()).hexdigest()[:20]
        meta_path, errors_path = self.root / f"{cid}.json", self.root / f"{cid}.npy"
        if meta_path.exists() or errors_path.exists():
            if not meta_path.exists() or not errors_path.exists():
                raise RuntimeError("Incomplete model checkpoint; no silent overwrite")
            meta = json.loads(meta_path.read_text())
            if meta["signature"] != self.signature or meta["columns"] != columns:
                raise RuntimeError("Checkpoint provenance mismatch")
            error = np.load(errors_path, allow_pickle=False)
            alpha = meta["alpha"]
            self.reused += 1
        else:
            t0 = time.perf_counter()
            prediction = ridge_path_predictions(self.train[columns], self.train.residual.to_numpy(), self.valid[columns])
            j, alpha, _ = best_alpha(prediction, self.valid.base_prediction.to_numpy(), self.valid.target_value.to_numpy(), self.valid.subject.to_numpy())
            error = self.valid.base_prediction.to_numpy() + prediction[:, j] - self.valid.target_value.to_numpy()
            meta = {"context": self.context, "columns": columns, "alpha": alpha, "training_windows": len(self.train), "validation_windows": len(self.valid),
                    "signature": self.signature, "seconds": time.perf_counter() - t0, "stage": EVIDENCE, "split": "validation", "fit_status": "COMPLETED"}
            if error.shape != (len(self.valid),) or not np.isfinite(error).all():
                raise ValueError("Invalid prediction error; checkpoint was not written")
            np.save(errors_path, error, allow_pickle=False)
            meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            self.computed += 1
        if error.shape != (len(self.valid),) or not np.isfinite(error).all():
            raise ValueError("Invalid checkpoint prediction error")
        result = {"alpha": float(alpha), "error": error, "subject_table": subject_table(self.valid.subject.to_numpy(), error), "checkpoint": cid}
        self.cache[key] = result
        self.check_budget()
        return result


def proxy_frames(train, valid, use_proxy):
    assert_trainval(train)
    assert_trainval(valid)
    if not use_proxy:
        return train.copy(), valid.copy()
    proxy = train.groupby("subject").residual.mean()
    tr, va = train.assign(identity_proxy=train.subject.map(proxy)), valid.assign(identity_proxy=valid.subject.map(proxy))
    if tr.identity_proxy.isna().any() or va.identity_proxy.isna().any():
        raise ValueError("Identity proxy requested for an unseen subject")
    return tr, va


def training_and_validation(table, minimum=20, complete=True):
    assert_trainval(table)
    keep = table.cohort_core
    if complete:
        keep = keep & np.isfinite(table[list(STATES.values())]).all(axis=1)
    part = table.loc[keep].copy()
    train = part.loc[part.split.eq("train") & part.residual.notna()].copy()
    valid = part.loc[part.split.eq("validation")].copy()
    if minimum is not None:
        valid = valid.loc[valid.groupby("subject").subject.transform("size") >= minimum].copy()
    return train.reset_index(drop=True), valid.reset_index(drop=True)


def paired(a, b, config, minimum=20):
    if not len(a) or not len(b):
        raise ValueError("No subjects in contrast")
    return contrast_row(a, b, minimum, config["bootstrap"]["resamples"], config["bootstrap"]["seed"])




def construct(table, config):
    all_values, cats, parameters, checks, coverage = [], [], {}, [], []
    for tag, name in config["protocols"].items():
        part = table[table.evaluation_protocol.eq(name)].reset_index(drop=True)
        km = kernel_map(config, tag)
        bmi = check_bmi(part, km, config["bmi_reconstruction_check"]["value_abs_tolerance"])
        checks.extend([{ "protocol": tag, **r} for r in bmi])
        eligible = np.isfinite(part[list(STATES.values())]).all(axis=1)
        complete = part.loc[eligible].reset_index(drop=True)
        p = lab_state_parameters(complete)
        parameters[tag] = p
        for split in ("train", "validation"):
            original = part[part.split.eq(split)]
            retained = complete[complete.split.eq(split)]
            core = retained[retained.cohort_core & (retained.residual.notna() if split == "train" else True)]
            evaluable = core if split == "train" else core[core.groupby("subject").subject.transform("size") >= 20]
            coverage.append({"protocol": tag, "split": split, "subjects_before": original.subject.nunique(), "windows_before": len(original),
                             "complete_subjects": retained.subject.nunique(), "complete_windows": len(retained), "excluded_lab_subjects": original.subject.nunique()-retained.subject.nunique(),
                             "core_oof_train_or_validation_windows": len(core), "analysis_subjects": evaluable.subject.nunique(), "analysis_windows": len(evaluable),
                             "subject_coverage": retained.subject.nunique()/original.subject.nunique(), "window_coverage": len(retained)/len(original)})
        for scheme in ("C-S1", "C-S2"):
            x = lab_interaction_columns(complete, km, p, scheme)
            train = complete.split.eq("train") & complete.cohort_core & complete.residual.notna()
            bad = [c for c in x if not np.isfinite(x[c]).all() or x.loc[train, c].nunique() < 2]
            if bad:
                raise ValueError("Invalid/constant interaction candidate: " + repr(bad))
            vals = complete[FOLD_KEY + ["split", "cohort_core"]].copy()
            vals["standardization"] = scheme
            for meal, col in km.items():
                vals["K_" + meal] = complete[col]
            for state, col in STATES.items():
                vals["L_" + state] = complete[col]
            all_values.append(pd.concat([vals, x], axis=1))
            cat = lab_family_catalog(km, scheme)
            cat["evaluation_protocol"] = name
            cats.append(cat)
    csv("bmi_reproduction.csv", checks)
    csv("coverage_analysis.csv", coverage)
    write_json("state_standardization.json", parameters)
    pd.concat(cats, ignore_index=True).to_csv(OUT / "candidate_catalog_C.csv", index=False)
    values = pd.concat(all_values, ignore_index=True)
    assert_trainval(values)
    if values.duplicated(FOLD_KEY + ["split", "standardization"]).any():
        raise ValueError("Candidate key duplication")
    values.to_parquet(OUT / "candidate_values_trainval.parquet", index=False)
    return values, checks


def prepare_context(table, values, tag, scheme, proxy, config):
    name = config["protocols"][tag]
    part = table[table.evaluation_protocol.eq(name)].reset_index(drop=True)
    candidates = values[values.evaluation_protocol.eq(name) & values.standardization.eq(scheme)]
    xcols = config["candidate_ids"]
    # Fixed rows first, then attach only corresponding constructed candidates.
    part = part.merge(candidates[KEY + xcols], on=KEY, how="inner", validate="one_to_one")
    tr, va = training_and_validation(part)
    return proxy_frames(tr, va, proxy)


def evaluate_core(table, values, config, deadline):
    metrics, comparisons, subjects, predictions, fits_record = [], [], [], [], []
    for tag in config["protocols"]:
        for proxy in ([False, True] if tag == "A" else [False]):
            for scheme in ("C-S1", "C-S2"):
                train, valid = prepare_context(table, values, tag, scheme, proxy, config)
                fitter = Fitter(train, valid, f"C3_{tag}_{scheme}_proxy{int(proxy)}", config, deadline)
                backbone = config["reference"]["manual_columns"] + (["identity_proxy"] if proxy else [])
                groups = {"K": list(kernel_map(config, tag).values()), "L": list(STATES.values()), "X": config["candidate_ids"]}
                fits = {}
                context = {"protocol": tag, "standardization": scheme, "identity_proxy": proxy, "split": "validation", "stage": EVIDENCE}
                common = valid[KEY].copy()
                common["base_prediction"] = valid.base_prediction
                common["target_value"] = valid.target_value
                for model, blocks in config["model_blocks"].items():
                    cols = backbone + [c for b in blocks for c in groups[b]]
                    fit = fitter.fit(cols)
                    fits[model] = fit
                    t = fit["subject_table"]
                    metrics.append({**context, "model": model, "alpha": fit["alpha"], "training_subjects": train.subject.nunique(), "training_windows": len(train),
                                    "subjects": len(t), "validation_windows": len(valid), "subject_equal_rmse": t.rmse.mean(), "subject_equal_mae": t.mae.mean(),
                                    "pooled_rmse": np.sqrt(t.sum_squared_error.sum()/t.n_windows.sum()), "pooled_mae": t.sum_abs_error.sum()/t.n_windows.sum(), "columns": len(cols)})
                    subjects.append(t.reset_index().assign(**context, model=model))
                    pred = common.copy()
                    pred["prediction"] = fit["error"] + valid.target_value.to_numpy()
                    predictions.append(pred.assign(**context, model=model))
                masks = {"all": np.ones(len(valid), bool), **strata_masks(valid)}
                for contrast in config["contrasts"]:
                    for stratum, mask in masks.items():
                        if stratum != "all" and contrast["role"] != "primary":
                            continue
                        a, b = [subject_table(valid.subject.to_numpy()[mask], fits[contrast[s]]["error"][mask]) for s in ("from", "to")]
                        for minimum in (20, None):
                            eligible = (a.n_windows >= minimum) if minimum else np.ones(len(a), bool)
                            if not np.asarray(eligible).any():
                                comparisons.append({**context, **contrast, "stratum": stratum, "min_window_rule": "applied >=20" if minimum else "not applied", "status": "NOT_EVALUABLE"})
                                continue
                            stats = paired(a, b, config, minimum)
                            comparisons.append({**context, **contrast, "stratum": stratum, "min_window_rule": "applied >=20" if minimum else "not applied",
                                                "training_windows": len(train), "alpha_from": fits[contrast["from"]]["alpha"], "alpha_to": fits[contrast["to"]]["alpha"], **stats, "status": "COMPLETED"})
                            fitter.check_budget()
                fits_record.append({**context, "new_fits": fitter.computed, "checkpoint_reuses": fitter.reused})
                print(f"Laboratory interactions {tag} {scheme} identity={proxy}: completed {len(fits)} models", flush=True)
    csv("metrics_validation.csv", metrics)
    csv("results_validation.csv", comparisons)
    pd.concat(subjects, ignore_index=True).to_csv(OUT / "subject_metrics_validation.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(OUT / "predictions_validation.parquet", index=False)
    write_json("core_fit_record.json", fits_record)
    return fits_record


def core(config):
    start = time.perf_counter()
    deadline = start + config["budget_seconds"]["core"]
    loader = Loader(ROOT)
    record(status="FITTING", statistical_isolation_policy=config["data_access_policy"], started_utc=stamp(), code_sha256=code_hash(), config_sha256=sha(CONFIG))
    table = loader.build(config)
    timing = {"data_loading_seconds": time.perf_counter()-start}
    table.to_parquet(OUT / "reference_trainval.parquet", index=False)
    t = time.perf_counter()
    values, _ = construct(table, config)
    timing["candidate_generation_seconds"] = time.perf_counter()-t
    t = time.perf_counter()
    fit_record = evaluate_core(table, values, config, deadline)
    timing["evaluation_seconds"] = time.perf_counter()-t
    timing["core_total_seconds"] = time.perf_counter()-start
    write_json("timing_core.json", timing)
    record(status="FITTED", timing=timing, finished_utc=stamp(), model_fits=sum(r["new_fits"] for r in fit_record))


def selection(config):
    start = time.perf_counter()
    deadline = start + config["budget_seconds"]["auxiliary_selection"]
    table = pd.read_parquet(OUT / "reference_trainval.parquet")
    values = pd.read_parquet(OUT / "candidate_values_trainval.parquet")
    assert_trainval(table)
    assert_trainval(values)
    loader = Loader(ROOT)
    summaries, selections, ic_records, filters, curves, contrast_records = [], [], [], [], [], []
    counts = {"new_fits": 0, "reused_checkpoints": 0}
    for tag, name in config["protocols"].items():
        # Only TRAIN lags needed for the fixed-base redundancy check.
        part = table[table.evaluation_protocol.eq(name)].reset_index(drop=True)
        tr0 = part[part.split.eq("train")].reset_index(drop=True)
        tr_lags = loader.add_selection_lags(tr0, config["selection_lag_columns"])
        for proxy in ([False, True] if tag == "A" else [False]):
            train, valid = prepare_context(table, values, tag, "C-S1", proxy, config)
            train = train.merge(tr_lags[KEY + config["selection_lag_columns"]], on=KEY, validate="one_to_one")
            first = part[part.split.eq("train")].groupby("subject").origin_time.min()
            thin = train.loc[every_fifteen_minutes(train, first)]
            ids = config["candidate_ids"]
            ic, sizes = subject_ic(thin, ids)
            ranking = pd.DataFrame({"candidate_id": ids, "mean_ic": ic.mean().reindex(ids).values, "subjects_with_ic": ic.notna().sum().reindex(ids).values})
            ranking["abs_mean_ic"] = ranking.mean_ic.abs()
            ranking = ranking.sort_values("abs_mean_ic", ascending=False, kind="stable")
            valid_rank = ranking[ranking.abs_mean_ic.notna()]
            pool_top = valid_rank.head(config["auxiliary_selection"]["keep_per_family"]).candidate_id.tolist()
            ranking["kept_top_k"] = ranking.candidate_id.isin(pool_top)
            summaries.append(ranking.assign(protocol=tag, identity_proxy=proxy, standardization="C-S1"))
            ic_records.append(ic.reset_index().melt(id_vars="subject", var_name="candidate_id", value_name="train_IC").assign(protocol=tag, identity_proxy=proxy))
            filt = redundancy_filter(pool_top, thin[pool_top], thin[config["selection_lag_columns"]], 0.9, "spearman")
            pool = filt.loc[filt.decision.eq("kept"), "candidate_id"].tolist() if len(filt) else []
            filters.append(filt.assign(protocol=tag, identity_proxy=proxy, standardization="C-S1"))
            backbone = config["reference"]["manual_columns"] + list(kernel_map(config, tag).values()) + list(STATES.values()) + (["identity_proxy"] if proxy else [])
            fitter = Fitter(train, valid, f"selection_{tag}_proxy{int(proxy)}", config, deadline)
            # Existing forward_selection starts k=0 at the bare glucose base.
            # Here k=0 must retain the fixed R+K+L backbone at every step.
            baseline = fitter.fit(backbone)
            tables = {0: baseline["subject_table"]}
            curve = [{"k": 0, "added": "", "alpha": baseline["alpha"], "subject_equal_rmse": float(tables[0].rmse.mean())}]
            chosen, remaining = [], list(pool)
            while remaining and len(chosen) < config["auxiliary_selection"]["k_max"]:
                trials = [(float((f := fitter.fit(backbone + chosen + [cid]))["subject_table"].rmse.mean()), pool.index(cid), cid, f) for cid in remaining]
                rmse, _, cid, fit = min(trials, key=lambda t: (t[0], t[1]))
                chosen.append(cid)
                remaining.remove(cid)
                tables[len(chosen)] = fit["subject_table"]
                curve.append({"k": len(chosen), "added": cid, "alpha": fit["alpha"], "subject_equal_rmse": rmse})
            curve = pd.DataFrame(curve)
            rule = one_standard_error_choice(curve, tables)
            selected = chosen[:rule["k_paired"]]
            context = {"protocol": tag, "identity_proxy": proxy, "standardization": "C-S1", "split": "validation", "stage": EVIDENCE}
            curves.append(curve.assign(**context))
            selections.append({**context, "valid_candidates": len(valid_rank), "top_IC_candidates": json.dumps(pool_top), "post_redundancy_candidates": json.dumps(pool),
                               "pool_size": len(pool), "k_best": rule["best_k"], "k_selected": rule["k_paired"], "k_unpaired_sensitivity": rule["k_unpaired"],
                               "selected_set": json.dumps(selected), "selected_expressions": json.dumps([f"{kernel_map(config,tag)[s[4:].split('_carbs_x_')[0]]} * z_train_subject({s.split('_carbs_x_')[1]})" for s in selected]),
                               "training_windows": len(train), "validation_subjects": valid.subject.nunique(), "validation_windows": len(valid),
                               "baseline_RKL_rmse": float(tables[0].rmse.mean()), "selected_rmse": float(tables[len(selected)].rmse.mean()), "same_validation_selection_and_evaluation": True})
            contrast_records.append({**context, "from": "R+K+L", "to": "R+K+L+selected_X", **paired(tables[0], tables[len(selected)], config), "selected_count": len(selected)})
            counts["new_fits"] += fitter.computed
            counts["reused_checkpoints"] += fitter.reused
            print(f"T-style {tag} identity={proxy}: {len(valid_rank)} valid, {len(pool)} after filtering, selected {len(selected)}", flush=True)
    csv("selection_C.csv", selections)
    pd.concat(summaries, ignore_index=True).to_csv(OUT / "screening_C.csv", index=False)
    pd.concat(ic_records, ignore_index=True).to_csv(OUT / "screening_subject_IC_C.csv", index=False)
    pd.concat(filters, ignore_index=True).to_csv(OUT / "redundancy_C.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "selection_curves_C.csv", index=False)
    csv("selection_comparisons_validation.csv", contrast_records)
    write_json("timing_selection.json", {"seconds": time.perf_counter()-start, **counts, "budget_seconds": config["budget_seconds"]["auxiliary_selection"]})
    record(status="COMPLETED", selection_finished_utc=stamp(), selection_model_fits=counts["new_fits"], selection_seconds=time.perf_counter()-start)


def main(argv=None):
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["all", "core", "selection"], default="all")
    ap.add_argument("--output-dir", type=Path, required=True, help="isolated run directory under reproductions/")
    args = ap.parse_args(argv)
    OUT = args.output_dir.resolve()
    if not OUT.is_relative_to((ROOT / "reproductions").resolve()):
        ap.error("output must be under reproductions/")
    if args.task != "selection" and OUT.exists():
        ap.error("core calculation requires a new output directory")
    if args.task == "selection" and (not (OUT / "experiment_summary.json").is_file() or any((OUT / name).exists() for name in ("selection_C.csv", "screening_C.csv", "selection_comparisons_validation.csv"))):
        ap.error("selection requires core outputs and no previous selection outputs")
    config = json.loads(CONFIG.read_text())
    if not config["execution_enabled"] or config["data_access_policy"]["test_statistical_use"]:
        raise ValueError("Configuration does not permit this run")
    if not np.array_equal(np.array(config["alpha_grid"]), RIDGE_ALPHAS):
        raise ValueError("Configured alpha grid differs from the model grid")
    try:
        if args.task == "all":
            for task, limit in [("core", config["budget_seconds"]["core"]), ("selection", config["budget_seconds"]["auxiliary_selection"])]:
                subprocess.run([sys.executable, "-B", "-m", "src.lab_experiment", "--task", task, "--output-dir", str(OUT)], cwd=ROOT, check=True, timeout=limit, env=os.environ.copy())
        elif args.task == "core":
            OUT.mkdir(parents=True)
            core(config)
        else:
            r = json.loads((OUT / "experiment_summary.json").read_text())
            if r.get("status") != "FITTED":
                raise RuntimeError("Selection requires completed core results")
            selection(config)
    except Exception as exc:
        write_json("failure_record.json", {"task": args.task, "exception_type": type(exc).__name__, "message": str(exc), "UTC": stamp()})
        record(status="STOPPED_ERROR", last_failure={"task": args.task, "type": type(exc).__name__, "message": str(exc)})
        raise


if __name__ == "__main__":
    main()
