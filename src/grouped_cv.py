"""Development-data analysis: 36-subject grouped cross-validation with the whole pipeline repeated inside every fold.

The 36 subjects are the 27 training and 9 validation subjects of Protocol B (the 9 test subjects are never loaded). For every outer fold the pipeline of theory-factor discovery is re-run on the other
subjects alone and evaluated once on the outer subjects:

  1. the GlucoBench linear regression (24 lags, unchanged configuration) is refitted on all windows of the training subjects;
  2. training residuals by the Protocol B OOF design: out-of-fold predictions of that base over 5 deterministic GroupKFold folds of the training subjects;
  3. screening on the thinned core training windows: subject-level Spearman IC of the 187 first-batch candidates, tau*/k* of every (nutrient, meal class) re-selected from these IC, the
     second-batch candidates rebuilt on the new kernels, IC of all main candidates, the five best |mean IC| per family, the Spearman redundancy filter (|rho| > 0.9);
  4. inner forward selection (k_max = 10) and the penalty from the configured grid, scored by 3-fold subject-grouped cross-validation *inside the training subjects*; the baseline of every inner
     split is refitted on that split's training part, as in the LLM/Random-DSL search fitness (`llm_fitness.group_splits`); the number of factors by the paired one-standard-error rule;
  5. ridge blocks base + M, base + T and base + M + T (M: the 50 existing features, a fixed set; T: the factors selected in step 4) are fitted on the training subjects' residuals with
     the penalty from step 4 and evaluated on the outer subjects, once.

The outer subjects' labels, lags, features and candidate values are not read before step 5. `run_fold(..., perturb_outer=True)` overwrites them with noise to let a test (and the method check) show
that every selection output is unchanged. The O_k, L_k and R_k searches are not part of this CV.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.baselines import fit_glucobench, predict_glucobench
from src.evaluation import every_fifteen_minutes, subject_ic
from src.factor_construction import kernel_selection
from src.llm_fitness import group_splits
from src.residuals import RIDGE_ALPHAS, contrast_row, grouped_folds, oof_predictions, ridge_path_predictions, subject_error_sums, subject_table
from src.selection import one_standard_error_choice, redundancy_filter
from src.stability import family_pool

__all__ = [ "inner_scores", "inner_forward_selection", "run_fold", "aggregate_folds", "FoldSetup", "build_inputs"]

MIN_WINDOWS = 20


def _groupkfold(groups: np.ndarray, n_folds: int) -> dict:
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    return {k: sorted(int(s) for s in np.unique(groups[held])) for k, (_, held) in enumerate(GroupKFold(n_splits=n_folds, shuffle=False).split(np.zeros(len(groups)), groups=groups))}


def membership_from_table(table: pd.DataFrame, n_folds: int = 6) -> tuple:
    """Fold membership of the subjects of the CV table by `GroupKFold(n_splits=6, shuffle=False)` over its rows (group = subject). Returns (folds, information for the record)."""
    import sklearn

    folds = _groupkfold(table["subject"].to_numpy(), n_folds)
    counts = table.groupby("subject").size()
    role = table.groupby("subject")["split"].first()
    rows = [{"subject": int(s), "fold": k, "n_windows": int(counts.loc[s]), "main_pipeline_role": role.loc[s]} for k, members in folds.items() for s in members]
    frame = pd.DataFrame(rows).sort_values(["fold", "subject"]).reset_index(drop=True)
    info = {"rule": "sklearn.model_selection.GroupKFold(n_splits=6, shuffle=False); groups = subject id; one row per admitted window of the 36 subjects; no random seed in the assignment",
            "sklearn_version": sklearn.__version__, "input_rows": len(table), "input_row_order": "sorted by (subject, origin_time), as the modeling table of factor-discovery notebook (membership does not depend on the row order)",
            "folds": {str(k): {"subjects": v, "n_subjects": len(v), "windows": int(counts.loc[v].sum()), "main_training_subjects": int((role.loc[v] == "train").sum()),
                               "main_validation_subjects": int((role.loc[v] == "validation").sum())} for k, v in folds.items()}}
    return folds, info, frame


def load_membership(csv_path, expected_sha256: str) -> dict:
    """The persisted fold membership of the formal CV; refuses a file whose SHA-256 differs from the registered one."""
    import hashlib
    from pathlib import Path

    raw = Path(csv_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("the fold membership file differs from the registered SHA-256; the formal CV must use the registered membership")
    frame = pd.read_csv(csv_path)
    return {int(k): sorted(int(s) for s in g["subject"]) for k, g in frame.groupby("fold")}


# ------------------------------------------------------------------ inner (training-subject) scoring
def inner_scores(frame: pd.DataFrame, splits: list, columns: list, min_windows: int = MIN_WINDOWS) -> tuple:
    """Inner cross-validated subject-equal RMSE of a ridge block over the penalty grid: (best alpha, per-subject RMSE at it, mean RMSE, all means).

    Per inner split the block is fitted on the split's training rows (their residual is the out-of-fold baseline residual inside the training part) and scores the held-out rows, whose
    residual comes from the baseline fitted on the whole training part. Every subject is held out in exactly one split.
    """
    who, errors = [], []
    for sp in splits:
        x = frame.iloc[sp.train][columns] if columns else None
        if columns:
            prediction = ridge_path_predictions(x, sp.residual[sp.train], frame.iloc[sp.held][columns])
        else:
            prediction = np.zeros((len(sp.held), len(RIDGE_ALPHAS)))                      # the baseline alone
        errors.append(sp.residual[sp.held][:, None] - prediction)
        who.append(frame["subject"].to_numpy()[sp.held])
    ids, windows, _, squared = subject_error_sums(np.concatenate(who), np.vstack(errors))
    rmse = np.sqrt(squared / windows[:, None])
    keep = windows >= min_windows
    means = rmse[keep].mean(axis=0)
    j = int(np.argmin(means))
    return float(RIDGE_ALPHAS[j]), pd.Series(rmse[keep, j], index=pd.Index(ids[keep], name="subject"), name="rmse"), float(means[j]), means


def inner_forward_selection(frame: pd.DataFrame, splits: list, pool: list, columns_of, k_max: int = 10, min_windows: int = MIN_WINDOWS) -> tuple:
    """Forward selection on the inner cross-validated RMSE; the same subjects are scored for every model. Returns (curve, per-k subject tables)."""
    alpha, table, mean, _ = inner_scores(frame, splits, [], min_windows)
    tables = {0: table.to_frame()}
    rows = [{"k": 0, "added": "", "alpha": np.nan, "subject_equal_rmse": mean}]
    selected, remaining = [], list(pool)
    while remaining and len(selected) < k_max:
        trials = []
        for name in remaining:
            a, t, m, _ = inner_scores(frame, splits, columns_of(selected + [name]), min_windows)
            trials.append((m, pool.index(name), name, a, t))
        m, _, name, a, t = min(trials, key=lambda x: (x[0], x[1]))
        selected.append(name)
        remaining.remove(name)
        tables[len(selected)] = t.to_frame()
        rows.append({"k": len(selected), "added": name, "alpha": a, "subject_equal_rmse": m})
    return pd.DataFrame(rows), tables


# ------------------------------------------------------------------ one outer fold
@dataclass
class FoldSetup:
    """What the fold needs from the data layer (all names refer to columns of `table`)."""
    lag_columns: list
    manual_columns: list
    catalog1: pd.DataFrame            # first-batch recipes with candidate_id, family, parameters, generator_params, uses_native_state (main families only)
    batch2_builder: object              # callable(tau_star dict) -> (catalog2 DataFrame, values DataFrame indexed like `table`)
    k_per_family: int = 5
    rho_threshold: float = 0.9
    k_max: int = 10
    inner_splits: int = 3
    inner_inner_folds: int = 5


def _stage(timing: dict, name: str, t0: float) -> float:
    import psutil

    now = time.perf_counter()
    timing[name] = round(now - t0, 2)
    timing["peak_rss_mb"] = max(timing.get("peak_rss_mb", 0.0), round(psutil.Process().memory_info().rss / 2 ** 20, 1))
    return now


def _perturb(table: pd.DataFrame, rows: np.ndarray, keep: list, seed: int = 0) -> pd.DataFrame:
    """Overwrite every numeric column of the given rows (except `keep`) with large noise: used to show that nothing of the outer subjects reaches a selection step."""
    out = table.copy()
    rng = np.random.default_rng(seed)
    for c in out.columns:
        if c in keep or not pd.api.types.is_numeric_dtype(out[c]) or pd.api.types.is_bool_dtype(out[c]):
            continue
        if pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].astype(float)
        out.loc[rows, c] = rng.normal(size=int(rows.sum())) * 1e3
    return out


def run_fold(table: pd.DataFrame, outer_subjects: list, setup: FoldSetup, fold_id: int = 0, perturb_outer: bool = False, evaluate: bool = True, seed: int = 42) -> dict:
    """Run the nested pipeline for one outer fold. `table` holds the windows of all subjects with keys, `subject`, `origin_time`, `target_value`, the lags, M, `cohort_core`, `native_missing`
    and the first-batch candidate columns; it must not contain residuals or second-batch candidates of the main pipeline."""
    timing, t0, started = {}, time.perf_counter(), time.perf_counter()
    if {"residual", "base_prediction", "oof_fold"} & set(table.columns):
        raise ValueError("the residuals of the main pipeline must not enter the CV")
    outer_mask = table["subject"].isin(set(outer_subjects)).to_numpy()
    if not outer_mask.any() or outer_mask.all():
        raise ValueError("CV requires nonempty outer and training subject sets")
    keys = ["subject", "origin_time", "task_id"]
    work = _perturb(table, outer_mask, keep=keys + ["cohort_core", "split"]) if perturb_outer else table
    train_all = work.loc[~outer_mask].reset_index(drop=True)
    train_subjects = sorted(train_all["subject"].unique())
    audit = {"outer_subjects": sorted(outer_subjects), "training_subjects": [int(s) for s in train_subjects], "subjects_disjoint": not set(outer_subjects) & set(train_subjects),
             "training_windows": len(train_all), "outer_windows": int(outer_mask.sum())}
    lags = setup.lag_columns

    # 1-2 base and out-of-fold residual (Protocol B OOF design) on the training subjects
    model, scaler = fit_glucobench(train_all, lags)
    fold5 = grouped_folds(train_all["subject"], 5)
    train_all["residual"] = train_all["target_value"].to_numpy(float) - oof_predictions(train_all, fold5, lags)
    audit["base_fit_subjects"] = sorted(int(s) for s in train_all["subject"].unique())
    t0 = _stage(timing, "base_and_oof_residual_s", t0)

    # 3 screening on the training subjects only
    ids1 = list(setup.catalog1["candidate_id"])
    first = train_all.groupby("subject")["origin_time"].min()
    core = train_all[train_all["cohort_core"] & train_all["residual"].notna()]
    thin_rows = every_fifteen_minutes(core, first)
    ic1, _ = subject_ic(core[thin_rows], ids1)
    ic1.index = sorted(core.loc[thin_rows, "subject"].unique())
    summary1 = setup.catalog1[["candidate_id", "family", "parameters", "generator_params"]].copy()
    summary1["abs_mean_ic"] = summary1["candidate_id"].map(ic1.mean(skipna=True).abs()).to_numpy()
    tau_star = kernel_selection(summary1)
    catalog2, values2 = setup.batch2_builder(tau_star)
    train_all = pd.concat([train_all, values2.loc[table.index[~outer_mask]].reset_index(drop=True)], axis=1)
    ids2 = list(catalog2["candidate_id"])
    core = train_all[train_all["cohort_core"] & train_all["residual"].notna()]
    thinned = core[every_fifteen_minutes(core, first)]
    ic, _ = subject_ic(thinned, ids1 + ids2)
    ic.index = sorted(thinned["subject"].unique())
    family_of = pd.concat([setup.catalog1.set_index("candidate_id")["family"], catalog2.set_index("candidate_id")["family"]])
    pool_order = family_pool(ic, family_of, setup.k_per_family)
    native_ids = set(setup.catalog1.loc[setup.catalog1["uses_native_state"], "candidate_id"]) | set(catalog2.loc[catalog2["uses_native_state"], "candidate_id"])
    filtered = redundancy_filter(pool_order, thinned[pool_order], thinned[lags], setup.rho_threshold, "spearman")
    pool = list(filtered[filtered["decision"].eq("kept")]["candidate_id"])
    audit["ic_subjects"] = [int(s) for s in ic.index]
    audit["ic_rows_subjects_subset_of_training"] = set(audit["ic_subjects"]) <= set(train_subjects)
    t0 = _stage(timing, "screening_s", t0)

    # 4 inner forward selection inside the training subjects
    inner_frame = train_all.assign(split="train")
    splits = group_splits(inner_frame, lags, setup.inner_splits, setup.inner_inner_folds)
    audit["inner_split_subjects_subset_of_training"] = all(set(inner_frame["subject"].iloc[np.r_[sp.train, sp.held]]) <= set(train_subjects) for sp in splits)
    audit["inner_split_held_subjects_disjoint_from_their_training"] = all(not set(inner_frame["subject"].iloc[sp.train]) & set(inner_frame["subject"].iloc[sp.held]) for sp in splits)

    def columns_of(selected):
        return list(selected) + (["native_missing"] if native_ids & set(selected) else [])

    curve, tables = inner_forward_selection(inner_frame, splits, pool, columns_of, setup.k_max)
    rule = one_standard_error_choice(curve, tables)
    selected = list(curve["added"].iloc[1:rule["k_paired"] + 1])
    alpha_t = float(curve.loc[curve["k"].eq(rule["k_paired"]), "alpha"].iloc[0]) if rule["k_paired"] > 0 else np.nan
    alpha_m = inner_scores(inner_frame, splits, list(setup.manual_columns))[0]
    alpha_mt = inner_scores(inner_frame, splits, list(setup.manual_columns) + columns_of(selected))[0]
    t0 = _stage(timing, "inner_selection_s", t0)

    selection = {"fold": fold_id, "tau_star": {f"{n}|{m}": list(v) for (n, m), v in tau_star.items()}, "pool_after_ic": pool_order, "pool_after_filter": pool, "factors_in_order": list(curve["added"].iloc[1:]),
                 "k_best": rule["best_k"], "k_selected": rule["k_paired"], "selected": selected, "alpha_T": alpha_t, "alpha_M": alpha_m, "alpha_MT": alpha_mt,
                 "inner_curve": curve.round(6).to_dict("records")}
    result = {"selection": selection, "audit": audit, "timing": timing}
    if not evaluate:
        timing["total_s"] = round(time.perf_counter() - started, 2)
        return result

    # 5 outer evaluation, once
    outer_all = table.loc[outer_mask].reset_index(drop=True)
    outer_all = pd.concat([outer_all, values2.loc[table.index[outer_mask]].reset_index(drop=True)], axis=1)
    outer = outer_all[outer_all["cohort_core"]]
    train_rows = train_all[train_all["cohort_core"] & train_all["residual"].notna()]
    base_prediction = predict_glucobench(model, scaler, outer, lags)
    configs = {"base": ([], None), "base + M": (list(setup.manual_columns), alpha_m), "base + T": (columns_of(selected), alpha_t),
               "base + M + T": (list(setup.manual_columns) + columns_of(selected), alpha_mt)}
    tables_out = {}
    for name, (cols, alpha) in configs.items():
        if cols:
            j = int(np.flatnonzero(np.isclose(RIDGE_ALPHAS, alpha))[0])
            prediction = ridge_path_predictions(train_rows[cols], train_rows["residual"].to_numpy(), outer[cols])[:, j]
        else:
            prediction = np.zeros(len(outer))
        error = base_prediction + prediction - outer["target_value"].to_numpy(float)
        tables_out[name] = subject_table(outer["subject"].to_numpy(), error)
    audit["outer_core_windows"] = len(outer)
    audit["outer_evaluations"] = 1
    contrasts = {}
    for a, b in (("base", "base + M"), ("base", "base + T"), ("base + M", "base + M + T")):
        keep = tables_out[a].index[(tables_out[a]["n_windows"] >= MIN_WINDOWS)]
        d = tables_out[b].loc[keep, "rmse"] - tables_out[a].loc[keep, "rmse"]
        contrasts[f"{a} -> {b}"] = {"subjects": len(keep), "rmse_from": float(tables_out[a].loc[keep, "rmse"].mean()), "rmse_to": float(tables_out[b].loc[keep, "rmse"].mean()), "d_rmse": float(d.mean())}
    result["outer"] = {name: t.reset_index().assign(fold=fold_id, config=name) for name, t in tables_out.items()}
    result["contrasts"] = contrasts
    timing["total_s"] = round(time.perf_counter() - started, 2)
    _stage(timing, "outer_evaluation_s", t0)
    timing["total_s"] = round(time.perf_counter() - started, 2)
    return result


def aggregate_folds(results: list, n_resamples: int = 2000) -> pd.DataFrame:
    """Outer-fold contrasts over all subjects of the folds: the per-subject tables are stacked (every subject is an outer subject once) and compared with the paired subject bootstrap."""
    tables = {c: pd.concat([r["outer"][c] for r in results], ignore_index=True).set_index("subject") for c in results[0]["outer"]}
    rows = []
    for a, b in (("base", "base + M"), ("base", "base + T"), ("base + M", "base + M + T")):
        for rule, minimum in (("applied (>= 20 outer windows)", MIN_WINDOWS), ("not applied", None)):
            rows.append({"from_config": a, "to_config": b, "min_window_rule": rule, **contrast_row(tables[a], tables[b], minimum, n_resamples=n_resamples)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ explicit factor-data context
def build_inputs(ns: dict) -> tuple:
    """Table, fold setup and a description of the 36-subject Protocol-B data from the explicit context of `factor_data.load_context`.

    From the main pipeline only raw inputs and labels are kept (keys, target, lags, M, cohort flag, native-reading indicator, first-batch candidates); the residuals and the second-batch
    candidates of the main pipeline (selected on the 27 training subjects) are dropped, because every fold recomputes them."""
    name = "Unseen-subject Generalization"
    full = ns["MODEL_TABLES"][name]
    catalog = ns["catalog"]
    batch2_ids = set(catalog.loc[catalog["batch"].eq(2), "candidate_id"])
    cat1 = ns["catalog_batch1"]
    cat1 = cat1[~cat1["family"].eq("meal_x_lab_state_conditional")].reset_index(drop=True)
    drop = {"residual", "residual_in_sample", "oof_fold", "oof_prediction", "base_prediction"} | batch2_ids
    table = full[[c for c in full.columns if c not in drop]].reset_index(drop=True)
    assert set(cat1["candidate_id"]) <= set(table.columns) and len(cat1) == 187
    keys = ns["WINDOW_KEY"]
    inputs = ns["get_inputs"](name)
    cache = {}

    def builder(tau_star: dict) -> tuple:
        key = tuple(sorted(tau_star.items()))
        if key not in cache:
            cat2 = ns["batch2_catalog"](tau_star, name)
            both = pd.concat([ns["catalog_batch1"], cat2], ignore_index=True)
            wanted = list(cat2["candidate_id"])
            frame = ns["generate_matrix"](name, both, inputs, wanted=wanted)
            merged = table[[*keys, "split"]].merge(frame[[*keys, "split", *wanted]], on=[*keys, "split"], how="left", validate="one_to_one")
            values = merged[wanted].copy()
            values.index = table.index
            cache[key] = (cat2, values)
        return cache[key]

    setup = FoldSetup(lag_columns=list(ns["lag_columns"][name]), manual_columns=list(ns["MANUAL"]), catalog1=cat1, batch2_builder=builder)
    info = {"subjects": int(table["subject"].nunique()), "windows": len(table), "splits_in_main_pipeline": table["split"].value_counts().to_dict(),
            "windows_per_subject": table.groupby("subject").size()}
    return table, setup, info
