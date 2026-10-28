"""Training/validation stability, 36-subject grouped cross-validation and native-anchor sensitivity."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import e8_anchor as E8
from src import grouped_cv as G
from src import factor_data as data
from src import stability as ST
from src.evaluation import every_fifteen_minutes, subject_ic


def _json(path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def compute_stability(ns: dict, out_dir, n_pairs: int = 50, seed: int = 42, loso: bool = True) -> dict:
    """E4: selection stability of the IC + Lasso selector over complementary half-subject pairs, LOSO Jaccard, A vs B Jaccard, same-sign share; cross-environment sensitivity."""
    started = time.perf_counter()
    screening, summary, rows, full_sets, stable_sets = ns["screening"], {}, [], {}, {}
    ics = {}
    for tag, name in data.PROTOCOLS.items():
        table = ns["MODEL_TABLES"][name]
        train = table[table["split"].eq("train")]
        first = train.groupby("subject")["origin_time"].min()
        core = train[train["cohort_core"] & train["residual"].notna()]
        thinned = core[every_fifteen_minutes(core, first)]
        subjects = sorted(train["subject"].unique())
        main = screening[screening["evaluation_protocol"].eq(name)]
        ids, family_of = list(main["candidate_id"]), main.set_index("candidate_id")["family"]
        ic, _ = subject_ic(thinned, ids)
        ics[tag] = ic
        full = ST.select_ic_lasso(thinned, ic, subjects, family_of)
        full_sets[tag] = full
        pairs = ST.complementary_pairs(subjects, n_pairs, seed)
        selections = [[ST.select_ic_lasso(thinned, ic, half, family_of) for half in pair] for pair in pairs]
        halves = [s for pair in selections for s in pair]
        freq, sets = ST.selection_frequency(halves, ids)
        within_pair = [ST.jaccard(*pair) for pair in selections]
        family_freq = {f: float(np.mean([any(family_of[c] == f for c in s) for s in halves])) for f in sorted(set(family_of))}
        loso_table = ST.loso_jaccard(lambda rest: ST.select_ic_lasso(thinned, ic, rest, family_of), subjects, full) if loso else pd.DataFrame()
        loso_freq = ST.selection_frequency(list(loso_table["selected"]), ids)[0] if loso else pd.Series(np.nan, index=ids)
        same = ST.same_sign_share(ic, ids).set_index("candidate")
        t_k = set(ns["T_K"][name])
        for c in ids:
            rows.append({"evaluation_protocol": name, "candidate_id": c, "family": family_of[c], "selection_frequency_halves": float(freq[c]), **{f"stable_at_{t}": c in sets[t] for t in sets},
                         "loso_inclusion_frequency": float(loso_freq.get(c, np.nan)), "in_full_sample_selection": c in full, "in_T_k": c in t_k, "mean_ic": float(same.loc[c, "mean_ic"]),
                         "subjects_with_ic": int(same.loc[c, "subjects_with_ic"]), "same_sign_share": float(same.loc[c, "same_sign_share"])})
        stable_sets[tag] = sets
        summary[tag] = {"training_subjects": len(subjects), "pairs": n_pairs, "half_sizes": [len(pairs[0][0]), len(pairs[0][1])], "thresholds": {str(t): len(v) for t, v in sets.items()},
                        "headline_threshold": 0.6, "stable_at_0.6": sets[0.6], "T_k_members_stable_at_0.6": sorted(set(sets[0.6]) & t_k), "T_k_size": len(t_k),
                        "within_pair_jaccard_mean": float(np.mean(within_pair)), "within_pair_jaccard_min": float(np.min(within_pair)), "family_selection_frequency": family_freq,
                        "full_sample_selection": full, "loso_jaccard_mean": float(loso_table["jaccard_with_full"].mean()) if loso else None,
                        "loso_jaccard_min": float(loso_table["jaccard_with_full"].min()) if loso else None, "loso_jaccard_max": float(loso_table["jaccard_with_full"].max()) if loso else None,
                        "same_sign_share_of_full_selection": {c: float(same.loc[c, "same_sign_share"]) for c in full}}
    summary["A_vs_B"] = {"jaccard_full_sample_selection": ST.jaccard(full_sets["A"], full_sets["B"]), "jaccard_stable_at_0.6": ST.jaccard(stable_sets["A"][0.6], stable_sets["B"][0.6])}
    pd.DataFrame(rows).to_csv(out_dir / "selection_stability.csv", index=False)
    # cross-environment sensitivity: the selected T_k of each protocol across three HbA1c groups of the protocol's own training subjects
    meta = pd.read_csv(data.FACTOR_DIR / "metabolic_covariates.csv").set_index("subject")
    curves, env_rows = ns["curves"], []
    for tag, name in data.PROTOCOLS.items():
        table = ns["MODEL_TABLES"][name]
        train = table[table["split"].eq("train")]
        first = train.groupby("subject")["origin_time"].min()
        core = train[train["cohort_core"] & train["residual"].notna()]
        thinned = core[every_fifteen_minutes(core, first)]
        subjects = sorted(train["subject"].unique())
        selected = ns["T_K"][name]
        alpha = float(curves[curves["evaluation_protocol"].eq(name) & curves["k"].eq(len(selected))]["alpha"].iloc[0])
        indicator = "native_missing" if ns["NATIVE_IDS"] & set(selected) else None
        env_rows.append(ST.group_stability(thinned, ics[tag], selected, ST.metabolic_group(meta.loc[subjects, "hba1c_percent"]), alpha, indicator).assign(evaluation_protocol=name, alpha=alpha))
    pd.concat(env_rows, ignore_index=True).to_csv(out_dir / "cross_environment_stability.csv", index=False)
    summary["seconds"] = round(time.perf_counter() - started, 1)
    _json(out_dir / "e4_summary.json", summary)
    return summary


def compute_grouped_cv(ns: dict, out_dir, folds=None, n_resamples: int = 2000) -> dict:
    """Six outer folds of nested subject-grouped cross-validation (the registered GroupKFold membership file) and the stacked outer-fold contrasts."""
    started = time.perf_counter()
    table, setup, info = G.build_inputs(ns)
    settings = json.loads((data.FACTOR_DIR / "discovery_settings.json").read_text(encoding="utf-8"))["cv_36"]
    membership = G.load_membership(data.FACTOR_DIR / settings["membership_file"], settings["membership_sha256"])      # the registered membership; never regenerated
    assert sorted(s for v in membership.values() for s in v) == sorted(table["subject"].unique())
    results = []
    for k in (folds if folds is not None else sorted(membership)):
        results.append(G.run_fold(table, membership[k], setup, fold_id=k))
        (out_dir / f"grouped_cv_fold{k}.json").write_text(json.dumps({"selection": results[-1]["selection"], "audit": results[-1]["audit"], "timing": results[-1]["timing"],
                                                                      "contrasts": results[-1]["contrasts"]}, indent=1, default=str), encoding="utf-8")
    aggregate = G.aggregate_folds(results, n_resamples)
    aggregate.to_csv(out_dir / "grouped_cv_results.csv", index=False)
    pd.concat([pd.concat(r["outer"].values()) for r in results]).to_csv(out_dir / "grouped_cv_subject_rmse.csv", index=False)
    factors = pd.Series([f for r in results for f in r["selection"]["selected"]]).value_counts()
    summary = {"folds": [r["selection"]["fold"] for r in results], "membership": {str(k): membership[k] for k in membership}, "k_selected": {str(r["selection"]["fold"]): r["selection"]["k_selected"] for r in results},
               "factor_selection_count_over_folds": factors.to_dict(), "main_pipeline_T_7": ns["T_K"]["Unseen-subject Generalization"], "seconds": round(time.perf_counter() - started, 1),
               "peak_rss_mb": max(r["timing"]["peak_rss_mb"] for r in results)}
    _json(out_dir / "grouped_cv_summary.json", summary)
    return summary


def compute_anchor_sensitivity(ns: dict, out_dir, subjects_by_protocol: dict | None = None, n_resamples: int = 2000) -> dict:
    """E8-a (anchor-causal base against the original on the same reliable validation windows) and E8-b (the selected M, T_k, O_k blocks on anchor-causal residuals)."""
    started = time.perf_counter()
    rows_a, rows_b, reports, audits, subject_rows = [], [], [], {}, []
    for tag, name in data.PROTOCOLS.items():
        table = ns["O_TABLES"][name]
        if subjects_by_protocol:
            table = table[table["subject"].isin(subjects_by_protocol[tag])]
        lags = ns["lag_columns"][name]
        _, minutes, _ = ns["get_inputs"](name)
        rebuilt, info = E8.rebuild_lags(minutes, table[["subject", "origin_time"]])
        audits[tag] = {"identity": E8.identity_check(rebuilt, table[lags], info), "causality": E8.causality_check(minutes, table[["subject", "origin_time"]], sample=120)}
        reports.append(E8.window_report(table[["subject", "split", "cohort_core"]], info).assign(evaluation_protocol=name))
        a = E8.e8a_compare(table, rebuilt, info, lags, n_resamples)
        rows_a.append(a["contrasts"].assign(evaluation_protocol=name, fit_windows=a["fit_windows"], evaluation_windows=a["evaluation_windows"], evaluation_subjects=a["evaluation_subjects"],
                                            excluded_share_of_validation_core=a["excluded_share_of_validation_core"]))
        for label, t in a["subject_tables"].items():
            subject_rows.append(t.reset_index().assign(evaluation_protocol=name, base=label))
        bounds = None
        if tag == "A":
            scope = pd.read_csv(ns["DATA_QUALITY_DIR"] / "split_scope_v1.csv", parse_dates=["start_time", "protocol_a_train_end_time"]).set_index("subject_id")
            bounds = pd.DataFrame({"lo": scope["start_time"], "hi": scope["protocol_a_train_end_time"]})
        anchored = E8.anchor_residuals(table, rebuilt, info, lags, tag == "A", bounds)
        m, t_cols, o_cols = ns["MANUAL"], ns["model_columns"](ns["T_K"][name]), ns["o_model_columns"](ns["O_K"][name])
        blocks = {"+ M": m, "+ T": t_cols, "+ O": o_cols, "+ M + T": m + t_cols, "+ M + O": m + o_cols}
        rows_b.append(E8.e8b_blocks(table, anchored, blocks, n_resamples).assign(evaluation_protocol=name))
    pd.concat(rows_a, ignore_index=True).to_csv(out_dir / "e8_base_validation.csv", index=False)
    pd.concat(rows_b, ignore_index=True).to_csv(out_dir / "e8_validation.csv", index=False)
    pd.concat(reports, ignore_index=True).to_csv(out_dir / "e8_window_report.csv", index=False)
    pd.concat(subject_rows, ignore_index=True).to_csv(out_dir / "e8_subject_rmse.csv", index=False)
    summary = {"audits": audits, "seconds": round(time.perf_counter() - started, 1)}
    _json(out_dir / "e8_summary.json", summary)
    return summary




def make_figures(input_dir, output_dir) -> None:
    """Figures of the formal development-data analysis outputs (read from the saved files; nothing is recomputed)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stab = pd.read_csv(input_dir / "selection_stability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (name, part) in zip(axes, stab.groupby("evaluation_protocol")):
        top = part.sort_values("selection_frequency_halves", ascending=False).head(14).iloc[::-1]
        ax.barh(top["candidate_id"], top["selection_frequency_halves"], color=np.where(top["in_T_k"], "tab:orange", "tab:blue"))
        for t in (0.5, 0.6, 0.7):
            ax.axvline(t, color="gray", lw=0.8, ls=":")
        ax.set_xlim(0, 1)
        ax.set_xlabel("selection frequency, 100 half-samples (orange: T_k)", fontsize=8)
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "stability_frequency.png", dpi=150)
    plt.close(fig)


    sub = pd.read_csv(input_dir / "grouped_cv_subject_rmse.csv")
    wide = sub.pivot(index="subject", columns="config", values="rmse")
    pairs = {"base + M minus base": wide["base + M"] - wide["base"], "base + T minus base": wide["base + T"] - wide["base"], "base + M + T minus base + M": wide["base + M + T"] - wide["base + M"]}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (label, v) in enumerate(pairs.items()):
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(42).uniform(-0.12, 0.12, len(v)), s=16, alpha=0.7)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color="black")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_yticks(range(3), list(pairs))
    ax.set_xlabel("outer-fold RMSE difference per subject (mg/dL; negative = lower error)", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "grouped_cv_subject_deltas.png", dpi=150)
    plt.close(fig)

    e8 = pd.read_csv(input_dir / "e8_subject_rmse.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (name, part) in zip(axes, e8.groupby("evaluation_protocol")):
        w = part[part["n_windows"] >= 20].pivot(index="subject", columns="base", values="rmse").dropna()
        d = w["anchor-causal base"] - w["original base"]
        ax.scatter(d, np.random.default_rng(42).uniform(-0.2, 0.2, len(d)), s=16, alpha=0.7)
        ax.plot([d.mean()] * 2, [-0.4, 0.4], color="black")
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_yticks([])
        ax.set_xlabel("anchor-causal minus original base, subject RMSE (mg/dL)", fontsize=8)
        ax.set_title(f"{name}: {len(d)} subjects", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "anchor_sensitivity.png", dpi=150)
    plt.close(fig)


def plot_search_results(input_dir, output_dir) -> None:
    """Plot saved selection paths and subject RMSE differences; no fitting or resampling.

    Curves use each method's saved eligible-subject selection path. Subject plots
    use the saved core-cohort E3 fits and retain subjects with >=20 base windows.
    The black mark is the unweighted mean paired difference, not a confidence interval.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    subjects = pd.read_csv(input_dir / "e3_subject_rmse.csv")
    curves = {method: pd.read_csv(input_dir / f"validation_curves_{method}.csv") for method in ("T", "O", "L", "R")}
    protocols = (data.PROTOCOLS["A"], data.PROTOCOLS["B"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, protocol in zip(axes, protocols):
        for method, curve in curves.items():
            part = curve[curve.evaluation_protocol.eq(protocol)].sort_values("k")
            ax.plot(part.k, part.subject_equal_rmse, marker="o", ms=3, label=method)
        ax.set(title=protocol, xlabel="Number of added factors", ylabel="Validation subject-equal RMSE (mg/dL)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "search_validation_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, protocol in zip(axes, protocols):
        part = subjects[subjects.evaluation_protocol.eq(protocol)]
        keep = part.loc[part.config.eq("base") & part.n_windows.ge(20), "subject"]
        wide = part.pivot(index="subject", columns="config", values="rmse").loc[keep]
        differences = {"L - base": wide["base + L"] - wide["base"], "R - base": wide["base + R"] - wide["base"], "L - R": wide["base + L"] - wide["base + R"]}
        for i, values in enumerate(differences.values()):
            ax.scatter(values, i + np.random.default_rng(42).uniform(-0.12, 0.12, len(values)), s=14, alpha=0.7)
            ax.plot([values.mean()] * 2, [i - 0.3, i + 0.3], color="black")
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_yticks(range(3), list(differences))
        ax.set(title=f"{protocol}\n{len(wide)} subjects", xlabel="Per-subject validation RMSE difference (mg/dL)")
    fig.tight_layout()
    fig.savefig(output_dir / "search_subject_differences.png", dpi=150)
    plt.close(fig)
