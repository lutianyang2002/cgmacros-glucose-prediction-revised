"""Comparison configurations and selected penalties reconstructed from saved validation tables.

The evaluation notebook reads the comparison plan without changing it. This module
reconstructs the corresponding scientific registry from validation artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROTOCOLS = {"A": "Within-subject Future Prediction", "B": "Unseen-subject Generalization"}
NATIVE_STATE_INPUTS = {"native_last", "native_diff", "native_slope_1h", "native_curv_1h"}
E9_SETTINGS = {"delay_free": "Dinner and Snack delayed 30 minutes", "delay_all": "all meals delayed 30 minutes", "logged": "logged time"}


def validate_saved_results(plan: dict, completion: dict, gains: pd.DataFrame, metrics: pd.DataFrame, strata: pd.DataFrame) -> pd.DataFrame:
    """Validate the completed 50-comparison release tables, without recomputing statistics.

    Coverage follows the plan's cohorts, minimum-window reporting and subgroup families.
    This checks table structure and values, not historical execution-code identity.
    """
    comparisons = pd.DataFrame(plan["comparisons"])
    if (completion.get("status") != "COMPLETED" or not comparisons["id"].is_unique
            or comparisons.groupby("protocol_tag").size().to_dict() != {"A": 24, "B": 26}
            or comparisons.duplicated(["evaluation_protocol", "from_config", "to_config"]).any()):
        raise ValueError("expected a completed evaluation with 50 unique planned comparisons (A24/B26)")
    primary = comparisons[comparisons["family"].eq("primary")]
    if any(set(primary.loc[primary["protocol_tag"].eq(tag), "id"]) != {f"{tag}:P{i}" for i in range(1, 5)} for tag in PROTOCOLS):
        raise ValueError("the primary family must contain P1-P4 for each protocol")
    rules = (f"applied (>= {plan['metrics']['min_windows_per_subject']} windows)", "not applied")
    key = ["id", "cohort", "stratum", "group", "min_window_rule"]
    expected_gains, expected_strata = set(), set()
    for c in comparisons.itertuples():
        short = c.id.split(":")[1]
        for rule in rules:
            expected_gains.update((c.id, cohort, "all", "all", rule) for cohort in ("core", "full"))
            if short in plan["metrics"]["metabolic_groups"]["applied_to"]:
                expected_gains.update((c.id, "core", "all", group, rule) for group in ("normal (<5.7)", "prediabetes (5.7-6.4)", "diabetes (>6.4)"))
            if short in plan["metrics"]["strata"]["applied_to"]:
                expected_strata.update((c.id, "core", stratum, "all", rule) for stratum in ("P30", "P31_120", "NoMeal120"))
    metadata = ["protocol_tag", "evaluation_protocol", "family", "from_config", "to_config"]
    numbers = ["subjects", "rmse_from", "rmse_to", "d_rmse", "d_rmse_low", "d_rmse_high", "d_mae", "pooled_d_rmse", "improved_share"]
    for name, frame, expected in (("comparisons", gains, expected_gains), ("meal strata", strata, expected_strata)):
        if not set(key + metadata + numbers + ["p_bootstrap", "p_holm_primary_family"]) <= set(frame):
            raise ValueError(f"{name}: missing result columns")
        if frame[key].isna().any().any() or frame.duplicated(key).any() or set(frame[key].itertuples(index=False, name=None)) != expected:
            raise ValueError(f"{name}: duplicate, missing or unexpected comparison/cohort/subset/rule rows")
        registered = comparisons.set_index("id").loc[frame["id"]]
        if not frame[metadata].reset_index(drop=True).equals(registered[metadata].reset_index(drop=True)):
            raise ValueError(f"{name}: configuration or protocol metadata differs from the plan")
        computable = frame["subjects"] >= 2
        if not np.isfinite(frame.loc[computable, numbers].to_numpy(float)).all() or not np.isfinite(frame["subjects"]).all() or (frame["subjects"] < 0).any():
            raise ValueError(f"{name}: invalid subject counts or nonfinite computable statistics")
        for field in ("p_bootstrap", "p_holm_primary_family"):
            required = computable & frame["stratum"].eq("all") & frame["group"].eq("all")
            if field == "p_holm_primary_family":
                required &= frame["family"].eq("primary") & frame["cohort"].eq("core") & frame["min_window_rule"].eq(rules[0])
            values = frame[field]
            if values[required].isna().any() or not values.dropna().between(0, 1).all():
                raise ValueError(f"{name}: invalid or missing {field}")
    metric_key = ["protocol_tag", "config", "cohort"]
    metric_values = ["subjects", "subjects_ge_20", "windows", "rmse_subject_equal_rule_applied", "rmse_subject_equal_all_subjects", "rmse_pooled", "mae_subject_equal_all_subjects"]
    expected_metrics = {(tag, config, cohort) for tag, configs in plan["configs"].items() for config in configs for cohort in ("core", "full")}
    if (not set(metric_key + metric_values) <= set(metrics) or metrics.duplicated(metric_key).any()
            or set(metrics[metric_key].itertuples(index=False, name=None)) != expected_metrics
            or not np.isfinite(metrics[metric_values].to_numpy(float)).all()):
        raise ValueError("configuration metrics: missing, duplicate or nonfinite configuration/cohort results")
    return comparisons


def _applied(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["min_window_rule"].astype(str).str.startswith("applied")]


class AlphaRegistry:
    """alpha of every configuration with the artifacts it was read from; a conflict between artifacts is an error."""

    def __init__(self):
        self.values, self.sources = {}, {}

    def add(self, key: str, alpha: float, source: str) -> None:
        alpha = None if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)) else float(alpha)
        if key in self.values and self.values[key] != alpha:
            raise ValueError(f"alpha conflict for {key}: {self.values[key]} ({self.sources[key]}) against {alpha} ({source})")
        self.values[key] = alpha
        self.sources.setdefault(key, []).append(source)


def build_registry(art: Path, selection: dict) -> tuple:
    """Configurations (with validation-selected penalties) and the comparison list of both protocols.

    `selection` = {"T": {protocol_tag: [ids]}, "O": {protocol_tag: [ids]}, "family_of": {candidate id: family}, "native_ids": set, "o_native": {O id: bool}} from saved factor selections.
    """
    e1 = _applied(pd.read_csv(art / "e1_validation.csv"))
    e1 = e1[e1["cohort"] == "cohort_core"]
    e2 = _applied(pd.read_csv(art / "e2_validation.csv"))
    e2o = _applied(pd.read_csv(art / "e2_O_validation.csv"))
    e5 = _applied(pd.read_csv(art / "e5_validation.csv"))
    e9 = _applied(pd.read_csv(art / "e9_validation.csv"))
    e8b = _applied(pd.read_csv(art / "e8_validation.csv"))
    configs, comparisons, problems = {}, [], []
    for tag, name in PROTOCOLS.items():
        alphas = AlphaRegistry()
        for frame, label in ((e2, "e2_validation.csv"), (e2o, "e2_O_validation.csv")):
            for r in frame[frame["evaluation_protocol"] == name].itertuples():
                for cfg, a in ((r.from_config, r.alpha_from), (r.to_config, r.alpha_to)):
                    alphas.add(f"{tag}|{cfg}", a, label)
        for r in e1[e1["evaluation_protocol"] == name].itertuples():
            for cfg, a in ((r.from_config, r.alpha_from), (r.to_config, r.alpha_to)):
                alphas.add(f"{tag}|E1:{cfg}", a, "e1_validation.csv")
        for r in e5[e5["evaluation_protocol"] == name].itertuples():
            alphas.add(f"{tag}|base + T", r.alpha_full, "e5_validation.csv")
            alphas.add(f"{tag}|base + T minus {r.family_removed}", r.alpha_reduced, "e5_validation.csv")
        for r in e9[(e9["evaluation_protocol"] == name) & (e9["record_type"] == "gain over the base")].itertuples():
            version = next(k for k, v in E9_SETTINGS.items() if v == r.setting)
            alphas.add(f"{tag}|{r.model}@{version}", r.alpha, "e9_validation.csv")
        for r in e8b[(e8b["evaluation_protocol"] == name) & e8b["contrast_type"].str.startswith("marginal") & (e8b["lags"] == "anchor-causal lags")].itertuples():
            alphas.add(f"{tag}|anchor-causal base {r.block}", r.alpha, "e8_validation.csv")
        # the E1 configuration with all groups is the model with M, the logged-time E9 models are the ordinary ones: same alpha required
        for cfg_a, cfg_b in ((f"{tag}|E1:G+A+D+B", f"{tag}|base + M"), (f"{tag}|base + M@logged", f"{tag}|base + M"), (f"{tag}|base + T@logged", f"{tag}|base + T")):
            if alphas.values.get(cfg_a, "x") != alphas.values.get(cfg_b, "y"):
                problems.append(f"{cfg_a} {alphas.values.get(cfg_a)} vs {cfg_b} {alphas.values.get(cfg_b)}")
        t_ids, o_ids = selection["T"][tag], selection["O"][tag]
        t_native = bool(selection["native_ids"] & set(t_ids))

        def block(name_, columns, alpha_key, **extra):
            configs.setdefault(tag, {})[name_] = {"columns": columns, "alpha": alphas.values.get(alpha_key), "alpha_sources": alphas.sources.get(alpha_key, []), **extra}

        t_cols = list(t_ids) + (["native_missing"] if t_native else [])
        o_cols = list(o_ids) + (["native_missing"] if any(selection["o_native"][o] for o in o_ids) else [])
        block("base", [], f"{tag}|base")
        block("base + M", ["<M>"], f"{tag}|base + M")
        block("base + T", t_cols, f"{tag}|base + T")
        block("base + O", o_cols, f"{tag}|base + O")
        block("base + M + T", ["<M>"] + t_cols, f"{tag}|base + M + T")
        block("base + M + O", ["<M>"] + o_cols, f"{tag}|base + M + O")
        for cfg in sorted({c for c in (e1[e1["evaluation_protocol"] == name]["from_config"].tolist() + e1[e1["evaluation_protocol"] == name]["to_config"].tolist())}):
            block(f"E1:{cfg}", [f"<groups {cfg}>"], f"{tag}|E1:{cfg}", groups=cfg)
        for r in e5[e5["evaluation_protocol"] == name].itertuples():
            rest = [c for c in t_ids if selection["family_of"][c] != r.family_removed]
            rest_cols = rest + (["native_missing"] if selection["native_ids"] & set(rest) else [])
            block(f"base + T minus {r.family_removed}", rest_cols, f"{tag}|base + T minus {r.family_removed}", removed_family=r.family_removed, removed=[c for c in t_ids if selection["family_of"][c] == r.family_removed])
        for version in ("logged", "delay_free", "delay_all"):
            block(f"base + M@{version}", ["<M>"], f"{tag}|base + M@{version}", meal_version=version)
            block(f"base + T@{version}", t_cols, f"{tag}|base + T@{version}", meal_version=version)
        block("original base, reliable windows", [], f"{tag}|base", lags="original", windows="core_and_anchor_reliable")
        block("anchor-causal base", [], f"{tag}|base", lags="anchor_causal", windows="core_and_anchor_reliable")
        block("anchor-causal base + M", ["<M>"], f"{tag}|anchor-causal base + M", lags="anchor_causal", windows="core_and_anchor_reliable")
        block("anchor-causal base + T", t_cols, f"{tag}|anchor-causal base + T", lags="anchor_causal", windows="core_and_anchor_reliable")
        block("anchor-causal base + O", o_cols, f"{tag}|anchor-causal base + O", lags="anchor_causal", windows="core_and_anchor_reliable")

        def add(cid, family, role, frm, to, purpose, ref=None, **extra):
            comparisons.append({"id": f"{tag}:{cid}", "protocol_tag": tag, "evaluation_protocol": name, "family": family, "role": role, "from_config": frm, "to_config": to, "purpose": purpose,
                                "validation_reference": ref, "windows": extra.pop("windows", "core"), **extra})

        e2o_ref = lambda f, t: {"artifact": "e2_O_validation.csv", "from_config": f, "to_config": t}
        e2_ref = lambda f, t: {"artifact": "e2_validation.csv", "from_config": f, "to_config": t}
        add("P1", "primary", "primary (Holm family of 4; descriptive)", "base", "base + M", "RQ1: the existing features on the strong base", e2o_ref("base", "base + M"))
        add("P2", "primary", "primary (Holm family of 4; descriptive)", "base", "base + T", "RQ1/RQ3: constructed factors T_k", e2o_ref("base", "base + T"))
        add("P3", "primary", "primary (Holm family of 4; descriptive)", "base + M", "base + M + T", "RQ1: T_k on top of M", e2o_ref("base + M", "base + M + T"))
        add("P4", "primary", "primary (Holm family of 4; descriptive)", "base", "base + O", "RQ3: OpenFE factors O_k", e2o_ref("base", "base + O"))
        add("S1", "secondary E2", "secondary descriptive", "base", "base + M + T", "combination M + T against the base",
            {"artifact": "e2_validation.csv", "derived": "rmse_to(base + M + T) - rmse_from(base)", "from_config": "base + M", "to_config": "base + M + T"})
        add("S2", "secondary E2", "secondary descriptive", "base", "base + M + O", "combination M + O against the base",
            {"artifact": "e2_O_validation.csv", "derived": "rmse_to(base + M + O) - rmse_from(base)", "from_config": "base + M", "to_config": "base + M + O"})
        add("S3", "secondary E2", "secondary, descriptive; not a Holm primary", "base + T", "base + O", "H3: direct comparison of the constructor with OpenFE", e2o_ref("base + T", "base + O"))
        for n, r in enumerate(e1[e1["evaluation_protocol"] == name].itertuples(), start=1):
            add(f"E1-{n}", "E1", "secondary descriptive", f"E1:{r.from_config}", f"E1:{r.to_config}", f"E1 {r.contrast_id}: group {r.group_tested}",
                {"artifact": "e1_validation.csv", "contrast_id": r.contrast_id, "cohort": "cohort_core"})
        for n, r in enumerate(e5[e5["evaluation_protocol"] == name].itertuples(), start=1):
            add(f"E5-{n}", "E5", "secondary descriptive", "base + T", f"base + T minus {r.family_removed}", f"E5: remove family {r.family_removed} from T_k",
                {"artifact": "e5_validation.csv", "family_removed": r.family_removed})
        n = 0
        for model in ("base + M", "base + T"):
            for version in ("delay_free", "delay_all"):
                n += 1
                add(f"E9-{n}", "E9", "timing sensitivity (descriptive)", f"{model}@logged", f"{model}@{version}", f"E9: {model} with {E9_SETTINGS[version]} against logged times",
                    {"artifact": "e9_validation.csv", "record_type": "same model, delayed minus logged", "setting": E9_SETTINGS[version], "model": model})
        add("E8-1", "E8 minimal", "exploratory validity sensitivity", "original base, reliable windows", "anchor-causal base", "E8-a: loss from rebuilding the base from native readings",
            {"artifact": "e8_base_validation.csv", "from": "original base", "to": "anchor-causal base"}, windows="core_and_anchor_reliable")
        for n, block_name in enumerate(("M", "T", "O"), start=2):
            add(f"E8-{n}", "E8 minimal", "exploratory validity sensitivity", "anchor-causal base", f"anchor-causal base + {block_name}", f"E8-b: marginal gain of {block_name} on the anchor-causal base",
                {"artifact": "e8_validation.csv", "contrast_type": "marginal effect of the block on the base of the same lags", "lags": "anchor-causal lags", "block": f"+ {block_name}"}, windows="core_and_anchor_reliable")
    if problems:
        raise ValueError("configurations that should share an alpha do not: " + "; ".join(problems))
    return configs, comparisons


def selection_from_files(art: Path) -> dict:
    """T_k, O_k, families and native-state flags from the selection files and catalog."""
    from src.openfe_runner import formula_columns
    from src.window_data import registries

    catalog_path = art / "candidate_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    st, so = pd.read_csv(art / "selection_T.csv"), pd.read_csv(art / "selection_O.csv")
    feats = pd.read_csv(art / "openfe_features.csv")
    t, o = {}, {}
    for tag, name in PROTOCOLS.items():
        t[tag] = json.loads(st.loc[st["evaluation_protocol"] == name, "selected_set"].iloc[0])
        o[tag] = json.loads(so.loc[so["evaluation_protocol"] == name, "selected_set"].iloc[0])
    formulas = feats.set_index("candidate_id")["formula"].to_dict()
    return {"T": t, "O": o, "family_of": catalog.drop_duplicates("candidate_id").set_index("candidate_id")["family"].to_dict(),
            "native_ids": set(catalog.loc[catalog["uses_native_state"], "candidate_id"]), "M": registries()["MANUAL"], "formulas": formulas,
            "o_native": {c: bool(NATIVE_STATE_INPUTS & set(formula_columns(formulas[c]))) for ids in o.values() for c in ids}, "catalog": catalog}
