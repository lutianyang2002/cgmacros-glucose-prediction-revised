"""Explicit train/validation inputs for discovery and downstream analyses.

Loading predicts with the stored baseline, without fitting, executing notebooks,
searching or writing. Filtering mixed-source files is statistical isolation only.
Minute inputs are loaded on demand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import window_data as W
from src.factor_construction import batch1_catalog, batch2_catalog
from src.openfe_runner import formula_columns
from src.paths import PROJECT_ROOT

FACTOR_DIR = PROJECT_ROOT / "results" / "factor_discovery"
PROTOCOLS = {"A": W.PROTOCOL_A, "B": W.PROTOCOL_B}


def load_context(include_factors: bool = True, include_openfe: bool = True, factor_dir=None, openfe_dir=None, openfe_selection_dir=None,
                 require_openfe_selection: bool = True, load_residuals: bool = True) -> dict:
    """Load explicit factor inputs; OpenFE pool inputs may come from a separate reconstruction.

    ``factor_dir`` must contain the required upstream artifacts, not just an empty output
    directory. ``openfe_dir`` supplies features/values and, when required, the O selection.
    Residual reconstruction uses ``include_factors=False, load_residuals=False`` to
    load upstream inputs without requiring a previous OOF residual table.
    """
    factor_dir = FACTOR_DIR if factor_dir is None else Path(factor_dir)
    openfe_dir = factor_dir if openfe_dir is None else Path(openfe_dir)
    openfe_selection_dir = openfe_dir if openfe_selection_dir is None else Path(openfe_selection_dir)
    registry = W.registries()
    tables, lags, inputs = {}, {}, {}
    for name in PROTOCOLS.values():
        tables[name], lags[name] = W.build_table(name, W.PRE_LOCK_SPLITS, registry=registry, factor_dir=factor_dir,
                                              **({"load_residuals": False} if not load_residuals else {}))
    covariates = pd.read_csv(factor_dir / "metabolic_covariates.csv")
    bmi = covariates.set_index("subject")["bmi_kg_m2"]

    def get_inputs(name):
        if name not in inputs:
            keys = tables[name][W.WINDOW_KEY + ["split"]].copy()
            minutes, events = W.load_inputs(keys, W.PRE_LOCK_SPLITS)
            inputs[name] = keys, minutes, events
        return inputs[name]

    def generate_matrix(name, catalog, source, version="logged", wanted=None):
        keys, minutes, events = source
        wanted = list(catalog["candidate_id"]) if wanted is None else wanted
        return W.candidate_matrix(keys, minutes, events, catalog, wanted, bmi, version)

    context = {**registry, "factor_dir": factor_dir, "tables": tables, "lag_columns": lags, "get_inputs": get_inputs,
               "generate_matrix": generate_matrix, "BMI": bmi, "covariates": covariates,
               "DATA_QUALITY_DIR": W.DATA_QUALITY_DIR, "WINDOW_KEY": W.WINDOW_KEY, "EVENT_COLUMNS": W.EVENT_COLUMNS,
               "PROTOCOL_A": W.PROTOCOL_A, "PROTOCOL_B": W.PROTOCOL_B,
               "catalog_batch1": batch1_catalog(), "batch2_catalog": batch2_catalog}
    if not include_factors:
        return context

    catalog_path = factor_dir / "candidate_catalog.csv"
    catalog = pd.read_csv(catalog_path)
    native_ids = set(catalog.loc[catalog["uses_native_state"], "candidate_id"])
    key = W.WINDOW_KEY + ["split"]
    native_flag = pd.concat([pd.read_parquet(factor_dir / f"primitives_{tag}.parquet", columns=key + ["native_missing"])
                             for tag in PROTOCOLS]).drop_duplicates(key)
    matrices = [pd.read_parquet(factor_dir / f"candidates_batch{batch}.parquet") for batch in (1, 2)]
    model_tables = {}
    for name, table in tables.items():
        frame = table.merge(native_flag, on=key, validate="one_to_one")
        for values in matrices:
            part = values[values["evaluation_protocol"].eq(name)].drop(columns="evaluation_protocol")
            frame = frame.merge(part, on=key, validate="one_to_one")
        model_tables[name] = frame
    selection = pd.read_csv(factor_dir / "selection_T.csv")
    t_k = {row.evaluation_protocol: json.loads(row.selected_set) for row in selection.itertuples()}
    context.update(catalog=catalog, MODEL_TABLES=model_tables, T_K=t_k, NATIVE_IDS=native_ids,
                   screening=pd.read_csv(factor_dir / "screening_results.csv"),
                   curves=pd.read_csv(factor_dir / "validation_curves_T.csv"),
                   model_columns=lambda selected: list(selected) + (["native_missing"] if native_ids & set(selected) else []))
    if include_openfe:
        features = pd.read_csv(openfe_dir / "openfe_features.csv")
        formulas = features.set_index("candidate_id")["formula"]
        values = pd.read_parquet(openfe_dir / "openfe_values.parquet")
        selection_o = pd.read_csv(openfe_selection_dir / "selection_O.csv") if require_openfe_selection else None

        def o_model_columns(selected):
            native = {"native_last", "native_diff", "native_slope_1h", "native_curv_1h"}
            return list(selected) + (["native_missing"] if any(native & set(formula_columns(formulas[c])) for c in selected) else [])

        context.update(O_TABLES={name: table.merge(values[values["evaluation_protocol"].eq(name)].drop(columns="evaluation_protocol"),
                                                  on=key, validate="one_to_one") for name, table in model_tables.items()},
                       O_K={row.evaluation_protocol: json.loads(row.selected_set) for row in selection_o.itertuples()} if selection_o is not None else {},
                       o_model_columns=o_model_columns, openfe_features=features, FORMULA_OF=formulas)
    return context
