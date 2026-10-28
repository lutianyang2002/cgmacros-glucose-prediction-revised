"""Training-only stability analyses: subsampling (E4) and metabolic-group sensitivity.

E4 selector (IC ranking followed by Lasso on residuals, q = 10): the subject-level Spearman IC of every candidate with the training residual r is averaged over the
subjects of a sample, the five candidates with the largest |mean IC| of each family form the pool, and a Lasso on r over the pool, followed along its regularization path, returns the
first q = 10 candidates that enter it. Implementation details that the protocol does not fix (and which are therefore exploratory choices, not preregistered): the pool is the
per-family top K = 5 of the screening (no redundancy filter), the Lasso path has 100 penalties from alpha_max down to 1e-3 alpha_max, the variables are median-imputed and standardized
within the sample, and when several candidates enter at once the q with the largest |coefficient| are kept. The candidate library (kernel parameters tau*, k* of the interaction and ratio
candidates) was fixed on all training subjects of the protocol in theory-factor discovery and is not re-estimated inside a half sample.

Nothing here reads a validation or test window: the functions receive the thinned core training windows and the subject-level IC table of one protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, lasso_path
from sklearn.preprocessing import StandardScaler

__all__ = ["family_pool", "lasso_first_q", "select_ic_lasso", "complementary_pairs", "selection_frequency", "jaccard", "loso_jaccard", "same_sign_share", "metabolic_group", "group_stability"]

PATH_ALPHAS = 100
PATH_EPS = 1e-3


def family_pool(sub_ic: pd.DataFrame, family_of: pd.Series, k_per_family: int = 5) -> list:
    """Per family the `k_per_family` candidates with the largest |mean IC| over the subjects of `sub_ic` (ties by column order; undefined mean IC ranks last)."""
    mean = sub_ic.mean(skipna=True)
    frame = pd.DataFrame({"candidate": mean.index, "abs": mean.abs().to_numpy(), "family": family_of.reindex(mean.index).to_numpy(), "order": np.arange(len(mean))})
    frame["abs"] = frame["abs"].fillna(-1.0)
    frame = frame.sort_values(["family", "abs", "order"], ascending=[True, False, True], kind="stable")
    frame["rank"] = frame.groupby("family").cumcount() + 1
    chosen = frame[frame["rank"] <= k_per_family].sort_values(["abs", "order"], ascending=[False, True], kind="stable")
    return list(chosen["candidate"])


def lasso_first_q(x: np.ndarray, r: np.ndarray, q: int = 10) -> list:
    """Column positions of the first `q` variables that enter a Lasso path on standardized, median-imputed `x` (largest |coefficient| at the first penalty with at least q non-zero)."""
    x = StandardScaler().fit_transform(SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(x))
    r = np.asarray(r, float) - np.mean(r)
    alphas, coefs, _ = lasso_path(x, r, n_alphas=PATH_ALPHAS, eps=PATH_EPS)[:3]
    nonzero = (np.abs(coefs) > 0).sum(axis=0)                      # alphas are in decreasing order
    reached = np.flatnonzero(nonzero >= q)
    j = int(reached[0]) if len(reached) else len(alphas) - 1
    column = np.abs(coefs[:, j])
    order = np.argsort(-column, kind="stable")
    return [int(c) for c in order[:q] if column[c] > 0]


def select_ic_lasso(thinned: pd.DataFrame, ic: pd.DataFrame, subjects, family_of: pd.Series, k_per_family: int = 5, q: int = 10, target: str = "residual") -> list:
    """E4 selection on the windows of `subjects`: IC ranking per family, then the Lasso on r over the pool. Only the rows and IC entries of `subjects` are read."""
    subjects = set(subjects)
    sub_ic = ic.loc[ic.index.isin(subjects)]
    pool = family_pool(sub_ic, family_of, k_per_family)
    rows = thinned[thinned["subject"].isin(subjects)]
    picked = lasso_first_q(rows[pool].to_numpy(float), rows[target].to_numpy(float), q)
    return [pool[i] for i in picked]


def complementary_pairs(subjects, n_pairs: int = 50, seed: int = 42) -> list:
    """`n_pairs` complementary half-subject splits: each pair is a permutation of the sorted subjects cut into floor(n/2) and the rest (no subject in both halves of a pair)."""
    ids = np.array(sorted(subjects))
    pairs = []
    for p in range(n_pairs):
        order = np.random.default_rng([seed, p]).permutation(len(ids))
        half = len(ids) // 2
        pairs.append((sorted(ids[order[:half]].tolist()), sorted(ids[order[half:]].tolist())))
    return pairs


def selection_frequency(selections: list, universe, thresholds=(0.5, 0.6, 0.7)) -> tuple:
    """Share of the samples that select each candidate, and for every threshold the candidates at or above it."""
    freq = pd.Series({c: float(np.mean([c in s for s in selections])) for c in universe}).sort_values(ascending=False, kind="stable")
    return freq, {t: [c for c, f in freq.items() if f >= t] for t in thresholds}


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def loso_jaccard(select_fn, subjects, full_selection) -> pd.DataFrame:
    """Selection with each subject left out (`select_fn(remaining subjects) -> list`) and its Jaccard index with the full-sample selection."""
    rows = []
    for s in sorted(subjects):
        chosen = select_fn([x for x in subjects if x != s])
        rows.append({"left_out_subject": s, "jaccard_with_full": jaccard(full_selection, chosen), "selected": chosen})
    return pd.DataFrame(rows)


def same_sign_share(ic: pd.DataFrame, columns, subjects=None) -> pd.DataFrame:
    """Per candidate: mean subject-level IC, number of subjects with a defined IC and the share of them whose sign equals the sign of the mean."""
    part = ic if subjects is None else ic.loc[ic.index.isin(set(subjects))]
    rows = []
    for c in columns:
        v = part[c].dropna().to_numpy()
        mean = float(v.mean()) if len(v) else np.nan
        rows.append({"candidate": c, "mean_ic": mean, "subjects_with_ic": len(v), "same_sign_share": float((np.sign(v) == np.sign(mean)).mean()) if len(v) and mean != 0 else np.nan})
    return pd.DataFrame(rows)


def metabolic_group(hba1c: pd.Series) -> pd.Series:
    """Three groups by HbA1c in percent (< 5.7, 5.7 to 6.4, > 6.4), the paper's grouping; used offline for stability only, never as a predictor."""
    return pd.cut(hba1c, [-np.inf, 5.7, 6.4 + 1e-9, np.inf], labels=["normal (<5.7)", "prediabetes (5.7-6.4)", "diabetes (>6.4)"], right=False)


def group_stability(thinned: pd.DataFrame, ic: pd.DataFrame, selected: list, group_of: pd.Series, alpha: float, indicator: str | None = None, target: str = "residual") -> pd.DataFrame:
    """Sign consistency of the selected factors across metabolic groups (the metabolic-group definition: environments are three HbA1c groups of the protocol's own training subjects).

    Per group and factor: the mean subject-level IC and the standardized ridge coefficient (a ridge of r on the selected factors, plus `indicator` when given, fitted in the group alone with
    penalty `alpha`). A factor is stable when both signs agree in every group. It never reselects or removes a factor.
    Missing group statistics produce nullable flags instead of a sign disagreement.
    """
    rows = []
    columns = list(selected) + ([indicator] if indicator else [])
    for group, members in group_of.groupby(group_of, observed=False).groups.items():
        subjects = set(members)
        part = thinned[thinned["subject"].isin(subjects)]
        coef = np.full(len(columns), np.nan)
        if len(part) and selected:
            x = StandardScaler().fit_transform(SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(part[columns].to_numpy(float)))
            coef = Ridge(alpha=alpha).fit(x, part[target].to_numpy(float) - part[target].mean()).coef_
        mean_ic = ic.loc[ic.index.isin(subjects), list(selected)].mean(skipna=True)
        for i, f in enumerate(selected):
            rows.append({"group": group, "group_subjects": len(subjects), "windows": len(part), "factor": f, "mean_ic": float(mean_ic[f]), "ridge_coefficient": float(coef[i])})
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["group", "group_subjects", "windows", "factor", "mean_ic", "ridge_coefficient", "same_sign_ic", "same_sign_coefficient", "stable"])
    def same_sign(values):
        return pd.NA if not np.isfinite(values).all() else bool((np.sign(values) == np.sign(values.iloc[0])).all() and values.iloc[0] != 0)
    stable = table.groupby("factor").agg(same_sign_ic=("mean_ic", same_sign), same_sign_coefficient=("ridge_coefficient", same_sign)).astype("boolean")
    stable["stable"] = stable["same_sign_ic"] & stable["same_sign_coefficient"]
    stable["stable"] = stable["stable"].mask(stable[["same_sign_ic", "same_sign_coefficient"]].isna().any(axis=1), pd.NA)
    return table.merge(stable.reset_index(), on="factor")
