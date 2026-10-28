"""Factor construction: inferred native-reading anchors of the Libre series.

The published Libre series is a 1-minute series obtained by linear interpolation between native readings taken every
15 minutes (original paper, p. 4). Inside a window that ends at the origin t, every interpolated minute after the last native
reading already contains the next native reading, which lies after t. The positions below are therefore *inferred* native-reading
anchors (a statistical inference from the second difference, not a recorded flag), and a feature at t may only use the values
on anchors at or before t; the interpolated minutes in (last anchor, t] must never be used.

Two ways of estimating the 15-minute phase are implemented:
  retrospective   one phase per continuous run, estimated from the whole run (an offline inference);
  prefix_causal   the phase at t is estimated from the run start up to t only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["ANCHOR_PERIOD", "ANCHOR_LEVELS", "PHASE_MODES", "AnchorRule", "split_libre_runs", "libre_kinks", "native_libre_anchors",
           "Factor", "factor", "ts_mean", "ts_max", "ts_coverage", "causal_convolve", "gamma_weights", "conv_gamma", "exp_decay",
           "time_since_event", "last_event_value", "ratio", "interact", "macronutrient_ratios", "native_state", "meal_delay", "meal_series",
           "meal_primitives", "activity_primitives", "native_primitives", "subject_primitives", "metabolic_covariates",
           "CATALOG_COLUMNS", "FAMILIES", "absorption_id", "batch1_catalog", "batch2_catalog", "conditional_lab_catalog", "CandidateContext",
           "generate_candidates", "kernel_selection", "diet_features", "breakfast_peak_response"]

ANCHOR_PERIOD = 15
ANCHOR_LEVELS = ("confirmed", "grid", "uncertain")
PHASE_MODES = ("retrospective", "prefix_causal")


@dataclass(frozen=True)
class AnchorRule:
    """Native-anchor inference parameters."""
    period: int = ANCHOR_PERIOD          # native sampling interval in minutes
    kink_tolerance: float = 1e-6         # |second difference| above this marks a kink (mg/dL per minute^2)
    min_kinks: int = 4                   # kinks needed to estimate a phase
    support_kinks: int = 8               # prefix_causal: the phase uses the most recent kinks only
    min_phase_share: float = 0.9         # share of the supporting kinks that must lie on the modal phase
    max_stale_minutes: int = 60          # prefix_causal: the newest supporting kink must be this recent
    n_anchors: int = 8                   # anchors returned per origin (8 x 15 min = 2 hours)


def split_libre_runs(frame: pd.DataFrame) -> list:
    """Row-index arrays of the continuous 1-minute runs of one subject.

    `frame` is sorted by timestamp and has `timestamp`, `libre_gl`, `source_segment_id` and `is_inserted_grid_row`.
    A run ends at a segment change, at a minute step other than 1, or at a row without a published Libre value.
    """
    step_ok = frame["timestamp"].diff().eq(pd.Timedelta(minutes=1)).to_numpy()
    published = (~frame["is_inserted_grid_row"].to_numpy(bool)) & np.isfinite(frame["libre_gl"].to_numpy(float))
    same_segment = frame["source_segment_id"].eq(frame["source_segment_id"].shift()).fillna(False).to_numpy(bool)
    new_run = ~(step_ok & same_segment & published & np.roll(published, 1))
    new_run[0] = True
    run_id = np.cumsum(new_run)
    return [idx for idx in (np.flatnonzero((run_id == r) & published) for r in np.unique(run_id)) if len(idx)]


def libre_kinks(values: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Kink at minute k: nonzero second difference x[k+1] - 2 x[k] + x[k-1]. Needs x[k+1], so it is observable from k+1 on."""
    x = np.asarray(values, float)
    kink = np.zeros(len(x), bool)
    if len(x) >= 3:
        kink[1:-1] = np.abs(x[2:] - 2 * x[1:-1] + x[:-2]) > tolerance
    return kink


def _modal_phase(positions: np.ndarray, period: int) -> tuple:
    counts = np.bincount(positions % period, minlength=period)
    phase = int(counts.argmax())
    return phase, counts[phase] / len(positions)


def native_libre_anchors(values: np.ndarray, *, phase_mode: str, rule: AnchorRule = AnchorRule()) -> pd.DataFrame:
    """Inferred native-reading anchors of one continuous run, evaluated at every origin t = 0 .. n-1.

    `values` is the finite 1-minute Libre series of a single run (minute 0 is the first row of the run).
    For each origin t the table holds
      estimated sampling phase at t (-1: none could be estimated),
      phase_reliable         enough supporting kinks, on the modal phase, and (prefix_causal) recent enough,
      level                  confirmed | grid | uncertain,
      last_anchor_index      newest anchor on the phase grid at or before t (-1: none),
      anchor_age             t - last_anchor_index in minutes (0 .. period-1),
      anchor_value_0 .. _7   Libre values on the newest anchor and the ones before it (NaN before the run start).
    Only x[0..t] enters the anchor values and the level, and prefix_causal additionally ignores every kink newer than t - period
    (a kink reads the following minute), so none of its outputs depends on a reading after t. `level` is `uncertain` unless the
    phase is reliable and two anchors exist; otherwise `confirmed` when the kink at the second-newest anchor is present, else
    `grid` (collinear neighbors). The newest anchor itself cannot be checked at t: its kink reads x[anchor+1], which already
    points at the next, future reading. A reading missing from the 15-minute schedule cannot be told apart from a collinear one;
    both are treated as `grid`.

    History: the anchor values read by the features lie at most 119 minutes back, but in `prefix_causal` mode the phase, and with it the
    level and the anchor positions, is calibrated on the newest `support_kinks` kinks of the continuous run at or before t - period
    (the run start bounds the history, no 120-minute cap applies). Everything stays at or before t.
    """
    if phase_mode not in PHASE_MODES:
        raise ValueError(f"phase_mode must be one of {PHASE_MODES}")
    x = np.asarray(values, float)
    assert x.ndim == 1 and np.isfinite(x).all(), "a continuous finite run is required"
    n, period = len(x), rule.period
    kink = libre_kinks(x, rule.kink_tolerance)
    kpos = np.flatnonzero(kink)
    t = np.arange(n)

    phase = np.full(n, -1)
    reliable = np.zeros(n, bool)
    if phase_mode == "retrospective":
        if len(kpos):
            ph, share = _modal_phase(kpos, period)
            phase[:] = ph
            reliable[:] = len(kpos) >= rule.min_kinks and share >= rule.min_phase_share
    else:
        state_phase = np.full(len(kpos), -1)
        state_ok = np.zeros(len(kpos), bool)
        for i in range(len(kpos)):
            recent = kpos[max(0, i - rule.support_kinks + 1): i + 1]
            ph, share = _modal_phase(recent, period)
            state_phase[i] = ph
            state_ok[i] = len(recent) >= rule.min_kinks and share >= rule.min_phase_share
        # A kink at k reads x[k+1], which already points at the next native reading. It is independent of readings after t
        # only when that reading lies at or before t, i.e. k <= t - period; newer kinks are not used.
        last = np.searchsorted(kpos, t - period, side="right") - 1
        has = last >= 0
        li = np.where(has, last, 0)
        phase = np.where(has, state_phase[li] if len(kpos) else -1, -1)
        fresh = has & (t - (kpos[li] if len(kpos) else 0) <= rule.max_stale_minutes)
        reliable = has & (state_ok[li] if len(kpos) else False) & fresh

    has_phase = phase >= 0
    g0 = np.where(has_phase, t - ((t - phase) % period), -1)
    g0 = np.where(g0 >= 0, g0, -1)
    out = {"phase": phase, "phase_reliable": reliable, "last_anchor_index": g0,
           "anchor_age": np.where(g0 >= 0, t - g0, np.nan)}
    for j in range(rule.n_anchors):
        gj = g0 - j * period
        out[f"anchor_value_{j}"] = np.where((g0 >= 0) & (gj >= 0), x[np.clip(gj, 0, n - 1)], np.nan)
    g1 = g0 - period                                                     # newest anchor whose kink is already determined by x[0..t]
    checkable = reliable & (g0 >= 0) & (g1 >= 1)
    level = np.where(checkable, np.where(kink[np.clip(g1, 0, n - 1)], "confirmed", "grid"), "uncertain")
    out["level"] = level
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------------------------------
# Causal operators. Every operator works on one subject's contiguous 1-minute series (NaN = unobserved) and returns a Factor whose
# value at minute t uses only minutes in [t + min_source_offset, t + max_source_offset].
# ---------------------------------------------------------------------------------------------------------------------

HISTORY_MINUTES = 120
MEAL_DELAY_MINUTES = 30
MEAL_CLASSES = {"Breakfast": "breakfast", "Lunch": "lunch", "Dinner": "dinner", "Snack": "snack"}
FREE_CHOICE_MEALS = ("dinner", "snack")
MEAL_NUTRIENTS = {"carbs": "carbs", "protein": "protein", "fat": "fat", "fiber": "fiber", "calories": "meal_calories"}
MEAL_TIMING_VERSIONS = ("logged", "delay_all", "delay_free")
ACTIVITY_WINDOWS = (15, 30, 60, 120)
MET_LAG_MINUTES = 10
MET_WINDOWS = (30, 110)         # a window ends at t - MET_LAG_MINUTES and spans window + 1 minutes, so the longest starts at t - 120


@dataclass(frozen=True)
class Factor:
    name: str
    formula: str
    family: str
    min_source_offset: int       # earliest source minute relative to t
    max_source_offset: int       # latest source minute relative to t
    values: np.ndarray


def factor(name: str, formula: str, family: str, min_source_offset: int, max_source_offset: int, values: np.ndarray) -> Factor:
    """The single place where the causality rule is enforced: a factor may not read any minute after t."""
    if max_source_offset > 0:
        raise ValueError(f"{name} reads minutes after t (max_source_offset={max_source_offset})")
    return Factor(name, formula, family, int(min_source_offset), int(max_source_offset), np.asarray(values))


def _shift(x: np.ndarray, lag: int) -> np.ndarray:
    if lag == 0:
        return np.asarray(x, float)
    y = np.full(len(x), np.nan)
    y[lag:] = x[:-lag]
    return y


def _prefix(x: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(x)]


def _trailing_counts(x: np.ndarray, window: int, lag: int) -> tuple:
    """Sum and count of the observed minutes in [t - lag - window, t - lag]."""
    y = _shift(x, lag)
    observed = ~np.isnan(y)
    csum, ccount = _prefix(np.where(observed, y, 0.0)), _prefix(observed)
    t = np.arange(len(y))
    lo = np.maximum(t - window, 0)
    return csum[t + 1] - csum[lo], ccount[t + 1] - ccount[lo]


def ts_mean(x: np.ndarray, window: int, lag: int = 0, name: str = "x") -> Factor:
    s, n = _trailing_counts(x, window, lag)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(n > 0, s / n, np.nan)
    return factor(f"ts_mean({name},{window},lag={lag})", "mean of the observed minutes in [t-lag-window, t-lag]", "activity", -(lag + window), -lag, values)


def ts_max(x: np.ndarray, window: int, lag: int = 0, name: str = "x") -> Factor:
    values = pd.Series(_shift(x, lag)).rolling(window + 1, min_periods=1).max().to_numpy()
    return factor(f"ts_max({name},{window},lag={lag})", "maximum of the observed minutes in [t-lag-window, t-lag]", "activity", -(lag + window), -lag, values)


def ts_coverage(x: np.ndarray, window: int, lag: int = 0, name: str = "x") -> Factor:
    _, n = _trailing_counts(x, window, lag)
    return factor(f"ts_coverage({name},{window},lag={lag})", "observed share of the window+1 minutes in [t-lag-window, t-lag]", "activity", -(lag + window), -lag, n / (window + 1))


def causal_convolve(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """y[t] = sum over ages a of weights[a] * x[t - a], zero before the start of the series."""
    return np.convolve(np.asarray(x, float), weights)[:len(x)]


def gamma_weights(tau: float, k: int, support: int = 120) -> np.ndarray:
    """Unit-sum gamma weights over ages 0..support minutes; tau is the time to peak, so the scale is tau / (k - 1)."""
    ages = np.arange(support + 1, dtype=float)
    theta = tau / (k - 1)
    shape = ages ** (k - 1) * np.exp(-ages / theta)
    return shape / shape.sum()


def conv_gamma(pulses: np.ndarray, tau: float, k: int, delay: int = 0, name: str = "x", support: int = 120) -> Factor:
    """Meal pulses convolved with the gamma kernel. A delay keeps the log time as the event time and drops events younger than `delay`."""
    weights = gamma_weights(tau, k, support)
    weights[:delay] = 0.0
    return factor(f"conv_gamma({name},tau={tau},k={k},delay={delay})", "pulses weighted by the unit-sum gamma kernel (time to peak tau, shape k) over ages delay..120 minutes",
                  "meal_absorption", -support, -delay, causal_convolve(pulses, weights))


def exp_decay(x: np.ndarray, tau: float, name: str = "x", support: int = 120) -> Factor:
    """Exponentially weighted mean of the observed minutes over ages 0..support.

    Weights are exp(-age / tau) and are renormalized over the observed minutes only, so they sum to 1 there; a missing minute is never read
    as zero, and the value is NaN when no minute in the window is observed (the same windows in which the 120-minute coverage is 0). This is
    a different convention from `gamma_weights`, whose weights sum to 1 over all ages 0..support whatever was observed.
    """
    weights = np.exp(-np.arange(support + 1) / tau)
    weights /= weights.sum()
    observed = ~np.isnan(x)
    numerator = causal_convolve(np.where(observed, x, 0.0), weights)
    denominator = causal_convolve(observed.astype(float), weights)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(denominator > 1e-12, numerator / denominator, np.nan)
    return factor(f"exp_decay({name},tau={tau})", "exponentially weighted mean of the observed minutes over the last 120 minutes", "activity", -support, 0, values)


def _last_event_position(events: np.ndarray, delay: int) -> np.ndarray:
    """Position of the latest event at or before t - delay (-1: none)."""
    last = np.maximum.accumulate(np.where(np.asarray(events) > 0, np.arange(len(events)), -1))
    return np.r_[np.full(delay, -1), last[:len(last) - delay]] if delay else last


def time_since_event(events: np.ndarray, delay: int = 0, history: int = 120, name: str = "event") -> Factor:
    """Minutes since the latest event that is available at t (log time <= t - delay); NaN when none lies within `history` minutes."""
    last = _last_event_position(events, delay)
    age = np.arange(len(events)) - last
    return factor(f"time_since_event({name},delay={delay})", "minutes since the latest available event", "meal_event", -history, -delay,
                  np.where((last >= 0) & (age <= history), age, np.nan))


def last_event_value(events: np.ndarray, values: np.ndarray, delay: int = 0, history: int = 120, name: str = "event") -> Factor:
    """Value attached to the latest available event within `history` minutes (NaN otherwise)."""
    last = _last_event_position(events, delay)
    age = np.arange(len(events)) - last
    picked = np.asarray(values, float)[np.maximum(last, 0)]
    return factor(f"last_event_value({name},delay={delay})", "attribute of the latest available event", "meal_event", -history, -delay,
                  np.where((last >= 0) & (age <= history), picked, np.nan))


def ratio(numerator: np.ndarray, denominator: np.ndarray, name: str = "ratio", min_source_offset: int = -120, max_source_offset: int = 0) -> Factor:
    """Elementwise ratio of two causal inputs; NaN where the denominator is not positive."""
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(denominator > 0, numerator / denominator, np.nan)
    return factor(name, "numerator / denominator", "nutrient_ratio", min_source_offset, max_source_offset, values)


def interact(a: np.ndarray, b: np.ndarray, name: str = "interaction", min_source_offset: int = -120, max_source_offset: int = 0) -> Factor:
    """First-order interaction (product) of two causal inputs; NaN propagates."""
    return factor(name, "a * b", "interaction", min_source_offset, max_source_offset, np.asarray(a, float) * np.asarray(b, float))


def macronutrient_ratios(carbs: np.ndarray, protein: np.ndarray, fat: np.ndarray, fiber: np.ndarray, suffix: str = "") -> dict:
    """The three nutrient-ratio operators of the candidate grid on meal-level or kernel-weighted nutrient inputs."""
    return {"carb_share": ratio(carbs, carbs + protein + fat, f"carbs/(carbs+protein+fat){suffix}"),
            "fiber_to_carbs": ratio(fiber, carbs, f"fiber/carbs{suffix}"),
            "protein_x_carbs": interact(protein, carbs, f"protein*carbs{suffix}")}


# ---- inferred native-reading state ---------------------------------------------------------------------------------------

_ANCHOR_TIMES = -ANCHOR_PERIOD * np.arange(5)                   # newest five anchors, minutes relative to the newest
_LINEAR = np.linalg.pinv(np.c_[np.ones(5), _ANCHOR_TIMES])[1]
_QUADRATIC = np.linalg.pinv(np.c_[np.ones(5), _ANCHOR_TIMES, _ANCHOR_TIMES ** 2])[2]


def native_state(anchors: pd.DataFrame, n_anchors: int = 8) -> dict:
    """Native-reading state at the origins of an anchor table: masked anchor values, missing indicator and the four state operators.

    `uncertain` windows and operators that lack the anchors they need are NaN; the indicator is 1 for `uncertain` windows or fewer than five
    anchors. Row-independent arithmetic keeps the values identical whatever the number of rows computed together.

    The declared source offsets describe the history of the *values* only: the newest anchor is at most 14 minutes back, the eight anchors
    reach 119 minutes back and the one-hour operators 74. They do not cover the calibration history of the phase that decides which minutes
    are anchors, the level and the missing indicator: in `prefix_causal` mode the phase is estimated from the newest supporting kinks of the
    continuous run up to t - 15 minutes (eight kinks, typically 130-160 minutes back, in rare cases several hours), so those columns also
    depend on Libre minutes older than t - 120. The phase is inferred from the kink structure of the glucose series; it is not an observed
    sampling-time label.
    """
    uncertain = (anchors["level"] == "uncertain").to_numpy()
    values = anchors[[f"anchor_value_{j}" for j in range(n_anchors)]].to_numpy(float).copy()
    values[uncertain] = np.nan
    five = values[:, :5]
    reach = ANCHOR_PERIOD - 1
    one_hour = -(reach + 4 * ANCHOR_PERIOD)
    out = {"native_missing": factor("native_missing", "level is uncertain, or fewer than five anchors", "native_state", one_hour, 0,
                                    (uncertain | np.isnan(five).any(axis=1)).astype(np.int8)),
           "native_last": factor("native_last", "newest anchor value", "native_state", -reach, 0, values[:, 0]),
           "native_diff": factor("native_diff", "newest minus second-newest anchor", "native_state", -(reach + ANCHOR_PERIOD), 0, values[:, 0] - values[:, 1]),
           "native_slope_1h": factor("native_slope_1h", "least-squares slope over the newest five anchors (mg/dL per minute)", "native_state", one_hour, 0, (five * _LINEAR).sum(axis=1)),
           "native_curv_1h": factor("native_curv_1h", "quadratic coefficient over the newest five anchors", "native_state", one_hour, 0, (five * _QUADRATIC).sum(axis=1))}
    for j in range(n_anchors):
        out[f"native_anchor_{j}"] = factor(f"native_anchor_{j}", f"value on the anchor {j} readings back", "native_state", -(reach + j * ANCHOR_PERIOD), 0, values[:, j])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Data interfaces: window-level primitives from one subject's minutes, meal events and the inferred native readings.
# ---------------------------------------------------------------------------------------------------------------------

def meal_delay(version: str, meal_class: str) -> int:
    """Availability delay in minutes of a meal class under a timing version (logged time / all meals delayed / only free-choice meals delayed)."""
    if version == "logged":
        return 0
    if version == "delay_all":
        return MEAL_DELAY_MINUTES
    if version == "delay_free":
        return MEAL_DELAY_MINUTES if meal_class in FREE_CHOICE_MEALS else 0
    raise ValueError(version)


def meal_series(events: pd.DataFrame, start: pd.Timestamp, n_minutes: int) -> dict:
    """Event indicators and nutrient values on the minute grid for every meal class and for all classes together."""
    position = ((events["timestamp"] - start) // pd.Timedelta(minutes=1)).to_numpy()
    inside = (position >= 0) & (position < n_minutes)
    events, position = events.loc[inside], position[inside]
    cls = events["meal_type_normalized"].map(MEAL_CLASSES).to_numpy()
    out = {}
    for name in (*MEAL_CLASSES.values(), "any"):
        pick = np.ones(len(events), bool) if name == "any" else cls == name
        indicator = np.zeros(n_minutes)
        indicator[position[pick]] = 1
        series = {"event": indicator}
        for nutrient, column in MEAL_NUTRIENTS.items():
            values = np.full(n_minutes, np.nan)
            values[position[pick]] = events[column].to_numpy(float)[pick]
            series[nutrient] = values
        out[name] = series
    return out


def meal_primitives(events: pd.DataFrame, start: pd.Timestamp, n_minutes: int, origin_position: np.ndarray) -> dict:
    """Minutes since and nutrients of the latest meal of each class (and minutes since the latest of any class) at the origins, per timing version."""
    series = meal_series(events, start, n_minutes)
    out = {}
    for version in MEAL_TIMING_VERSIONS:
        for name, s in series.items():
            if name == "any":
                last = np.max([_last_event_position(series[c]["event"], meal_delay(version, c)) for c in MEAL_CLASSES.values()], axis=0)
                age = np.arange(n_minutes) - last
                out[f"meal_any_{version}_minutes_since"] = np.where((last >= 0) & (age <= HISTORY_MINUTES), age, np.nan)[origin_position]
                continue
            delay = meal_delay(version, name)
            out[f"meal_{name}_{version}_minutes_since"] = time_since_event(s["event"], delay, name=name).values[origin_position]
            for nutrient in MEAL_NUTRIENTS:
                out[f"meal_{name}_{version}_{nutrient}"] = last_event_value(s["event"], s[nutrient], delay, name=name).values[origin_position]
    return out


def activity_primitives(hr: np.ndarray, calories: np.ndarray, met: np.ndarray, origin_position: np.ndarray) -> dict:
    """Trailing mean, maximum, coverage and missing flag of heart rate and activity calories; MET only with a lag of at least 10 minutes.

    The lagged MET windows are [t-40, t-10] and [t-120, t-10] (30 and 110 minutes plus both end points), inside the 120-minute history."""
    out = {}
    for short, x in (("hr", hr), ("calories", calories)):
        for window in ACTIVITY_WINDOWS:
            coverage = ts_coverage(x, window, name=short).values[origin_position]
            out[f"act_{short}_mean_{window}m"] = ts_mean(x, window, name=short).values[origin_position]
            out[f"act_{short}_max_{window}m"] = ts_max(x, window, name=short).values[origin_position]
            out[f"act_{short}_coverage_{window}m"] = coverage
            out[f"act_{short}_missing_{window}m"] = (coverage == 0).astype(np.int8)
    out["act_met_value_lag10"] = _shift(met, MET_LAG_MINUTES)[origin_position]
    for window in MET_WINDOWS:
        coverage = ts_coverage(met, window, MET_LAG_MINUTES, "met").values[origin_position]
        out[f"act_met_mean_{window}m_lag10"] = ts_mean(met, window, MET_LAG_MINUTES, "met").values[origin_position]
        out[f"act_met_coverage_{window}m_lag10"] = coverage
        out[f"act_met_missing_{window}m_lag10"] = (coverage == 0).astype(np.int8)
    return out


def native_primitives(minutes: pd.DataFrame, origin_position: np.ndarray, rule: AnchorRule = AnchorRule()) -> dict:
    """Prefix-causal inferred native-reading state at the origins; minutes outside any continuous Libre run are `uncertain`.

    The anchor values come from the last 119 minutes; the phase behind them is calibrated on a longer causal history of the run (see
    `native_libre_anchors`)."""
    n = len(minutes)
    columns = ["phase", "phase_reliable", "last_anchor_index", "anchor_age", *[f"anchor_value_{j}" for j in range(rule.n_anchors)]]
    table = pd.DataFrame({c: np.full(n, np.nan) for c in columns})
    table["phase"], table["level"] = -1, "uncertain"
    values = minutes["libre_gl"].to_numpy(float)
    for run in split_libre_runs(minutes):
        rows = native_libre_anchors(values[run], phase_mode="prefix_causal", rule=rule)
        for column in ("phase", "anchor_age", "level", *[f"anchor_value_{j}" for j in range(rule.n_anchors)]):
            table.iloc[run, table.columns.get_loc(column)] = rows[column].to_numpy()
    chosen = table.iloc[origin_position].reset_index(drop=True)
    state = native_state(chosen, rule.n_anchors)
    out = {"native_level": chosen["level"].to_numpy(), "native_phase": chosen["phase"].to_numpy(int), "native_age": chosen["anchor_age"].to_numpy(float)}
    out.update({name: f.values for name, f in state.items()})
    return out


def subject_primitives(minutes: pd.DataFrame, events: pd.DataFrame, origins: pd.Series, rule: AnchorRule = AnchorRule()) -> pd.DataFrame:
    """All window-level primitives of one subject. `minutes` is the subject's contiguous 1-minute frame (Libre, heart rate, activity
    calories, MET, publication flag, segment id); `origins` are the window origin times. Only minutes and meal events at or before each origin
    reach a value."""
    minutes = minutes.reset_index(drop=True)
    assert minutes["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=1)).all()
    start, n = minutes["timestamp"].iloc[0], len(minutes)
    position = ((origins - start) // pd.Timedelta(minutes=1)).to_numpy()
    published = minutes["is_published_source_row"].to_numpy(bool)
    hr, calories, met = (np.where(published, minutes[c].to_numpy(float), np.nan) for c in ("heart_rate_bpm", "activity_calories_per_minute", "activity_met"))
    out = {}
    out.update(meal_primitives(events, start, n, position))
    out.update(activity_primitives(hr, calories, met, position))
    out.update(native_primitives(minutes, position, rule))
    return pd.DataFrame(out, index=origins.index)


def metabolic_covariates(static: pd.DataFrame) -> pd.DataFrame:
    """Baseline metabolic covariates per subject. BMI belongs to the main interaction family; HbA1c (percent), fasting glucose, fasting insulin
    and HOMA-IR = glucose [mg/dL] x insulin [uU/mL] / 405 are for offline sensitivity analysis only."""
    out = pd.DataFrame({"subject": static["subject"], "bmi_kg_m2": static["bio__bmi"], "hba1c_percent": static["bio__hba1c_lab_raw"],
                        "fasting_glucose_mg_dl": static["bio__fasting_glucose_lab_mg_dl"], "fasting_insulin_uU_ml": static["bio__fasting_insulin_raw"]})
    out["homa_ir"] = out["fasting_glucose_mg_dl"] * out["fasting_insulin_uU_ml"] / 405
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Candidate catalog and generation (`candidate_families`). A catalog row is a recipe; `CandidateContext` turns recipes into
# window-level values for one subject, so the same recipes can be recomputed from the raw minutes and events by any later notebook.
# ---------------------------------------------------------------------------------------------------------------------

ABSORPTION_TAUS = (20, 30, 45, 60, 90)
ABSORPTION_KS = (2, 3)
ABSORPTION_NUTRIENTS = ("carbs", "protein", "fat", "fiber")
ACTIVITY_SIGNALS = ("hr", "calories")
EXP_DECAY_TAUS = (30, 60)
FAMILIES = ("meal_absorption", "meal_x_baseline_state", "nutrient_ratios", "activity_kernels", "native_glucose_state")
LAB_STATES = ("fasting_glucose", "hba1c_percent", "homa_ir")
CATALOG_COLUMNS = ["candidate_id", "evaluation_protocol", "batch", "family", "formula", "generator", "generator_params", "inputs", "parameters",
                     "min_source_offset", "max_source_offset", "uses_native_state", "availability", "causal_source", "missing_handling", "generation_status"]


def absorption_id(meal_class: str, nutrient: str, tau: int, k: int) -> str:
    return f"abs_{meal_class}_{nutrient}_t{tau}_k{k}"


def _row(candidate_id, batch, family, formula, generator, params, inputs, parameters, min_offset, availability, causal_source, missing, native=False, protocol="both",
         status="generated"):
    return {"candidate_id": candidate_id, "evaluation_protocol": protocol, "batch": batch, "family": family, "formula": formula, "generator": generator,
            "generator_params": json.dumps(params, sort_keys=True), "inputs": json.dumps(inputs), "parameters": json.dumps(parameters, sort_keys=True),
            "min_source_offset": -int(abs(min_offset)), "max_source_offset": 0, "uses_native_state": bool(native), "availability": availability,
            "causal_source": causal_source, "missing_handling": missing, "generation_status": status}


MEAL_AVAILABILITY = "logged meal row time (about the pre-meal photo); nutrients known at the log time for the designed meals, assumed for free-choice meals"
KERNEL_SOURCE = "logged meal events at ages 0..120 minutes before t"
ACTIVITY_SOURCE = "published Fitbit minutes at or before t"
NATIVE_SOURCE = "inferred native-reading anchors at or before t (prefix-causal phase, calibrated on the continuous run up to t - 15 minutes)"


def batch1_catalog() -> pd.DataFrame:
    """Candidates that need no fitted choice: the absorption kernels, the most-recent-meal ratios, the activity statistics and the native state."""
    rows = []
    for meal_class in MEAL_CLASSES.values():
        for nutrient in ABSORPTION_NUTRIENTS:
            for tau in ABSORPTION_TAUS:
                for k in ABSORPTION_KS:
                    rows.append(_row(absorption_id(meal_class, nutrient, tau, k), 1, "meal_absorption",
                                     f"conv_gamma({nutrient} pulses of {meal_class} meals, tau={tau}, k={k}); unit-sum gamma weights over ages 0..120, tau = time to peak",
                                     "conv_gamma_class", {"meal_class": meal_class, "nutrient": nutrient, "tau": tau, "k": k}, [], {"tau": tau, "k": k}, 120,
                                     MEAL_AVAILABILITY, KERNEL_SOURCE, "0 when no meal of the class lies in the history (never NaN)"))
    for name, formula in (("carb_share", "carbs/(carbs+protein+fat) of the most recent meal"), ("fiber_to_carbs", "fiber/carbs of the most recent meal"),
                          ("protein_x_carbs", "protein*carbs of the most recent meal")):
        rows.append(_row(f"ratio_last_meal_{name}", 1, "nutrient_ratios", formula, "ratio_last_meal", {"ratio": name}, [], {"aggregation": "most_recent_meal"}, 120,
                         MEAL_AVAILABILITY, "latest logged meal of any class within 120 minutes before t", "NaN when no meal in the last 120 minutes or the denominator is 0"))
    for signal in ACTIVITY_SIGNALS:
        for stat in ("mean", "max"):
            for window in ACTIVITY_WINDOWS:
                rows.append(_row(f"actstat_{signal}_{stat}_{window}m", 1, "activity_kernels", f"ts_{stat}({signal}, {window}) over the observed minutes in [t-{window}, t]",
                                 "activity_stat", {"signal": signal, "stat": stat, "window": window}, [], {"window": window}, window, "wearable minutes, published rows only",
                                 ACTIVITY_SOURCE, "NaN when no observed minute in the window"))
        for tau in EXP_DECAY_TAUS:
            rows.append(_row(f"actdecay_{signal}_t{tau}", 1, "activity_kernels", f"exp_decay({signal}, tau={tau}); weights renormalized over the observed minutes of the last 120",
                             "activity_expdecay", {"signal": signal, "tau": tau}, [], {"tau": tau}, 120, "wearable minutes, published rows only", ACTIVITY_SOURCE,
                             "NaN when no observed minute in the last 120 minutes"))
    for name, formula in (("native_last", "newest inferred native-reading anchor value"), ("native_diff", "newest minus second-newest anchor"),
                          ("native_slope_1h", "least-squares slope over the newest five anchors"), ("native_curv_1h", "quadratic coefficient over the newest five anchors")):
        rows.append(_row(name, 1, "native_glucose_state", formula, "native_state", {"name": name}, [], {"max_readings": 8}, {"native_last": 14, "native_diff": 29}.get(name, 74),
                         "inferred native-reading anchors (not observed sampling labels)", NATIVE_SOURCE, "NaN plus the native_missing indicator when the anchors are uncertain", native=True))
    return pd.DataFrame(rows, columns=CATALOG_COLUMNS)


def batch2_catalog(tau_star: dict, protocol: str) -> pd.DataFrame:
    """Candidates built on the training-selected kernels. `tau_star[(nutrient, meal_class)] = (tau, k)`."""
    def kernel(nutrient, meal_class):
        tau, k = tau_star[(nutrient, meal_class)]
        return absorption_id(meal_class, nutrient, tau, k), tau, k

    totals = {n: [{"meal_class": c, "tau": tau_star[(n, c)][0], "k": tau_star[(n, c)][1]} for c in MEAL_CLASSES.values()] for n in ABSORPTION_NUTRIENTS}
    rows = []
    for name, formula in (("carb_share", "carbs/(carbs+protein+fat)"), ("fiber_to_carbs", "fiber/carbs"), ("protein_x_carbs", "protein*carbs")):
        rows.append(_row(f"ratio_kernel_{name}", 2, "nutrient_ratios", f"{formula} of the kernel-weighted nutrient totals (sum over the four meal classes at their selected tau, k)",
                         "ratio_kernel_total", {"ratio": name, "totals": totals}, [], {"aggregation": "kernel_weighted_total"}, 120, MEAL_AVAILABILITY, KERNEL_SOURCE,
                         "NaN when the denominator total is 0", protocol=protocol))
    for meal_class in MEAL_CLASSES.values():
        base_id, tau, k = kernel("carbs", meal_class)
        for state, native in (("bmi", False), ("native_last", True), ("native_slope_1h", True)):
            rows.append(_row(f"int_{meal_class}_carbs_x_{state}", 2, "meal_x_baseline_state", f"{base_id} * {state}", "interaction",
                             {"left": base_id, "right": state, "left_kind": "candidate", "right_kind": "static" if state == "bmi" else "candidate"}, [base_id] + ([] if state == "bmi" else [state]),
                             {"tau": tau, "k": k, "state": state}, 120,
                             "BMI is an enrollment measurement; kernels as in the absorption family" if state == "bmi" else NATIVE_SOURCE, KERNEL_SOURCE if state == "bmi" else KERNEL_SOURCE + "; " + NATIVE_SOURCE,
                             "NaN where the native state is uncertain (plus the native_missing indicator)" if native else "none", native=native, protocol=protocol))
    carbs_total = totals["carbs"]
    for signal in ACTIVITY_SIGNALS:
        for tau in EXP_DECAY_TAUS:
            rows.append(_row(f"actdecay_{signal}_t{tau}_x_carbs_kernel", 2, "activity_kernels", f"exp_decay({signal}, {tau}) * kernel-weighted carbohydrate total",
                             "activity_x_meal_kernel", {"signal": signal, "tau": tau, "carbs_total": carbs_total}, [f"actdecay_{signal}_t{tau}"], {"tau": tau}, 120,
                             MEAL_AVAILABILITY, ACTIVITY_SOURCE + "; " + KERNEL_SOURCE, "NaN when the activity decay is NaN", protocol=protocol))
    return pd.DataFrame(rows, columns=CATALOG_COLUMNS)


def conditional_lab_catalog() -> pd.DataFrame:
    """The conditional laboratory interaction family: cataloged, never generated or screened in the main space (offline sensitivity analysis only)."""
    rows = []
    for meal_class in MEAL_CLASSES.values():
        for state in LAB_STATES:
            rows.append(_row(f"lab_{meal_class}_carbs_x_{state}", 2, "meal_x_lab_state_conditional", f"carbs kernel of {meal_class} meals at its selected tau, k * {state}", "interaction",
                             {"left_kind": "candidate", "right": state, "right_kind": "laboratory"}, [], {"state": state}, 120,
                             "laboratory values: availability at the prediction time not provable", "offline sensitivity analysis only", "not applicable",
                             status="cataloged only (conditional; not generated, not screened)"))
    return pd.DataFrame(rows, columns=CATALOG_COLUMNS)


class CandidateContext:
    """Values of catalog recipes at the window origins of one subject.

    `minutes` is the subject's contiguous 1-minute frame (as for `subject_primitives`), `events` its meal events, `origins` the window origin times.
    `version` is the meal timing version (logged / delay_all / delay_free); `bmi` the subject's BMI. Every value uses minutes and events at or
    before the origin; the single causality check is `factor()`.
    """

    def __init__(self, minutes: pd.DataFrame, events: pd.DataFrame, origins: pd.Series, bmi: float, version: str = "logged", rule: AnchorRule = AnchorRule()):
        minutes = minutes.reset_index(drop=True)
        self.start, self.n = minutes["timestamp"].iloc[0], len(minutes)
        self.position = ((origins - self.start) // pd.Timedelta(minutes=1)).to_numpy()
        published = minutes["is_published_source_row"].to_numpy(bool)
        self.activity = {"hr": np.where(published, minutes["heart_rate_bpm"].to_numpy(float), np.nan),
                         "calories": np.where(published, minutes["activity_calories_per_minute"].to_numpy(float), np.nan)}
        self.series = meal_series(events, self.start, self.n)
        self.minutes, self.rule, self.version, self.bmi = minutes, rule, version, bmi
        self.cache, self.offsets, self.lookup = {}, {}, {}
        self._native = None

    def native(self) -> dict:
        if self._native is None:
            self._native = native_primitives(self.minutes, self.position, self.rule)
        return self._native

    def _kernel(self, meal_class: str, nutrient: str, tau: int, k: int) -> Factor:
        pulses = np.nan_to_num(self.series[meal_class][nutrient], nan=0.0)
        return conv_gamma(pulses, tau, k, delay=meal_delay(self.version, meal_class), name=f"{meal_class}.{nutrient}")

    def _last_meal(self) -> dict:
        last = {c: _last_event_position(self.series[c]["event"], meal_delay(self.version, c)) for c in MEAL_CLASSES.values()}
        newest = np.max([last[c] for c in MEAL_CLASSES.values()], axis=0)
        age = np.arange(self.n) - newest
        valid = (newest >= 0) & (age <= HISTORY_MINUTES)
        out = {}
        for nutrient in ABSORPTION_NUTRIENTS:
            value = np.full(self.n, np.nan)
            for c in MEAL_CLASSES.values():
                mine = valid & (last[c] == newest)
                value[mine] = self.series[c][nutrient][np.maximum(last[c][mine], 0)]
            out[nutrient] = value
        return out

    def ensure(self, candidate_id: str) -> np.ndarray:
        row = self.lookup[candidate_id]
        return self.values(candidate_id, row["generator"], json.loads(row["generator_params"]), json.loads(row["inputs"]))

    def values(self, candidate_id: str, generator: str, params: dict, inputs: list) -> np.ndarray:
        """Window values of one recipe at the origins; the declared offset is returned through `self.offsets`."""
        if candidate_id in self.cache:
            return self.cache[candidate_id]
        pos = self.position
        if generator == "interaction":
            for needed in [params["left"]] + ([] if params["right"] == "bmi" else [params["right"]]):
                self.ensure(needed)
        if generator == "conv_gamma_class":
            f = self._kernel(params["meal_class"], params["nutrient"], params["tau"], params["k"])
            out = f.values[pos]
        elif generator == "ratio_last_meal":
            last = self._last_meal()
            f = macronutrient_ratios(last["carbs"], last["protein"], last["fat"], last["fiber"])[params["ratio"]]
            out = f.values[pos]
        elif generator == "activity_stat":
            x = self.activity[params["signal"]]
            f = (ts_mean if params["stat"] == "mean" else ts_max)(x, params["window"], name=params["signal"])
            out = f.values[pos]
        elif generator == "activity_expdecay":
            f = exp_decay(self.activity[params["signal"]], params["tau"], name=params["signal"])
            out = f.values[pos]
        elif generator == "native_state":
            nat = self.native()
            out = nat[params["name"]]
            f = factor(candidate_id, "native state", "native_state", -74, 0, out)
        elif generator in ("ratio_kernel_total", "activity_x_meal_kernel"):
            def total(nutrient_parts):
                return np.sum([self._kernel(p["meal_class"], p["nutrient"], p["tau"], p["k"]).values for p in nutrient_parts], axis=0)
            if generator == "ratio_kernel_total":
                t = {n: total([dict(p, nutrient=n) for p in params["totals"][n]]) for n in ABSORPTION_NUTRIENTS}
                f = macronutrient_ratios(t["carbs"], t["protein"], t["fat"], t["fiber"])[params["ratio"]]
                out = f.values[pos]
            else:
                carbs = total([dict(p, nutrient="carbs") for p in params["carbs_total"]])
                decay = exp_decay(self.activity[params["signal"]], params["tau"], name=params["signal"])
                f = interact(decay.values, carbs, candidate_id)
                out = f.values[pos]
        elif generator == "interaction":
            left = self.cache[params["left"]]
            right = self.bmi if params["right"] == "bmi" else self.cache[params["right"]]
            f = interact(left, np.full_like(left, right) if params["right"] == "bmi" else right, candidate_id)
            out = f.values
        else:
            raise ValueError(generator)
        self.cache[candidate_id] = np.asarray(out, float)
        self.offsets[candidate_id] = f.min_source_offset
        return self.cache[candidate_id]


def generate_candidates(catalog: pd.DataFrame, context: CandidateContext) -> pd.DataFrame:
    """Window values of the generated catalog rows; an interaction pulls its inputs from the same catalog (batch 1 candidates)."""
    out = {}
    context.lookup = catalog.set_index("candidate_id").to_dict("index")
    for row in catalog[catalog["generation_status"].eq("generated")].itertuples():
        out[row.candidate_id] = context.values(row.candidate_id, row.generator, json.loads(row.generator_params), json.loads(row.inputs))
    return pd.DataFrame(out)


def kernel_selection(catalog_ic: pd.DataFrame) -> dict:
    """tau*, k* per (nutrient, meal class): the (tau, k) pair with the largest |mean IC| among the 10 absorption candidates (ties: smallest tau, then k)."""
    rows = catalog_ic[catalog_ic["family"].eq("meal_absorption")].copy()
    params = rows["parameters"].map(json.loads)
    rows["tau"], rows["k"] = params.map(lambda p: p["tau"]), params.map(lambda p: p["k"])
    gp = rows["generator_params"].map(json.loads)
    rows["meal_class"], rows["nutrient"] = gp.map(lambda p: p["meal_class"]), gp.map(lambda p: p["nutrient"])
    rows = rows.sort_values(["abs_mean_ic", "tau", "k"], ascending=[False, True, True], kind="stable")
    best = rows.groupby(["nutrient", "meal_class"], sort=False).head(1)
    return {(r.nutrient, r.meal_class): (int(r.tau), int(r.k)) for r in best.itertuples()}


# ---------------------------------------------------------------------------------------------------------------------
# manual-ablation notebook diet features under a meal-class availability delay (E9) and the repeated-breakfast response (T16)
# ---------------------------------------------------------------------------------------------------------------------

DIET_WINDOWS = (30, 120)
DIET_NUTRIENT_COLUMNS = ("meal_calories", "carbs", "protein", "fat", "fiber")


def diet_features(events: pd.DataFrame, start: pd.Timestamp, n_minutes: int, origin_position: np.ndarray, version: str = "logged") -> dict:
    """The manual-ablation notebook diet features (window counts, no-meal flags, nutrient sums, minutes since and attributes of the last meal) with each meal
    class available `meal_delay(version, class)` minutes after its log time. A meal counts when its log time lies in [t - window, t - delay]."""
    position = ((events["timestamp"] - start) // pd.Timedelta(minutes=1)).to_numpy()
    inside = (position >= 0) & (position < n_minutes)
    events, position = events.loc[inside], position[inside]
    cls = events["meal_type_normalized"].map(MEAL_CLASSES).to_numpy()
    o = origin_position
    parts = {}
    for name in MEAL_CLASSES.values():
        pick = cls == name
        event = np.zeros(n_minutes)
        event[position[pick]] = 1
        values, nonnull = {}, {}
        for nutrient in DIET_NUTRIENT_COLUMNS:
            v = np.zeros(n_minutes)
            v[position[pick]] = events[nutrient].fillna(0).to_numpy(float)[pick]
            known = np.zeros(n_minutes)
            known[position[pick]] = events[nutrient].notna().to_numpy(float)[pick]
            values[nutrient], nonnull[nutrient] = _prefix(v), _prefix(known)
        parts[name] = {"event": _prefix(event), "last": np.maximum.accumulate(np.where(event > 0, np.arange(n_minutes), -1)), "values": values, "nonnull": nonnull,
                       "delay": meal_delay(version, name), "carbs": np.zeros(n_minutes), "calories": np.zeros(n_minutes)}
        row_carbs, row_cal = np.full(n_minutes, np.nan), np.full(n_minutes, np.nan)
        row_carbs[position[pick]] = events["carbs"].to_numpy(float)[pick]
        row_cal[position[pick]] = events["meal_calories"].to_numpy(float)[pick]
        parts[name]["carbs"], parts[name]["calories"] = row_carbs, row_cal
    out = {}
    for window in DIET_WINDOWS:
        count = np.zeros(len(o))
        sums = {nutrient: np.zeros(len(o)) for nutrient in DIET_NUTRIENT_COLUMNS}
        known = {nutrient: np.zeros(len(o)) for nutrient in DIET_NUTRIENT_COLUMNS}
        for p in parts.values():
            hi, lo = o - p["delay"] + 1, o - window
            count += p["event"][hi] - p["event"][lo]
            for nutrient in DIET_NUTRIENT_COLUMNS:
                sums[nutrient] += p["values"][nutrient][hi] - p["values"][nutrient][lo]
                known[nutrient] += p["nonnull"][nutrient][hi] - p["nonnull"][nutrient][lo]
        out[f"diet_logged_meal_count_{window}m"] = count
        out[f"diet_no_logged_meal_{window}m"] = (count == 0).astype(np.int8)
        for nutrient in DIET_NUTRIENT_COLUMNS:
            label = "meal_calories" if nutrient == "meal_calories" else nutrient
            out[f"diet_logged_{label}_sum_{window}m"] = np.where((count > 0) & (known[nutrient] == count), sums[nutrient], np.nan)
    latest = np.max([p["last"][o - p["delay"]] for p in parts.values()], axis=0)
    valid = latest >= o - HISTORY_MINUTES
    out["diet_minutes_since_last_logged_meal"] = np.where(valid, o - latest, np.nan)
    rows = np.where(valid, latest, 0)
    carbs, calories = np.full(len(o), np.nan), np.full(len(o), np.nan)
    types = {name: np.full(len(o), np.nan) for name in parts}
    for name, p in parts.items():
        mine = valid & (p["last"][o - p["delay"]] == latest)
        carbs[mine], calories[mine] = p["carbs"][rows[mine]], p["calories"][rows[mine]]
    for name, p in parts.items():
        types[name] = np.where(valid, (p["last"][o - p["delay"]] == latest).astype(float), np.nan)
    out["diet_last_meal_carbs"], out["diet_last_meal_calories"] = carbs, calories
    for name in parts:
        out[f"diet_last_meal_type_{name}"] = types[name]
    return out


def breakfast_peak_response(glucose: np.ndarray, timestamps: pd.Series, events: pd.DataFrame, training_end: pd.Timestamp, horizon: int = 180, min_carbs: float = 0.0) -> pd.DataFrame:
    """Per-gram carbohydrate peak increment of each breakfast whose whole response window [log time, log time + horizon] lies in the training period.

    `glucose` is the subject's contiguous 1-minute Libre series (NaN where unpublished) on `timestamps`. The increment is the maximum over the window minus
    the value at the log time, divided by the carbohydrate grams of the meal; windows with a missing minute or no carbohydrate are skipped.
    """
    start = timestamps.iloc[0]
    breakfast = events[events["meal_type_normalized"].eq("Breakfast") & (events["carbs"] > min_carbs)]
    rows = []
    for event in breakfast.itertuples():
        i = int((event.timestamp - start) // pd.Timedelta(minutes=1))
        if i < 0 or i + horizon >= len(glucose) or event.timestamp + pd.Timedelta(minutes=horizon) > training_end:
            continue
        window = glucose[i:i + horizon + 1]
        if np.isnan(window).any():
            continue
        rows.append({"timestamp": event.timestamp, "carbs": float(event.carbs), "increment": float(window.max() - window[0]), "per_gram": float((window.max() - window[0]) / event.carbs)})
    return pd.DataFrame(rows, columns=["timestamp", "carbs", "increment", "per_gram"])
