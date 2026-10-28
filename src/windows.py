"""Window construction and split assignment for the forecasting task.

Builds the per-subject origin candidates with sequential structural eligibility and assigns the admitted windows
to the two evaluation protocols.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["build_windows", "assign_splits"]


def build_windows(dynamic, segments, *, task_id, target_field, origin_field, target_unit,
                  grid_delta, history_delta, horizon_delta, stride_delta):
    """Construct per-subject origin candidates and apply sequential structural eligibility.

    Returns (candidates, sample_flow, manifest, exclusion_stages): all candidate origins with their exclusion
    reason, the sequential flow table, the eligible-window manifest, and the ordered (reason, failure) stages.
    """
    expected_points = int(history_delta / grid_delta) + 1
    candidate_parts = []
    for subject, subject_rows in dynamic.groupby("subject", sort=True):
        subject_rows = subject_rows.set_index("timestamp").sort_index()
        times = subject_rows.index
        assert times.equals(pd.date_range(times.min(), times.max(), freq=grid_delta))
    
        # Anchor stride at each subject's first released timestamp; preserve phase across gaps.
        # Every candidate is assigned its inherited segment, including null segments in inserted rows.
        origins = pd.date_range(times.min(), times.max(), freq=stride_delta)
        starts = origins - history_delta
        targets = origins + horizon_delta
        origin_rows = subject_rows.reindex(origins)
        target_rows = subject_rows.reindex(targets)
        valid_values = (
            subject_rows["is_published_source_row"]
            & ~subject_rows["is_inserted_grid_row"]
            & subject_rows[target_field].notna()
            & np.isfinite(subject_rows[target_field])
            & subject_rows[origin_field].eq("published_source_value_anchor_status_unknown")
        )

        # Timestamp search locates inclusive bounds; prefix counts are completeness audits, not features.
        left = times.searchsorted(starts, side="left")
        right = times.searchsorted(origins, side="right")
        valid_prefix = np.r_[0, valid_values.to_numpy(dtype=np.int64).cumsum()]
        inserted_prefix = np.r_[0, subject_rows["is_inserted_grid_row"].to_numpy(dtype=np.int64).cumsum()]
        candidate = pd.DataFrame({
            "task_id": task_id, "subject": int(subject), "origin_time": origins,
            "history_start_time": starts, "history_end_time": origins, "target_time": targets,
            "target_field": target_field, "target_unit": target_unit,
            "target_value": target_rows[target_field].to_numpy(),
            "segment_id": pd.array(origin_rows["source_segment_id"], dtype="Int64"),
            "history_expected_points": expected_points,
            "history_grid_points": right - left,
            "history_valid_points": valid_prefix[right] - valid_prefix[left],
            "history_inserted_points": inserted_prefix[right] - inserted_prefix[left],
            "history_in_subject_bounds": starts >= times.min(),
            "target_in_subject_bounds": targets <= times.max(),
            "target_available": (
                target_rows["is_published_source_row"].eq(True)
                & target_rows["is_inserted_grid_row"].eq(False)
                & target_rows[target_field].notna()
                & np.isfinite(target_rows[target_field])
                & target_rows[origin_field].eq("published_source_value_anchor_status_unknown")
            ).to_numpy(),
            "source_file": origin_rows["source_file"].to_numpy(),
            "origin_source_row_index": pd.array(origin_rows["source_row_index"], dtype="Int64"),
            "target_source_row_index": pd.array(target_rows["source_row_index"], dtype="Int64"),
            "target_value_origin": target_rows[origin_field].to_numpy(),
        })
        segment_bounds = segments.loc[segments["subject"].eq(subject), ["subject", "segment_id", "segment_start_time", "segment_end_time"]]
        candidate = candidate.merge(segment_bounds, on=["subject", "segment_id"], how="left", validate="many_to_one")
        candidate["continuity_ok"] = (candidate["history_start_time"].ge(candidate["segment_start_time"]) & candidate["target_time"].le(candidate["segment_end_time"]))
        candidate["history_complete"] = candidate["history_valid_points"].eq(expected_points) & candidate["history_grid_points"].eq(expected_points) & candidate["history_inserted_points"].eq(0)
        candidate_parts.append(candidate)

    candidates = pd.concat(candidate_parts, ignore_index=True).sort_values(["subject", "origin_time"]).reset_index(drop=True)
    exclusion_stages = [
        ("insufficient_history_extent", ~candidates["history_in_subject_bounds"]),
        ("target_outside_subject_extent", ~candidates["target_in_subject_bounds"]),
        ("disallowed_source_discontinuity", ~candidates["continuity_ok"]),
        ("incomplete_published_primary_history", ~candidates["history_complete"]),
        ("target_unavailable", ~candidates["target_available"]),
    ]
    candidates["exclusion_reason"] = "eligible"
    remaining = pd.Series(True, index=candidates.index)
    flow_records = [{"stage": 0, "rule": "candidate_origins", "input_count": len(remaining), "excluded_count": 0, "remaining_count": len(remaining)}]
    for stage, (reason, failure) in enumerate(exclusion_stages, start=1):
        rejected = remaining & failure
        flow_records.append({"stage": stage, "rule": reason, "input_count": int(remaining.sum()), "excluded_count": int(rejected.sum()), "remaining_count": int((remaining & ~failure).sum())})
        candidates.loc[rejected, "exclusion_reason"] = reason
        remaining &= ~failure
    flow_records.append({"stage": len(flow_records), "rule": "eligible_windows", "input_count": int(remaining.sum()), "excluded_count": 0, "remaining_count": int(remaining.sum())})
    sample_flow = pd.DataFrame(flow_records)
    sample_flow.insert(0, "task_id", task_id)
    manifest = candidates.loc[remaining].drop(columns=["history_in_subject_bounds", "target_in_subject_bounds", "exclusion_reason"]).copy().reset_index(drop=True)
    return candidates, sample_flow, manifest, exclusion_stages


def assign_splits(windows, scope, *, window_columns, labels, protocol_a_name, protocol_b_name, sample_key):
    """Map each admitted window to both evaluation protocols, preserving its original identity.

    `windows` holds the structural window columns, `scope` the per-subject split scope. Returns
    (fold_a, fold_b, fold_assignments).
    """
    joined = windows.merge(scope[["subject", "start_time", "end_time",
        "protocol_a_train_end_time", "protocol_a_val_end_time", "protocol_b_group"]],
        on="subject", how="left", validate="many_to_one")
    if not (joined["history_start_time"].ge(joined["start_time"]) & joined["target_time"].le(joined["end_time"])).all():
        raise ValueError("window history and target must lie within the subject's recorded time range")

    # Inherited endpoint semantics: train [start,b1], validation (b1,b2], test (b2,end].
    temporal_membership = pd.DataFrame({
        "train": joined["history_start_time"].ge(joined["start_time"])
            & joined["target_time"].le(joined["protocol_a_train_end_time"]),
        "validation": joined["history_start_time"].gt(joined["protocol_a_train_end_time"])
            & joined["target_time"].le(joined["protocol_a_val_end_time"]),
        "test": joined["history_start_time"].gt(joined["protocol_a_val_end_time"])
            & joined["target_time"].le(joined["end_time"]),
    })
    assert temporal_membership.sum(axis=1).le(1).all()
    temporal_crosses_boundary = (
        (joined["history_start_time"].le(joined["protocol_a_train_end_time"])
         & joined["target_time"].gt(joined["protocol_a_train_end_time"]))
        | (joined["history_start_time"].le(joined["protocol_a_val_end_time"])
           & joined["target_time"].gt(joined["protocol_a_val_end_time"]))
    )
    assert (~temporal_membership.any(axis=1)).equals(temporal_crosses_boundary)
    fold_a = joined[window_columns].copy()
    fold_a["evaluation_protocol"] = protocol_a_name
    fold_a["split"] = np.select([temporal_membership[s] for s in labels], labels, default="excluded")
    fold_a["admitted_to_split"] = temporal_membership.any(axis=1)
    fold_a["exclusion_reason"] = np.where(fold_a["admitted_to_split"], "none", "temporal_boundary_crossing")
    fold_a["partition_start_time"] = pd.NaT
    fold_a["partition_end_time"] = pd.NaT
    fold_a["partition_start_inclusive"] = pd.Series(pd.NA, index=fold_a.index, dtype="boolean")
    for label, lower, upper, inclusive in [
        ("train", "start_time", "protocol_a_train_end_time", True),
        ("validation", "protocol_a_train_end_time", "protocol_a_val_end_time", False),
        ("test", "protocol_a_val_end_time", "end_time", False),
    ]:
        chosen = fold_a["split"].eq(label)
        fold_a.loc[chosen, "partition_start_time"] = joined.loc[chosen, lower]
        fold_a.loc[chosen, "partition_end_time"] = joined.loc[chosen, upper]
        fold_a.loc[chosen, "partition_start_inclusive"] = inclusive

    # The unseen-subject assignment is inherited directly; no random operation is used.
    fold_b = joined[window_columns].copy()
    fold_b["evaluation_protocol"] = protocol_b_name
    fold_b["split"] = joined["protocol_b_group"].replace({"val": "validation"})
    fold_b["admitted_to_split"] = True
    fold_b["exclusion_reason"] = "none"
    fold_b["partition_start_time"] = joined["start_time"]
    fold_b["partition_end_time"] = joined["end_time"]
    fold_b["partition_start_inclusive"] = pd.Series(True, index=fold_b.index, dtype="boolean")
    fold_assignments = pd.concat([fold_a, fold_b], ignore_index=True)
    fold_assignments = fold_assignments.sort_values(["evaluation_protocol", *sample_key]).reset_index(drop=True)
    return fold_a, fold_b, fold_assignments
