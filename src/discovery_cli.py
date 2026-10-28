"""Command-line entry points for factor stability, grouped cross-validation and native-anchor sensitivity.

Use --help for commands. Historical saved settings keep their original schema;
closed execution permissions are never reopened automatically."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import factor_data as data

SETTINGS_PATH = data.FACTOR_DIR / "discovery_settings.json"


def settings() -> dict:
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


MEMBERSHIP_CSV = data.FACTOR_DIR / "grouped_cv_fold_membership.csv"
MEMBERSHIP_JSON = data.FACTOR_DIR / "grouped_cv_fold_membership.json"


def cmd_cv_folds(save: bool = False, ns=None, output_dir=None) -> dict:
    """Derive membership from admitted subjects; exports never change historical settings."""
    import hashlib

    from src import grouped_cv as G

    if save:
        if output_dir is None:
            raise ValueError("saving membership requires a new output directory under reproductions/")
        output_dir = Path(output_dir).resolve()
        if output_dir.exists() or not output_dir.is_relative_to((data.PROJECT_ROOT / "reproductions").resolve()):
            raise ValueError("membership output must be a new directory under reproductions/")
    ns = ns or data.load_context(include_openfe=False)
    table, setup, info = G.build_inputs(ns)
    folds, record, frame = G.membership_from_table(table, 6)
    record["subjects"] = sorted(int(s) for s in table["subject"].unique())
    assert sorted(s for v in folds.values() for s in v) == record["subjects"] and all(len(v) == 6 for v in folds.values())
    frame_text = frame.to_csv(index=False, lineterminator="\n")
    record["membership_csv"] = MEMBERSHIP_CSV.name
    record["membership_csv_sha256"] = hashlib.sha256(frame_text.encode("utf-8")).hexdigest()
    record["seed_note"] = "there is no random seed in this assignment; seed 42 is used for the bootstrap, the E4 subsampling and other steps with randomness, not for the folds"
    if save:
        output_dir.mkdir(parents=True)
        (output_dir / MEMBERSHIP_CSV.name).write_bytes(frame_text.encode("utf-8"))
        (output_dir / MEMBERSHIP_JSON.name).write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record




def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["cv-folds", "cv", "e4", "e8", "report", "search-figures"])
    parser.add_argument("--output-dir", type=Path, help="new output directory under reproductions/")
    parser.add_argument("--save", action="store_true", help="cv-folds: write a new fold-membership snapshot")
    parser.add_argument("--dry-run", action="store_true", help="run a small development-data configuration in the explicit new output directory")
    args = parser.parse_args(argv)
    if args.command in ("cv", "e4", "e8"):
        from src import discovery_experiments as experiments

        if not args.dry_run and not settings().get("execution_enabled"):
            print("Execution is disabled; no calculation started.")
            return 2
        if args.output_dir is None:
            parser.error("calculation requires --output-dir under reproductions/")
        out_dir = args.output_dir.resolve()
        if out_dir.exists() or not out_dir.is_relative_to((data.PROJECT_ROOT / "reproductions").resolve()):
            parser.error("output must be a new directory under reproductions/")
        out_dir.mkdir(parents=True)
        ns = data.load_context()
        if args.command == "e4":
            result = experiments.compute_stability(ns, out_dir, n_pairs=2 if args.dry_run else 50, loso=not args.dry_run)
        elif args.command == "cv":
            result = experiments.compute_grouped_cv(ns, out_dir, folds=[0] if args.dry_run else None, n_resamples=200 if args.dry_run else 2000)
        else:
            subset = None
            if args.dry_run:
                full = ns["O_TABLES"]
                b_train = sorted(full[data.PROTOCOLS["B"]].query("split == 'train'")["subject"].unique())[:9]
                b_valid = sorted(full[data.PROTOCOLS["B"]].query("split == 'validation'")["subject"].unique())[:3]
                subset = {"A": sorted(full[data.PROTOCOLS["A"]]["subject"].unique())[:8], "B": b_train + b_valid}
            result = experiments.compute_anchor_sensitivity(ns, out_dir, subset, n_resamples=200 if args.dry_run else 2000)
        print(json.dumps(result, indent=1, default=str)[:3000])
        return 0
    if args.command in ("report", "search-figures"):
        from src import discovery_experiments as experiments

        if args.output_dir is None:
            parser.error("report requires --output-dir under reproductions/")
        out = args.output_dir.resolve()
        if not out.is_relative_to((data.PROJECT_ROOT / "reproductions").resolve()) or out.exists():
            parser.error("report output must be a new directory under reproductions/")
        if args.command == "search-figures":
            experiments.plot_search_results(data.FACTOR_DIR, out)
        else:
            out.mkdir(parents=True)
            experiments.make_figures(data.FACTOR_DIR, out)
        print("Saved discovery figures updated; historical authorization records are unchanged.")
        return 0
    if args.command == "cv-folds":
        print(json.dumps(cmd_cv_folds(args.save, output_dir=args.output_dir), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
