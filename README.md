# CGMacros Glucose Prediction and Factor Discovery

Availability-constrained factor discovery for **30-minute-ahead glucose forecasting** on CGMacros. The primary task uses a closed 120-minute Libre glucose history, with prediction origins every five minutes. **Protocol A** predicts future periods within participants; **Protocol B** evaluates unseen participants. The primary metric is participant-equal RMSE.

See the [research report](report/report.pdf) for methods, results, interpretation, and limitations.

## Notebook guide

Read the notebooks in numbered order. Their saved outputs are included unchanged.

| Notebook | Role |
|---|---|
| [01_data_audit.ipynb](notebooks/01_data_audit.ipynb) | Source quality, participant coverage, and split registration |
| [02_clean_align.ipynb](notebooks/02_clean_align.ipynb) | Schema cleaning, timestamp alignment, events, and feature availability |
| [03_eda_task_samples.ipynb](notebooks/03_eda_task_samples.ipynb) | TRAIN exploration, forecasting windows, splits, and glucose-history features |
| [04_baseline_models.ipynb](notebooks/04_baseline_models.ipynb) | Baseline fitting and validation comparison |
| [05_factor_ablation.ipynb](notebooks/05_factor_ablation.ipynb) | Manual features, joint models, and ablation comparisons |
| [06_evaluation_report.ipynb](notebooks/06_evaluation_report.ipynb) | Manual-model TEST metrics, paired comparisons, and error analysis |
| [07_optional_studies.ipynb](notebooks/07_optional_studies.ipynb) | Microbiome, device/horizon sensitivity, PPGR, and sequence models |
| [08_factor_discovery.ipynb](notebooks/08_factor_discovery.ipynb) | T/O/L/R factor construction, selection, and validation analyses |
| [09_factor_evaluation.ipynb](notebooks/09_factor_evaluation.ipynb) | Saved factor-evaluation results and optional validation reconstruction |

## Methods and interpretation

Baselines include persistence, a trend reference, GlucoBench-style linear regression, Sparacino AR(1)-WLS, Ridge, and shallow gradient boosting. Learned preprocessing uses TRAIN only; residual targets use independent out-of-fold fits.

The two main experiment families are distinct:

| Family | Design |
|---|---|
| **42 manual ablations** | Joint Ridge models of glucose, activity, diet, and baseline/static features, with penalties 0.1, 1, and 10; specified boosting and cohort checks are also reported |
| **50 factor comparisons** | Residual blocks added to fixed GlucoBench predictions, with 13 Ridge penalties from 0.001 to 1,000; 24 comparisons in A and 26 in B |

M denotes the 50-feature manual block. T/O/L/R denote theory-constrained, restricted OpenFE, LLM-guided DSL, and budget-matched Random-DSL factors. Factor selection uses participants with at least 20 core validation windows; common-core refits use all core validation participants before reporting filters. Manual ablation has its own cohort rules. The approximately 0.382 mg/dL Protocol A difference between the full manual models is descriptive and does not isolate an effect of residual modeling.

**Both families used the same historical TEST partitions. The factor evaluation is not independent confirmation.** LLM/Random-DSL comparisons, sequence-factor transfer, and laboratory-interaction extensions are validation-only. Validation selection and penalty tuning make these estimates optimistic. Saved intervals use 2,000 paired participant resamples on fixed predictions, excluding selection uncertainty. P1-P4 form one Holm family per protocol; bootstrap p-values are approximate and nominal family-wise control is unverified. No preregistration is claimed.

Prediction-time availability remains a methodological constraint. Mixed-source filtering provides statistical isolation, not physical isolation of decoded rows. PPGR is a partial reconstruction; GPU execution and environment differences can affect sequence-model reproduction.

## Repository contents

```text
notebooks/    Nine numbered notebooks and their saved outputs
src/          Complete current research implementation
tests/        Synthetic, interface, and saved-result regression tests
configs/      Supplementary experiment specifications
data/         External-data setup guide
results/      Selected result tables, scientific definitions, and feature metadata
figures/      Factor-selection curves and participant error differences
report/       Research report PDF
```

Results retain the paths used by the code. They are a selected public subset, not a complete execution cache.

| Results | Contents |
|---|---|
| [Baselines](results/baselines/) | Validation metrics and model specifications |
| [Manual ablation](results/manual_ablation/) / [evaluation](results/manual_evaluation/) | Validation comparisons, feature definitions, 42-comparison intervals, metrics, and selected plots |
| [Factor discovery](results/factor_discovery/) | T/O/L/R selections, candidate/formula metadata, validation comparisons and curves, search-budget summaries, and plotting inputs |
| [Factor evaluation](results/factor_evaluation/) | The 50-comparison tables, configuration metrics, meal strata, and a read-only completion summary |
| [Supplementary studies](results/supplementary_studies/) | Microbiome, device/horizon, PPGR, and sequence summaries |
| [Sequence transfer](results/sequence_transfer/) / [laboratory interactions](results/laboratory_interactions/) | Validation-only extension results |
| [Forecast specifications](results/forecast_windows/) | Task, window, split, feature, and sample-flow definitions |

The numerical tables are copied without editing. The factor TEST tables are the delivered presentation derivatives; their numerical fields are unchanged from the formal results. The completion summary is not execution permission. Historical permission records, execution-source archives, and raw search logs are not distributed here, so this bundle cannot independently reverify their historical identity. Current source is not presented as the original TEST execution snapshot.

## Data and environments

Download [CGMacros v1.0.0 from PhysioNet](https://physionet.org/content/cgmacros/1.0.0/) separately under its source license. Follow [data/README.md](data/README.md). The default root is `data/raw/cgmacros-v1.0.0/`; alternatively set `CGMACROS_DATA_DIR` to an existing dataset root containing `CGMacros/`, the data dictionaries, license, and checksum file. Raw dataset files, photos, and environments are excluded; unchanged notebook outputs retain limited previews and summaries of the published anonymized data.

Dataset citation: Gutierrez-Osuna et al. (2025), *CGMacros*, version 1.0.0, PhysioNet, [doi:10.13026/3z8q-x658](https://doi.org/10.13026/3z8q-x658).

Use Python 3.11 for the main environment. For example, on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On POSIX, create the environment with `python3.11 -m venv .venv` and activate it with `source .venv/bin/activate`. The editable installation makes `src` importable from the notebooks. Use VS Code or an existing Jupyter frontend with this environment selected.

- [requirements.txt](requirements.txt): main scientific and notebook dependencies.
- [requirements-gpu.txt](requirements-gpu.txt): recorded PyTorch/CUDA environment; follow its CUDA-build instructions. A different Torch wheel does not guarantee identical GPU numerics.
- [requirements-openfe.txt](requirements-openfe.txt): **separate** Python 3.11 environment for OpenFE 0.0.12; do not combine its scikit-learn pins with the main environment.
- [requirements-ppgr.txt](requirements-ppgr.txt): separate Python 3.9/XGBoost 1.6.2 environment. Set `CGMACROS_PPGR_PYTHON` to its interpreter. The Windows fallback is `.venv-ppgr/Scripts/python.exe`; POSIX users must set the variable. These dependencies differ in documented respects from the author's original environment.

LLM-guided search used the Claude Code CLI (`claude -p`), with recorded model identifier `claude-sonnet-5`. It requires a separately configured client and the adapter's model, login, and billing safeguards. Search settings remain disabled, credentials are not included, and the historical model identifier may not be available to another account. Published selections and budget summaries support inspection; raw response logs and exact search replay inputs are omitted.

## Viewing results and reconstruction limits

**Notebooks 08 and 09 default to read-only result viewing:** `REBUILD_TARGET=None` and `REBUILD_VALIDATION=False`. Their required display tables and figures are included. No prediction, fitting, resampling, or TEST execution is needed for these views.

Notebooks 01-07 retain computation and historical TEST code. Inspect their saved outputs; do not Run All over the included results. Their logical dependency order is 01 through 06, with 07 supplementary, followed by factor discovery/evaluation. This ordering is not an instruction to repeat historical TEST evaluation.

Optional reconstruction requires upstream aligned minutes, admitted windows, manual features, stored baselines, OOF residuals/folds, candidate values, and sometimes offline search logs. These large inputs and checkpoints are **not bundled**. Raw-data download alone does not establish a verified end-to-end reconstruction. Choose an explicit target and a new output directory under `reproductions/`; reconstructed prerequisites are not promoted automatically. Reconstruction has not been validated end to end from this selected bundle. No TEST permission is supplied.

The included CSVs do support regenerating the two search figures without training or resampling, using a new directory:

```powershell
python -B -m src.discovery_cli search-figures --output-dir reproductions/search_figures
```

The [selection curves](figures/search_validation_curves.png) use saved eligible-participant selection paths. The [participant differences](figures/search_subject_differences.png) use saved E3 errors with at least 20 core validation windows; black marks are unweighted means, not confidence intervals.

## Tests

The tests cover synthetic data, temporal boundaries, pairing, DSL budgets, model behavior, and selected saved-result contracts. TEST-control tests use temporary synthetic fixtures, not the real TEST evaluation. Examples in the main environment:

```powershell
python -B tests/test_execution_control.py
python -B tests/test_factor_evaluation.py
```

OpenFE and sequence tests have their respective environment requirements; OpenFE search tests perform synthetic searches. The presence of these tests does not establish full reconstruction or validate new scientific results.
