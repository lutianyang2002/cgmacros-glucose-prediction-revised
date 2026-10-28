# Data

CGMacros v1.0.0 is distributed by [PhysioNet](https://physionet.org/content/cgmacros/1.0.0/). Obtain it under its original license; this release does not bundle the raw participant records or environments.

Set `CGMACROS_DATA_DIR` to the directory containing `CGMacros/`, `DataDictionary*`, `LICENSE.txt` and `SHA256SUMS.txt`. Otherwise the default is `data/raw/cgmacros-v1.0.0/`. `CGMacros/` contains participant directories and `bio.csv`, `gut_health_test.csv`, `microbes.csv`.

Raw inputs remain unchanged. The data-quality notebook checks the files listed in the release checksum file; participant CSVs and static tables are not covered by those checksums. Keep original line endings and encodings. Selected derived inputs and saved outputs are under `results/`; their presence does not establish a clean-room reconstruction.
