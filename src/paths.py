"""Project and dataset locations.

This module is the single source of truth for filesystem locations used by the
notebooks. The project root is derived from this file's location, so it does
not depend on the current working directory or on where the raw data lives.

Raw data location:

- ``CGMACROS_DATA_DIR`` set     -> that directory is the dataset root;
- ``CGMACROS_DATA_DIR`` not set -> ``<project>/data/raw/cgmacros-v1.0.0``.

The dataset root is the directory that contains ``CGMacros/`` and the data
dictionary files (not the ``CGMacros/`` directory itself).
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_DATA_ROOT",
    "DATA_ENV_VAR",
    "DATA_ROOT",
    "CGMACROS_DIR",
    "DATA_DICTIONARY_PDF",
    "BIO_DICTIONARY",
    "DYNAMIC_DICTIONARY",
    "GUT_DICTIONARY",
    "MICROBIOME_DICTIONARY",
    "LICENSE_FILE",
    "CHECKSUM_FILE",
    "BIO_TABLE",
    "GUT_HEALTH_TABLE",
    "MICROBIOME_TABLE",
    "REQUIRED_DATA_PATHS",
    "ARTIFACTS_ROOT",
    "artifact_dir",
    "validate_data_root",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "cgmacros-v1.0.0"
DATA_ENV_VAR = "CGMACROS_DATA_DIR"


def _resolve_data_root() -> Path:
    override = os.environ.get(DATA_ENV_VAR, "").strip()
    root = Path(override) if override else DEFAULT_DATA_ROOT
    return root.expanduser().resolve()


DATA_ROOT = _resolve_data_root()

CGMACROS_DIR = DATA_ROOT / "CGMacros"

DATA_DICTIONARY_PDF = DATA_ROOT / "DataDictionary.pdf"
BIO_DICTIONARY = DATA_ROOT / "DataDictionary_Bio.csv"
DYNAMIC_DICTIONARY = DATA_ROOT / "DataDictionary_CGMacros-00X.csv"
GUT_DICTIONARY = DATA_ROOT / "DataDictionary_Gut_Health_Test.csv"
MICROBIOME_DICTIONARY = DATA_ROOT / "DataDictionary_Microbes.csv"
LICENSE_FILE = DATA_ROOT / "LICENSE.txt"
CHECKSUM_FILE = DATA_ROOT / "SHA256SUMS.txt"

BIO_TABLE = CGMACROS_DIR / "bio.csv"
GUT_HEALTH_TABLE = CGMACROS_DIR / "gut_health_test.csv"
MICROBIOME_TABLE = CGMACROS_DIR / "microbes.csv"

# Files and directories the notebooks read unconditionally. LICENSE.txt is
# cataloged by the data audit but is not required to run the pipeline.
REQUIRED_DATA_PATHS = (
    CGMACROS_DIR,
    DATA_DICTIONARY_PDF,
    BIO_DICTIONARY,
    DYNAMIC_DICTIONARY,
    GUT_DICTIONARY,
    MICROBIOME_DICTIONARY,
    CHECKSUM_FILE,
    BIO_TABLE,
    GUT_HEALTH_TABLE,
    MICROBIOME_TABLE,
)

ARTIFACTS_ROOT = PROJECT_ROOT / "results"


def artifact_dir(name: str, create: bool = True) -> Path:
    """Return ``results/<name>``, creating it unless ``create`` is False."""
    if not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"Artifact directory name must be a relative name inside results/: {name!r}")
    path = ARTIFACTS_ROOT / name
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def validate_data_root() -> Path:
    """Check that the CGMacros v1.0.0 dataset is present and return its root.

    Raises a setup-oriented ``FileNotFoundError`` that names the resolved
    location, how it was chosen, and every missing path.
    """
    missing = [path for path in REQUIRED_DATA_PATHS if not path.exists()]
    if not missing:
        return DATA_ROOT

    if os.environ.get(DATA_ENV_VAR, "").strip():
        source = f"from the {DATA_ENV_VAR} environment variable"
    else:
        source = f"the default location (set {DATA_ENV_VAR} to use another directory)"
    lines = [
        "CGMacros v1.0.0 was not found or is incomplete.",
        f"Dataset root: {DATA_ROOT} ({source}).",
        "Missing:",
        *[f"  - {path.relative_to(DATA_ROOT).as_posix()}" for path in missing],
        "Extract the dataset so that the dataset root contains CGMacros/ and the "
        "DataDictionary* files, placing it at data/raw/cgmacros-v1.0.0/ or setting "
        f"{DATA_ENV_VAR} to the dataset root. See data/README.md.",
    ]
    if DATA_ROOT.name == "CGMacros" and (DATA_ROOT / "bio.csv").exists():
        lines.append(
            f"{DATA_ENV_VAR} appears to point at the CGMacros/ subdirectory; "
            "point it at its parent directory instead."
        )
    raise FileNotFoundError("\n".join(lines))
