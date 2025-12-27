"""
This __init__.py 
- Makes constants and paths available with a single import, example:

    from src.solubility import (
        DATA_PROCESSED,
        DATA_RAW,    
        DESCRIPTORS,
        TARGET,
)

- Defines the public API via __all__ so that `from src.solubility import *`
is safe.
"""

# Import everything from config.py 
from .config import (
    DATA_RAW,
    DATA_PROCESSED,
    MODELS,
    FEATURES,
    FEATURE_SELECTION,
    FIGURES,
    TABLES,
    SAVE_MODELS,
    SEARCH,
    SCORER,
    SEED,
    RECOMPUTE,
    N_JOBS,
    DPI,
    TEST_SIZE,
    DESCRIPTORS,
    TARGET,
    CLASS_NAMES,
)

# Only names listed here will be imported with `from src.python import *`
__all__ = [
    "DATA_RAW",
    "DATA_PROCESSED",
    "MODELS",
    "FEATURES",
    "FEATURE_SELECTION",
    "FIGURES",
    "TABLES",
    "SAVE_MODELS",
    "SEARCH",
    "SCORER",
    "SEED",
    "RECOMPUTE",
    "N_JOBS",
    "DPI",
    "TEST_SIZE",
    "DESCRIPTORS",
    "TARGET",
    "CLASS_NAMES",
]
