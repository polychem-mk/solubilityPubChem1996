from pathlib import Path
from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    Lipinski,
)
from sklearn.metrics import make_scorer, precision_score

# --- Paths ----------------------------------------------------
try:
    from pyprojroot import here
    PROJECT_DIR = here()
except ImportError:
    try:
        PROJECT_DIR = Path(__file__).resolve().parents[2]
    except NameError:
        PROJECT_DIR = Path.cwd()

DATA_PROCESSED = PROJECT_DIR / "data" / "processed"
DATA_RAW = PROJECT_DIR / "data" / "raw"
FEATURES =  PROJECT_DIR / "results" / "feature_selection"
FIGURES = PROJECT_DIR / "results" / "figures"
MODELS = PROJECT_DIR / "results" / "models"
SEARCH = PROJECT_DIR / "results" / "hyperparameter_searches"
TABLES = PROJECT_DIR / "results" / "tables"

# Create if missing
for p in [DATA_PROCESSED, FEATURES, FIGURES, MODELS, SEARCH, TABLES, ]:
    p.mkdir(parents=True, exist_ok=True)

# --- Global constants -----------------------------------------
DPI = 300

# FEATURE_SELECTION settings:
# "load" - load pre-selected features from FEATURES / "selected_features.joblib"
# "skip" - skip feature selection and use all descriptors
# "recompute" - perform feature selection 
FEATURE_SELECTION = "recompute" 

N_JOBS = -2

# Set RECOMPUTE to True to to re-run the final model tuning step. 
# Set RECOMPUTE to False to load saved model
RECOMPUTE = True

# Set SAVE_MODELS to True to save all intermediate models.
# If SAVE_MODELS is False, only the final model will be saved.
SAVE_MODELS = False

SEED = 42
TEST_SIZE = 0.2

# --- RDKit / modeling defaults --------------------------------
CLASS_NAMES = ["low", "moderate", "high"]

DESCRIPTORS = {
    'bertz': ('BertzCT', Descriptors.BertzCT),
    'c_o': ('fr_C_O', Descriptors.fr_C_O),
    'ether': ('fr_ether', Descriptors.fr_ether),
    'max_part_charge': ('MaxPartialCharge', Descriptors.MaxPartialCharge),
    'min_part_charge': ('MinPartialCharge', Descriptors.MinPartialCharge),
    'n_chiral': ('Number_chiral_atoms', lambda x: len(Chem.FindMolChiralCenters(x))),
    'logp': ('MolLogP', Descriptors.MolLogP),
    'mw': ('MolWt', Descriptors.MolWt),
    'nhoh': ('NHOHCount', Descriptors.NHOHCount),
    'n_on': ('NOCount', Descriptors.NOCount),
    'aromatic_rings': ('NumAromaticRings', Descriptors.NumAromaticRings),
    'hba': ('NumHAcceptors', Descriptors.NumHAcceptors),
    'hbd': ('NumHDonors', Descriptors.NumHDonors),
    'nrb': ('NumRotatableBonds', Descriptors.NumRotatableBonds),
    'sps': ('SPS', Descriptors.SPS),
    'tpsa': ('Topological_polar_surface_area', Descriptors.TPSA),
    'n_ha': ('HeavyAtomCount', Lipinski.HeavyAtomCount),
    'n_het': ('NumHeteroatoms', Lipinski.NumHeteroatoms),
}

SCORER = make_scorer(
    precision_score,
    average="macro",
    labels=["high", "moderate"],   # labels of interest
    zero_division=0,
    pos_label=None 
)

TARGET = 'solubility'