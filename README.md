## Predicting Aqueous Solubility of Small Molecules

🧪  Aqueous Solubility Classification Using 2D Molecular Descriptors.

**Goal:** Build a classification model for predicting the solubility of small molecules 
in water (***low***, ***moderate***, ***high***) at physiological pH
 using simple 2D molecular descriptors calculated with the RDKit package.

### Data
* The initial dataset was sourced from PubChem: Assay Identifier (AID) 1996,  the [Aqueous Solubility from MLSMR Stock Solutions](https://pubchem.ncbi.nlm.nih.gov/bioassay/1996) dataset.   
* The original dataset contained 30 variables and 57859  compounds.
* The first 3 rows of the dataset were dropped (did not contain data) and 5 CID duplicates  were  Removed
* For this project, three variables were retained:
	- PUBCHEM_CID (renamed to CID)  
	- PUBCHEM_EXT_DATASOURCE_SMILES (renamed to SMILES) 
	- Solubility.at.pH.7.4_Mean (renamed to solubility_mean) - the mean of the solubility results of the test compound at pH 7.4 in μg/mL.   
* A target variable, solubility, was derived from solubility_mean:   
  - <10 ug/mL = ***low*** solubility (Inactive)  
  - 10-60 ug/mL = ***moderate*** solubility (Active)  
  - \>60 ug/mL = ***high*** solubility (Active)   
  
* The resulting 3.7MB starting dataset is included in this repository in data/raw/PubChemAID1996.csv. Using the curated initial dataset as the starting point avoids duplicating existing data found online and ensures better reproducibility (no web fetching required). The data directory is organized into *raw* and *processed* subdirectories to maintain a clear workflow. The raw data remains unchanged, while generated descriptors and metadata are stored separately in the processed folder.

### Quick Start

This project supports two main workflows: **Conda** (recommended for RDKit + R) and **pure Python (pip)**.

#### Option 1: Conda (Python + R)

1. Install [conda](https://docs.conda.io/en/latest/miniconda.html).  
2. Create and activate the environment:   

```
conda env create -f environment.yml
conda activate solubility-env
```

#### Option 2: pip + venv (Python-only)

```
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Usage

The following directories and files are **excluded** from the repository due to size:

- `data/processed/` (computed RDKit descriptors and metadata)
- `results/models/` (saved Random Forest models)
- `results/hyperparameter_searches/` (intermediate search objects)

These are automatically created:
```
# Full pipeline — generates everything (recommended first run)
python -m src.solubility.main

# Or step-by-step:
# 1. create data/processed/
python -m src.solubility.main --step descriptors
# 2. create results/models/ and results/hyperparameter_searches/
python -m src.solubility.main --step modeling
```

Also:

```
# Run a quick test
python -m src.solubility.main --help

# Generate HTML report (R)
cd analysis
Rscript -e "rmarkdown::render('solubility_-_report.Rmd')"
```

### Project Structure

<pre>
.
├── analysis
│   ├── bibliography.bibtex
│   ├── citation_style.csl
│   ├── solubility_report.Rmd
│   ├── styles.css
│   └── utils
│       └── print_gt_table.R
├── data
│   ├── processed
│   │   ├── descriptor_metadata.csv
│   │   └── PubChem1996Descriptors.feather
│   └── raw
│       └── PubChemAID1996.csv
├── environment.yml
├── index.html                # HTML report
├── LICENSE
├── requirements.in
├── requirements.txt
├── results
│   ├── feature_selection
│   │   └── selected_features.joblib
│   ├── figures
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   └── roc_curves.png
│   ├── hyperparameter_searches
│   │   ├── grid_search_rf.joblib
│   │   └── random_search_rf.joblib
│   ├── models
│   │   ├── all_features_rf.joblib
│   │   ├── baseline_rf.joblib
│   │   ├── final_model_rf.joblib
│   │   └── selected_features_rf.joblib
│   └── tables
│       ├── all_features_rf_metrics.csv
│       ├── baseline_rf_metrics.csv
│       ├── confusion_matrix.csv
│       ├── feature_importances.csv
│       ├── final_rf_metrics.csv
│       ├── model_comparison.csv
│       ├── roc_auc.csv
│       └── selected_features_rf_metrics.csv
├── solubilityPubChem1996.Rproj
└── src
    └── solubility
        ├── config.py
        ├── descriptors.py
        ├── __init__.py
        ├── main.py
        ├── modeling.py
        └── utils
            ├── __init__.py
            ├── plot_confusion_matrix.py
            ├── plot_feature_importance.py
            └── plot_roc_curve.py
</pre>

The `citation_style.csl` file was downloaded from this repository:
[github.com/citation-style-language](<https://github.com/citation-style-language/styles/blob/master/american-chemical-society.csl>)  
and follows the American Chemical Society citation style.  

### AI Assistance Disclosure

🤖  This project was developed with assistance from **Grok 4** (xAI) for:   
• Code structure and best practices  
• Debugging and optimization suggestions  
• README and documentation drafting  

🙋  All scientific and analysis decisions, data preprocessing, and model interpretation were made by the author.
All AI assistance was double-checked.


