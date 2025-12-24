import inspect
import pandas as pd
from pathlib import Path
import re
from typing import Callable

from rdkit import Chem
from rdkit.Chem import (
    Descriptors,
    rdchem,    
)

from src.solubility import (
    DATA_PROCESSED,
    DATA_RAW,    
    DESCRIPTORS,
    TARGET,
)

# --- 1. Load data -----------------------------------------

def load_data(file: str | Path | None = None) -> pd.DataFrame:
    """
    Load the data from the CSV file and display the first 6 rows.

    Parameters
    ----------
    file : str | Path | None, default None
        Path to the CSV file.
        If None, uses the default location defined in config.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.
    
    Raises
    ------
    FileNotFoundError
        If the file is missing.
    """
    default_path = DATA_RAW / "PubChemAID1996.csv"  
    path = Path(file) if file is not None else default_path

    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    print(f"Loading raw data from {path}")

    print(pd.read_csv(path).head())

    return pd.read_csv(path)

# --- 2. Compute descriptors -----------------------------------

def compute_descriptors(
    df: pd.DataFrame,
    descriptors: dict[str, tuple[str, Callable]],
    smiles_col: str = "SMILES",
) -> None:
    """
    Add RDKit descriptors to the DataFrame and drop rows with invalid SMILES (in-place).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a column with SMILES strings.
    descriptors : dict[str, tuple[str, Callable]]
        Mapping new column name (descriptor) from {descriptor, (original RDKit name, function)}
    smiles_col : str, default "SMILES"
        Name of the column containing SMILES.

    Notes
    -----
    - Invalid SMILES are removed **in-place**.
    - The function modifies ``df`` directly and does not return anything.    
    """
    if smiles_col not in df.columns:
        raise KeyError(f"Column '{smiles_col}' not found in DataFrame.")

    initial_count = len(df)
    print(f"Computing descriptors for {initial_count:,} molecules...")

    # Convert SMILES to RDKit molecules 
    mols = df[smiles_col].apply(Chem.MolFromSmiles)

    # Drop invalid SMILES
    valid_mask = mols.notna()
    n_invalid = len(df) - valid_mask.sum()

    if n_invalid:
        print(f"Dropping {n_invalid:,} rows with invalid SMILES")
        df.drop(df.index[~valid_mask], inplace=True)
        mols = mols[valid_mask]

    valid_count = len(df)
    print(f"Proceeding with {valid_count:,} valid molecules")

    # Compute and add descriptors
    new_columns = []
    for col_name, (_, func) in descriptors.items():
        if col_name in df.columns:
            continue

        df[col_name] = [func(mol) if mol else None for mol in mols]
        new_columns.append(col_name)

    if new_columns:
        print(f"Added {len(new_columns)} descriptors: {new_columns}")
    else:
        print("All requested descriptors already present")

# --- 3. Descriptor metadata -----------------------------------

def get_descriptor_metadata() -> pd.DataFrame:
    """
    Generate and save a table of molecular descriptors added to ``df``.

    For each descriptor column in the dataset, this function extracts:
      - The column name (key in ``DESCRIPTORS``)
      - The original RDKit function name
      - A short description taken from the function's docstring (first line)
    
    Returns
    -------
    pd.DataFrame
        Table with columns:
          - ``descriptor``: final column name used in modeling
          - ``rdkit_function``: original RDKit calculator name
          - ``molecular_property``: short description of what the descriptor measures

    Notes
    -----
    - The metadata file is saved to ``RESULTS / "descriptor_metadata.csv"``.
    - If a descriptor has no docstring, the RDKit function name is used as an alternative.
    """
    print("Generating descriptor metadata...")
    rows = []
    for col_name, (rdkit_name, func) in DESCRIPTORS.items():
        doc = inspect.getdoc(func)
        # Take first line of docstring, before any "->", strip whitespace
        desc = (
            doc.split("\n", 1)[0].split("->", 1)[0].strip()
            if doc
            else rdkit_name
        )
        rows.append(
            {
                "descriptor": col_name,
                "rdkit_function": rdkit_name,
                "molecular_property": desc,
            }
        )
    df = pd.DataFrame(rows)

    metadata_file = DATA_PROCESSED / "descriptor_metadata.csv"
    df.to_csv(metadata_file, index=False)
    print(f"Descriptor metadata saved to {metadata_file}")
    
    return df    

# --- 4. Modify descriptors ------------------------------------

def modify_descriptors(df: pd.DataFrame) -> None:
    """
    Add carbon-normalized versions of count-based descriptors and update metadata.

    This function:
    - Adds a new column ``n_c`` = heavy atoms - heteroatoms
    - For every RDKit descriptor whose name contains "Num", "Count" or "fr_",
      creates ``<name>_rel = <name> / n_c``
    - Appends the new descriptors to the on-disk ``descriptor_metadata.csv``

    The function modifies ``df`` in place and **does not require** a ``metadata``
    DataFrame to be passed — it reads the current metadata file directly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that already contains the original RDKit descriptors and the
    columns ``n_ha`` (heavy atom count) and ``n_het`` (heteroatom count).    
    """
    # Add carbon count descriptor
    if "n_c" in df.columns:
        print("Column 'n_c' already exists, skipping creation")
    else:
        df["n_c"] = df["n_ha"] - df["n_het"]
        print("Added column 'n_c' (carbon atom count)")

    # Identify count-based descriptors
    count_descr = [
        col for col, (rdkit_name, _) in DESCRIPTORS.items()
        if re.search(r"Num|Count|fr_", rdkit_name)
    ]

    if not count_descr:
        print("No count-based descriptors found, nothing to do")
        return

    # Load current metadata
    metadata_path = DATA_PROCESSED / "descriptor_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    meta = pd.read_csv(metadata_path)

    # Create relative descriptors + update metadata
    new_rows = []
    new_columns = []

    for col in count_descr:
        rel_col = f"{col}_rel"
        if rel_col in df.columns:
            continue

        df[rel_col] = df[col] / df["n_c"]
        new_columns.append(rel_col)

        # Update the original metadata
        mask = meta["descriptor"] == col
        if not mask.any():
            raise KeyError(f"Descriptor '{col}' not found in metadata")
        orig_row = meta.loc[mask].squeeze()

        new_rows.append(
            {
                "descriptor": rel_col,
                "rdkit_function": orig_row["rdkit_function"],
                "molecular_property": f"{orig_row['molecular_property']} (relative to carbon count)",
            }
        )

    if new_rows:
        meta = pd.concat([meta, pd.DataFrame(new_rows)], ignore_index=True)
        meta.to_csv(metadata_path, index=False)
        print(f"Added {len(new_rows)} relative descriptors: {new_columns}")
        print(f"Updated metadata {metadata_path}")
    else:
        print("All relative descriptors already present, nothing added")

# --- 5. Add counts for chemical bond types --------------------

def add_bond_types(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
) -> None:
    """
    Add counts of single, double, triple, and aromatic bonds to ``df``.

    The function:
    - Computes bond type counts using RDKit's bond iteration
    - Adds four new columns: ``single``, ``double``, ``triple``, ``aromatic``
    - Appends the new descriptors to the on-disk ``descriptor_metadata.csv``
    - For invalid SMILES, the bond counts values are set to 0, with warning

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column with valid SMILES strings.
    smiles_col : str, default "SMILES"
        Name of the column containing SMILES.
    """
    metadata_path = DATA_PROCESSED / "descriptor_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if smiles_col not in df.columns:
        raise KeyError(f"Column '{smiles_col}' not found")

    # Convert SMILES to molecules (with validation)
    mols = df[smiles_col].apply(Chem.MolFromSmiles)
    if mols.isna().any():
        n_invalid = mols.isna().sum()
        print(f"Warning: {n_invalid} invalid SMILES — bond counts set to 0")
        mols = mols.fillna(None)

    # Bond type counting using numeric values
    bond_types = {
        "single": rdchem.BondType.SINGLE,      # 1.0
        "aromatic": rdchem.BondType.AROMATIC,  # 1.5
        "double": rdchem.BondType.DOUBLE,      # 2.0
        "triple": rdchem.BondType.TRIPLE,      # 3.0
    }

    # Create bond type counts descriptors
    new_columns = {}
    for name, bond_type in bond_types.items():
        col_name = name
        if col_name in df.columns:
            print(f"Column '{col_name}' already exists — skipping")
            continue

        counts = []
        for mol in mols:
            if mol is None:
                counts.append(0)
                continue
            count = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == bond_type)
            counts.append(count)

        new_columns[col_name] = counts

    if not new_columns:
        print("All bond type descriptors already present")
        return

    # Add to DataFrame
    for col_name, values in new_columns.items():
        df[col_name] = values

    print(f"Added bond type descriptors: {list(new_columns)}")

    # Update metadata
    meta = pd.read_csv(metadata_path)

    new_rows = [
        {
            "descriptor": name,
            "rdkit_function": "Custom (bond type count)",
            "molecular_property": f"Number of {name} bonds",
        }
        for name in new_columns
    ]

    meta = pd.concat([meta, pd.DataFrame(new_rows)], ignore_index=True)
    meta.to_csv(metadata_path, index=False)
    print(f"Updated descriptor metadata in {metadata_path}")

# --- 6. Add all PEOE_VSA descriptors --------------------------

def add_peoe_vsa(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
) -> None:
    """
    Add all PEOE_VSA descriptors to the DataFrame and update metadata.

    The function:
    - Computes the 14 PEOE_VSA descriptors 
    - Adds them as new columns to ``df`` (peoe_vsa1 … peoe_vsa14)
    - Appends PEOE_VSA descriptors to the file ``descriptor_metadata.csv`` 
    - For invalid SMILES, the PEOE_VSA values are set to 0.0, with warning

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a column with valid SMILES strings.
    smiles_col : str, default "SMILES"
        Name of the column containing SMILES.    
    """
    metadata_path = DATA_PROCESSED / "descriptor_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if smiles_col not in df.columns:
        raise KeyError(f"Column '{smiles_col}' not found")

    # Convert SMILES to molecules
    mols = df[smiles_col].apply(Chem.MolFromSmiles)
    if mols.isna().any():
        n_invalid = mols.isna().sum()
        print(f"Warning: {n_invalid} invalid SMILES — PEOE_VSA values set to 0")
        mols = mols.fillna(None)

    # Load current metadata
    meta = pd.read_csv(metadata_path)

    # Identify PEOE_VSA descriptors
    peoe_descriptors = [
        (name.lower(), name, func)
        for name, func in Descriptors._descList
        if "PEOE_VSA" in name
    ]

    # Create PEOE_VSA descriptors + update metadata
    new_columns = []
    new_meta_rows = []

    for col_name, rdkit_name, func in peoe_descriptors:
        if col_name in df.columns:
            continue

        # Add peoe_vsa* columns 
        values = [func(mol) if mol else 0.0 for mol in mols]
        df[col_name] = values
        new_columns.append(col_name)

        # Update current metadata
        doc = inspect.getdoc(func)
        description = doc.split("\n", 1)[0].strip() if doc else "PEOE partial charge weighted surface area"

        new_meta_rows.append({
            "descriptor": col_name,
            "rdkit_function": rdkit_name,
            "molecular_property": description,
        })

    if not new_columns:
        print("All PEOE_VSA descriptors already present")
        return

    # Update metadata file
    meta = pd.concat([meta, pd.DataFrame(new_meta_rows)], ignore_index=True)
    meta.to_csv(metadata_path, index=False)

    print(f"Added {len(new_columns)} PEOE_VSA descriptors: {new_columns}")
    print(f"Updated metadata in {metadata_path}")

# --- 7. Impute missing values ​​for the calculated descriptors

def impute_missing_by_target(
    df: pd.DataFrame,
    target_col: str = "solubility",
) -> None:
    """
    Impute missing descriptor values with class-specific means (grouped by target).

    Missing values in RDKit descriptors are rare, so class-conditional mean imputation
    is a safe, interpretable choice that preserves solubility-specific patterns.

    The function:
    - Uses all descriptor columns from the on-disk ``descriptor_metadata.csv``
    - Imputes in-place using the mean within each ``target_col`` class
    - Skips descriptors with no missing values

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing descriptors and the target column.
    target_col : str, default "solubility"
        Name of the categorical target column used for stratified imputation.
    """
    metadata_path = DATA_PROCESSED / "descriptor_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")

    # Load descriptor names from metadata
    meta = pd.read_csv(metadata_path)
    descriptor_cols = meta["descriptor"].tolist()

    # Find columns with any missing values
    missing_in_cols = df[descriptor_cols].columns[df[descriptor_cols].isna().any()].tolist()

    if not missing_in_cols:
        print("No missing values in descriptors — nothing to impute")
        return

    print(f"Imputing missing values in {missing_in_cols} descriptors using {target_col}-specific means")

    # Stratified mean imputation
    means = df.groupby(target_col)[missing_in_cols].transform("mean")
    df[missing_in_cols] = df[missing_in_cols].fillna(means)

    n_imputed = df[missing_in_cols].isna().sum().sum()
    if n_imputed == 0:
        print("Successfully imputed all missing values")
    else:
        print(f"Warning: {n_imputed} values could not be imputed")

# --- 8. Save descriptors --------------------------------------

def save_descriptors(df: pd.DataFrame) -> None:
    """
    Save descriptor table as Feather.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains the RDKit descriptors.

    Notes
    -----
    The descriptor data file is saved to ``DATA_PROCESSED / "PubChem1996Descriptors.feather"``.
    """
    descriptors_file = DATA_PROCESSED / "PubChem1996Descriptors.feather"
    
    df.to_feather(descriptors_file)
    print(f"Descriptors saved to {descriptors_file}")

#  ------------------ Run descriptors pipeline -----------------

def compute_descriptors_pipeline() -> None:
    """
    Run descriptors pipeline:
    - Load data containing SMILES
    - Compute descriptors
    - Save descriptors data
    - Save descriptors metadata

    This function is used by the main.py file to perform step 1:
    calculating RDKit descriptors and generating metadata.
    """
            
    # 1. Load raw data 
    df = load_data()

    # 2. Compute descriptors
    compute_descriptors(df, descriptors=DESCRIPTORS)

    # 3. Creates and save the initial descriptors metadata
    get_descriptor_metadata()

    # 4. Modify count-based descriptors and update metadata file
    modify_descriptors(df)

    # 5. Add counts for chemical bond types and update metadata file
    add_bond_types(df)

    # 6. Add PEOE_VSA descriptors and update metadata file
    add_peoe_vsa(df)

    # 7. Impute missing values
    impute_missing_by_target(df, target_col=TARGET)

    # 8. Save descriptors
    save_descriptors(df)

    print("Descriptor pipeline completed successfully\n")
    print("=" * 70)
