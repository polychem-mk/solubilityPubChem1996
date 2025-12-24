import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyarrow import ArrowInvalid
from pathlib import Path
import re
from scipy.stats import randint
from typing import Literal

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)

from src.solubility import (
    CLASS_NAMES,
    DATA_PROCESSED,
    DPI,
    FEATURES,
    FIGURES,
    MODELS,
    N_JOBS,
    RECOMPUTE,
    SCORER,
    SEARCH,
    SEED,    
    TABLES,
    TARGET,
    TEST_SIZE,
)

from src.solubility.utils import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)

# --- 1. Load data ---------------------------------------------

def load_descriptors_data(file: str | Path | None = None) -> pd.DataFrame:
    """
    Load descriptor table from Feather file.

    Parameters
    ----------
    file : str | Path | None, default None
        Path to the Feather file.
        If None, uses the default location defined in config.

    Returns
    -------
    pd.DataFrame
        DataFrame with computed descriptors.

    Raises
    ------
    RuntimeError
        If the file is missing or corrupted (incompatible PyArrow version, truncated, etc.).
    """
    default_path = DATA_PROCESSED / "PubChem1996Descriptors.feather"
    path = Path(file) if file is not None else default_path

    try:
        df = pd.read_feather(path)
    except (FileNotFoundError, ArrowInvalid) as e:
        raise RuntimeError(f"Cannot load descriptors: {e}\nRun descriptor step again.") from e

    print(f"Loaded data: {df.shape[0]:,} molecules, {df.shape[1]} columns from {path}")
   
    return df

# --- 2. Load metadata -----------------------------------------

def load_metadata(file: str | Path | None = None) -> pd.DataFrame:
    """
    Load metadata from CSV file.

    Parameters
    ----------
    file : str | Path | None, default None
        Path to the CSV file.
        If None, uses the default location defined in config.

    Returns
    -------
    pd.DataFrame
        DataFrame containing metadata.
    
    Raises
    ------
    FileNotFoundError
        If the file is missing.
    """
    metadata_file = DATA_PROCESSED / "descriptor_metadata.csv"
    path = Path(file) if file is not None else metadata_file

    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    print(f"Loading metadata from {path}")

    return pd.read_csv(path)

# --- 3. Split to test/train -----------------------------------

def split_data(
    df: pd.DataFrame,
    *,
    target: str = "solubility",
    features: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train/test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target : str, default "solubility"
        Target column name.
    features : list[str]
        **Must** be provided explicitly — no auto-guessing.
    test_size, random_state, stratify : see train_test_split

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found.")

    missing = set(features) - set(df.columns)
    if missing:
        raise KeyError(f"Requested features not in DataFrame: {sorted(missing)}")

    X = df[features]
    y = df[target]

    stratify_array = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_array,
    )

# --- 4. RandomForest model ------------------------------------

def train_rf_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_name: str,
    oob: bool = True,   
) -> RandomForestClassifier:
    """
    Train and save a RandomForest model with balanced class weights.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    model_name : str
        Identifier used in the filename, e.g. "baseline" or "final".
    oob : bool, default True
        Whether to use out-of-bag samples to estimate the generalization score.
    
    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    print(f"Using {X_train.shape[1]} features: {list(X_train.columns)}")
    
    # Train and fit RandomForest model with default max_depth,
    # min_samples_split, min_samples_leaf and max_features
    rf = RandomForestClassifier(
        n_estimators=2000,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=N_JOBS,
        oob_score=oob,
    )

    rf.fit(X_train, y_train)

    if oob:
        print(f"OOB accuracy:  {rf.oob_score_:.4f}")
        print(f"OOB error rate: {1 - rf.oob_score_:.4f}")
    
    # Save the model
    path = MODELS / f"{model_name}_rf.joblib"
    joblib.dump(rf, path, compress=3)
    print(f"{model_name} model saved to {path}")
    return rf

# --- 5. Save metrics ------------------------------------------

def save_metrics(
    model_name: str,
    y_test,
    y_pred,
    class_names: list[str],
    extra_metrics: dict | None = None,
) -> None:
    """
    Save classification metrics as a CSV table.
    
    Parameters
    ----------
    model_name : str
        Identifier used in the filename, e.g. "baseline" or "final_rf".
    y_test : pd.Series
        True labels from the test set.
    y_pred : pd.Series or np.ndarray
        Predicted labels.
    class_names : list[str] | None, default None
        Desired class order (e.g. ["low", "moderate", "high"]).
        If ``None``, uses ``CLASS_NAMES`` from config.
    extra_metrics : dict[str, float] | None, default None
        Additional metrics to append as new columns
        (e.g. ``{"oob_score": 0.823}``).   
    """    
    # Get classification report (ordered by class_names)
    report = classification_report(
        y_test,
        y_pred,
        labels=class_names,
        output_dict=True,
        digits=4
    )

    # Add optional extra metrics
    if extra_metrics:
        report.update(extra_metrics)

    # Convert to tidy DataFrame
    df = (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "class"})
    )

    # Remove the support column 
    df = df.drop(columns="support", errors="ignore")
    
    # Save
    path = TABLES / f"{model_name}_metrics.csv"
    df.to_csv(path, index=False)
    print(f"Metrics saved to {path}")

# --- 6. Feature selection -------------------------------------

def select_features(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    importance_threshold: float = 0.02,
    corr_threshold: float = 0.9,
) -> list[str]:
    """
    Select features using MDI + permutation importance + correlation filtering.    

    Parameters
    ----------
    model : fitted estimator with feature_importances_
        Trained RandomForest model
    X_train, y_train : pd.DataFrame, pd.Series
        Training data.
    importance_threshold : float, default 0.02
        Drop features below this importance.
    corr_threshold : float, default 0.9
        Drop less important feature if correlation > threshold.

    Returns
    -------
    list[str]
        Selected feature names.

    Notes
    -----
    This function saves feature importances table as csv file to
    ``TABLES / "feature_importances.csv"``.
    """
    # Feature importances: MDI + Permutation 
    mdi = model.feature_importances_
    perm = permutation_importance(
        model,
        X_train,
        y_train,
        scoring=SCORER,
        n_repeats=5,
        random_state=SEED,
        n_jobs=1,
    )
    perm_imp = perm.importances_mean

    # Normalize permutation importance to [0,1]
    perm_imp_normalized = perm_imp / perm_imp.max()

    # Combined score
    combined = (mdi + perm_imp_normalized) / 2
    
    # Set combined feature importances in descending order
    ranked_idx = np.argsort(combined)[::-1]

    # Correlation-based removal 
    corr_matrix = X_train.corr().abs()
    to_drop = set()

    for i in range(len(ranked_idx)):
        col_i = X_train.columns[ranked_idx[i]]
        imp_i = combined[ranked_idx[i]]

        # Drop very low importance
        if imp_i < importance_threshold:
            to_drop.add(col_i)
            continue

        # Drop highly correlated, less important features
        for j in range(i + 1, len(ranked_idx)):
            col_j = X_train.columns[ranked_idx[j]]
            if corr_matrix.loc[col_i, col_j] > corr_threshold:
                to_drop.add(col_j)

    selected = [col for col in X_train.columns if col not in to_drop]
    n_kept = len(selected)
    n_total = X_train.shape[1]

    print(f"Feature selection: {n_kept}/{n_total} features kept")    
    if n_kept < n_total:
        print(f"Kept: {selected}")
    print(f"Dropped {n_total - n_kept}: {sorted(to_drop)}")

    path = FEATURES / "selected_features.joblib"
    joblib.dump(selected, path)
    print(f"Selected features saved to {path}")

    # Save feature_importances_df as csv file
    feature_importances_df = pd.DataFrame(
        {"descriptor": model.feature_names_in_,
        "mdi": mdi,
        "perm_imp": perm_imp,
        "perm_imp_normalized": perm_imp_normalized,
        "combined": combined,        
        }
    )
    feature_importances_df["selected"] = feature_importances_df["descriptor"].isin(selected)
    path = TABLES / "feature_importances.csv"
    feature_importances_df.to_csv(path, index=False)
    print(f"Feature importances saved to {path}")

    return selected

# --- 7. Find hyperparameters and train the final model --------

def tune_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    recompute: bool = True,
) -> RandomForestClassifier:
    """
    Perform two-stage hyperparameter tuning (Random, Grid) and return final model.

    Parameters
    ----------
    X_train, y_train : pd.DataFrame, pd.Series
        Training data (before feature selection).
    recompute : bool, default True
        If False, load pre-trained final model from disk.
    
    Returns
    -------
    RandomForestClassifier
        Best model trained on full training data with 2000 trees.
    """
    final_model_path = MODELS / "final_model_rf.joblib"
    random_search_path = SEARCH / "random_search_rf.joblib"
    grid_search_path = SEARCH / "grid_search_rf.joblib"

    # Load pre-computed model if 'recompute' is set to False
    if not recompute and final_model_path.exists():
        print(f"Loading pre-tuned final model from {final_model_path}")
        return joblib.load(final_model_path)
    
    # Set cross validation object
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # --- Step 1: Randomized Search ---
    print("\nStarting RandomizedSearchCV ...")
    rf = RandomForestClassifier(
        n_estimators=300,           # fast for search
        class_weight="balanced",
        random_state=SEED,
        n_jobs=N_JOBS,
    )

    param_dist = {
        "max_depth": [None] + list(range(4, 16, 2)),
        "min_samples_split": randint(4, 20),
        "min_samples_leaf": randint(1, 10),
    }

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv,
        scoring=SCORER,
        random_state=SEED,
        n_jobs=N_JOBS,
        verbose=1,
        error_score='raise',
    )
    random_search.fit(X_train, y_train)

    print(f"Best random search score: {random_search.best_score_:.4f}")
    print(f"Best random params: {random_search.best_params_}")

    joblib.dump(random_search, random_search_path, compress=3)
    print(f"Random search saved to {random_search_path}")

    # --- Step 2: Narrow Grid Search around best params ---
    print("\nStarting GridSearchCV around best random params...")

    best_params = random_search.best_params_.copy()
    
    max_features_sqrt = max(round(np.sqrt(X_train.shape[1])), 3)
    max_depth = (np.array([None, 1, 2]) 
        if best_params['max_depth'] is None 
        else max(best_params['max_depth'], 2)+ np.array([-1, 0, 1]))   

    param_grid = {
        "max_depth": max_depth,        
        "min_samples_split": max(best_params['min_samples_split'], 4) + np.array([-2, 0, 2]),        
        "min_samples_leaf": max(best_params['min_samples_leaf'], 2) + np.array([-1, 0, 1]),        
        "max_features": max_features_sqrt + np.array([-1, 0, 1]),
    }
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring=SCORER,
        n_jobs=N_JOBS,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best grid search score: {grid_search.best_score_:.4f}")
    print(f"Best grid params: {grid_search.best_params_}")

    joblib.dump(grid_search, grid_search_path, compress=3)
    print(f"Grid search saved to {grid_search_path}")

    # --- Final model: best params + more trees ---
    print("\nTraining final model with n_estimators=2000...")
    final_model = clone(grid_search.best_estimator_)
    final_model.set_params(
        n_estimators=2000,
        warm_start=False,
        oob_score=True,
    )
    final_model.fit(X_train, y_train)

    joblib.dump(final_model, final_model_path, compress=3)
    print(f"Final model saved to {final_model_path}")

    return final_model

# --- 8. Model Evaluation and plots ----------------------------

def evaluate_and_plot(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    class_names: list[str] | None = None,
    save_plots: bool = True,
    show_plots: bool = False,
) -> None:
    """
    Evaluate final model and save diagnostic plots and table.

    Parameters
    ----------
    model : fitted estimator
        Must have `.predict()`, `.predict_proba()`, `.feature_importances_`, 
        `.feature_names_in_`, and `.classes_`.
    X_test, y_test : pd.DataFrame, pd.Series
        Test set.
    class_names : list[str] | None, default None
        Desired class order (e.g. ["low", "moderate", "high"]).
        If ``None``, uses ``CLASS_NAMES`` from config.
    save_plots : bool, default True
        Save plots to FIGURES directory.
    show_plots : bool, default False
        Display plots.
    """
    # Output paths 
    cm_path = FIGURES / "confusion_matrix.png"
    cm_df_path = TABLES / "confusion_matrix.csv"
    fi_path = FIGURES / "feature_importance.png"
    roc_path = FIGURES / "roc_curves.png"

    # Set class names with correct order 
    if class_names is None:
        class_names = CLASS_NAMES

    # Check the class names
    model_classes = set(model.classes_)
    desired_classes = set(class_names)
    if model_classes != desired_classes:
        raise ValueError(
            f"Model was trained on classes {sorted(model_classes)}, "
            f"but CLASS_NAMES expects {class_names}. "
            "Update config or retrain model."
        )
    
    print(f"Evaluating on test set: {X_test.shape[0]:,} samples, "
          f"{X_test.shape[1]} features")

    # Predictions 
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    # --- Classification report ---
    print("\n" + "-"*70)
    print("Classification report for the final Random Forest model")
    print("-"*70)
    print(classification_report(y_test, y_pred, labels=class_names, digits=4))
    print("-"*70)

    # --- Plot 1: Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred, labels=class_names)

    # Save confusion matrix to csv file
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(cm_df_path, index=False)
    print(f"Confusion matrix saved to {cm_df_path}")

    # Confusion matrix plot
    fig, ax = plot_confusion_matrix(cm, class_names)
    if save_plots:
        fig.savefig(cm_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"Confusion matrix plot saved to {cm_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # --- Plot 2: Feature Importance ---
    feat_imp = pd.DataFrame({
        "feature": model.feature_names_in_,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    fig, ax = plot_feature_importance(feat_imp, top_n=model.n_features_in_)
    if save_plots:
        fig.savefig(fi_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"Feature importance saved to {fi_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    # --- Plot 3: ROC Curves (multi-class) ---
    fig, ax = plot_roc_curve(
        y_true=y_test,
        y_score=y_score,
        class_names=model.classes_,
        title="Final Model — Multi-class ROC Curves",
    )
    if save_plots:
        fig.savefig(roc_path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"ROC curves saved to {roc_path}")
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    print("\nEvaluation complete.")

# -------------------- Modeling pipeline -----------------------

def run_full_modeling_pipeline(
    *,
    recompute_tuning: bool | None = None,
    feature_selection: Literal["load", "skip", "recompute"] = "load",
    save_plots: bool = True,
    show_plots: bool = False,
) -> None:
    """
    Run the complete modeling pipeline from data loading to final evaluation.

    This function is used by main.py file to perform step 2:
    train baseline + tuned model, evaluate, save results.
    
    Parameters
    ----------
    recompute_tuning : bool | None, default None
        If None, uses RECOMPUTE from config. If True/False, overrides it.
    feature_selection: FeatureSelection = FeatureSelection.LOAD
        - "recompute" - perform feature selection using the input model
        - "load" (default) - load pre-selected features from FEATURES / "selected_features.joblib"
        - "skip" - skip feature selection and use all descriptors
    save_plots : bool, default True
        Save plots to FIGURES directory.
    show_plots : bool, default False
        Display plots.        
    """
    choices = {"load", "skip", "recompute"}    
    if feature_selection not in choices:
        raise ValueError(f"Invalid action '{feature_selection}'. Expected one of {choices}")

    # 1. Load processed data with descriptors and metadata 
    print("\n1. Loading descriptor data")
    df = load_descriptors_data()
    meta = load_metadata()
    
    # 2. Train/test split with all descriptors
    print("\n2. Splitting data")
    X_train, X_test, y_train, y_test = split_data(
        df,
        target=TARGET,
        features=list(meta['descriptor']),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=True,
    )
    print(f"Train set: {X_train.shape[0]:,} samples")
    print(f"Test set : {X_test.shape[0]:,} samples")
    print("\n   Class distribution (train):")
    print(y_train.value_counts(normalize=True).sort_index().round(4))
    print("   Class distribution (test):")
    print(y_test.value_counts(normalize=True).sort_index().round(4))
    
    # 3. Train baseline model with the original descriptors
    print("\n3. Training baseline Random Forest")
    baseline_features = [descr for descr in list( meta['descriptor']) if not re.search(r"_rel", descr)]
    
    baseline_model = train_rf_model(
        X_train[baseline_features],
        y_train,
        model_name="baseline",
        oob=True)
    baseline_pred = baseline_model.predict(X_test[baseline_features])

    oob_score = getattr(baseline_model, "oob_score_", None)
    extra_metrics = {"oob_score": oob_score} if oob_score is not None else None

    save_metrics(
        model_name="baseline_rf",
        y_test=y_test,
        y_pred=baseline_pred,
        class_names=CLASS_NAMES,
        extra_metrics=extra_metrics,
    )
    
    # 4. Train rf model with original + modified descriptors (using entire train set)
    print("\n4. Training Random Forest using original + modified descriptors")

    all_features_model = train_rf_model(
        X_train,
        y_train,
        model_name="all_features",
        oob=True)
    all_features_pred = all_features_model.predict(X_test)

    oob_score = getattr(all_features_model, "oob_score_", None)
    extra_metrics = {"oob_score": oob_score} if oob_score is not None else None

    save_metrics(
        model_name="all_features_rf",
        y_test=y_test,
        y_pred=all_features_pred,
        class_names=CLASS_NAMES,
        extra_metrics=extra_metrics,
    )

    # 5. Feature selection    
    print("\n5. Feature selection")

    match feature_selection:
        case "recompute":
            print("Running feature selection...")
            selected_features = select_features(
                all_features_model,
                X_train,
                y_train,
                importance_threshold=0.005,
                corr_threshold=0.90,
            )
            print("\n6. Training Random Forest using selected descriptors")
            
        case "load":
            path = FEATURES / "selected_features.joblib"
            if not path.exists():
                raise FileNotFoundError(
                    f"Selected features not found: {path}\n"
                    "Run with feature_selection='recompute' first."
                )
            selected_features = joblib.load(path)
            print(f"Loaded {len(selected_features)} pre-selected features")
            print("\n6. Training Random Forest using selected descriptors")

        case "skip":
            selected_features = X_train.columns.tolist()
            print("Feature selection skipped")            
            print(f"\n6. Training Random Forest using all {len(selected_features)} descriptors")            
    
    # 6. Train rf model using selected descriptors
    selected_features_model = train_rf_model(
        X_train[selected_features],
        y_train,
        model_name="selected_features",
        oob=True)
    selected_features_pred = selected_features_model.predict(X_test[selected_features])

    oob_score = getattr(selected_features_model, "oob_score_", None)
    extra_metrics = {"oob_score": oob_score} if oob_score is not None else None

    save_metrics(
        model_name="selected_features_rf",
        y_test=y_test,
        y_pred=selected_features_pred,
        class_names=CLASS_NAMES,
        extra_metrics=extra_metrics,
    )

    # 7. Final model with hyperparameter tuning
    print("\n7. Hyperparameter tuning and final model training")
    recompute = RECOMPUTE if recompute_tuning is None else recompute_tuning

    final_model = tune_model(
        X_train[selected_features],
        y_train,
        recompute=recompute,
    )

    # Get features that are used in the final model
    final_model_features = final_model.feature_names_in_

    print(f"Final model features: {list(final_model_features)}")

    final_pred = final_model.predict(X_test[final_model_features])
   
    oob_score_final = getattr(final_model, "oob_score_", None)
    extra_metrics_final = ({"oob_score": oob_score_final} 
        if oob_score_final is not None 
        else None)

    save_metrics(
        model_name="final_rf",
        y_test=y_test,
        y_pred=final_pred,
        class_names=CLASS_NAMES,
        extra_metrics=extra_metrics_final,
    )
    
    # 8. Model comparison (precision for main classes)
    print("\n8. Model comparison - Precision")
    print("-" * 70)

    baseline_df = pd.read_csv(TABLES / "baseline_rf_metrics.csv", index_col=0)
    all_features_df = pd.read_csv(TABLES / "all_features_rf_metrics.csv", index_col=0)
    selected_features_df = pd.read_csv(TABLES / "selected_features_rf_metrics.csv", index_col=0)
    final_df = pd.read_csv(TABLES / "final_rf_metrics.csv", index_col=0)

    # Keep only the three classes in correct order
    comparison = pd.DataFrame({
        "class": CLASS_NAMES,
        "baseline_precision": baseline_df.loc[CLASS_NAMES, "precision"].round(4).values,
        "all_features_precision": all_features_df.loc[CLASS_NAMES, "precision"].round(4).values,
        "selected_features_precision": selected_features_df.loc[CLASS_NAMES, "precision"].round(4).values,
        "final_precision": final_df.loc[CLASS_NAMES, "precision"].round(4).values,
    })

    # Save comparison table
    path = TABLES / "model_comparison.csv"
    comparison.to_csv(path, index=False)
    print(f"Metrics saved to {path}")
    
    print(comparison.to_string(index=False))
    print("-" * 70)    
    
    # 9. Final evaluation and plots
    print("\n9. Final model evaluation and visualization")
    evaluate_and_plot(
        final_model,
        X_test[final_model_features],
        y_test,
        class_names=CLASS_NAMES,
        save_plots=save_plots,
        show_plots=show_plots,
    )
