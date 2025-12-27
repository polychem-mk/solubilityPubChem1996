"""
Cheminformatics ML Pipeline

Usage:
    python -m src.solubility.main                      # full pipeline
    python -m src.solubility.main --step modeling      # only modeling
    python -m src.solubility.main --step descriptors   # only descriptors
    python -m src.solubility.main --help               # show help
"""

import argparse
from pathlib import Path

from src.solubility import (
    FEATURE_SELECTION,
    RECOMPUTE,
    SAVE_MODELS,
)
from src.solubility.descriptors import compute_descriptors_pipeline
from src.solubility.modeling import run_full_modeling_pipeline

def step_compute_descriptors() -> None:
    """Step 1: Compute RDKit descriptors and generate metadata."""
    print("Step 1: Computing molecular descriptors + metadata")
    print("=" * 70)
    compute_descriptors_pipeline()

def step_modeling() -> None:
    """Step 2: Train baseline + tuned model, evaluate, save results."""
    print("Step 2: Training and evaluating models")
    print("=" * 70)
    run_full_modeling_pipeline(
        recompute_tuning=RECOMPUTE,
        feature_selection=FEATURE_SELECTION,
        save_models = SAVE_MODELS,
        save_plots = True,
        show_plots=False,
    )

# --- CLI entry point ------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solubility prediction ML pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--step",
        choices=["all", "descriptors", "modeling"],
        default="all",
        help="Which part of the pipeline to run",
    )    
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    print(f"Project root: {project_root}")
    print(f"Running step: {args.step}")
    print("\n" + "=" * 70)

    if args.step in ("all", "descriptors"):
        step_compute_descriptors()

    if args.step in ("all", "modeling"):
        step_modeling()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()