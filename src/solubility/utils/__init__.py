"""
This __init__.py 
- Makes utils functions available with a single import, example:

    from src.solubility.utils import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)

- Defines the public API via __all__
"""

from .plot_confusion_matrix import plot_confusion_matrix
from .plot_feature_importance import plot_feature_importance
from .plot_roc_curve import plot_roc_curve

__all__ = [
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_roc_curve",
]
