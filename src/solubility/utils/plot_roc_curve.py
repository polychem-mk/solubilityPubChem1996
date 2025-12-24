import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay
from typing import Sequence, Tuple, Optional

def plot_roc_curve(
    y_true,
    y_score: np.ndarray,
    class_names: Sequence[str],
    *,
    title: str = "One-vs-Rest ROC Curves (Multi-class)",
    title_fontsize: int = 22,
    label_fontsize: int = 18,
    tick_fontsize: int = 14,
    legend_fontsize: int = 11,
    colors: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (10, 8)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot One-vs-Rest ROC curves for multi-class classification.

    Parameters
    ----------
    y_true : pd.Series
        True labels from the test set.
    y_score : np.ndarray of shape (n_samples, n_classes)
        Predicted probabilities from model.predict_proba()
    class_names : Sequence[str]
        List or array of class names in correct order
    title, title_fontsize, label_fontsize, tick_fontsize, legend_fontsize, colors
        styling
    figsize : tuple, default (10, 8)

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        Use fig.savefig(...) to save the plot
    """
    # Check the class names
    n_classes = len(class_names)
    if y_score.shape[1] != n_classes:
        raise ValueError("y_score columns must match number of class_names")

    # Default colors
    if colors is None:
        colors = plt.cm.tab10.colors[:n_classes]

    # Make ROC curves for each class
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_true == class_names[i], 
            y_score[:, i],
            name=f"{class_names[i]} vs Rest",
            ax=ax,
            curve_kwargs=dict(color=color, lw=2),
        )

    # Diagonal line
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Chance")

    # Styling
    ax.set_title(title, fontsize=title_fontsize, pad=20, color="#708090")
    ax.set_xlabel("False Positive Rate", fontsize=label_fontsize)
    ax.set_ylabel("True Positive Rate", fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.legend(loc="lower right", fontsize=legend_fontsize)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax
