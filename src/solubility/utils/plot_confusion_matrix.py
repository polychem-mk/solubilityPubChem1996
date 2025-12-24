import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Sequence, Tuple

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    *,
    title: str = "Confusion Matrix",
    title_fontsize: int = 22,
    label_fontsize: int = 18,
    tick_fontsize: int = 14,
    cell_fontsize: int = 14,
    figsize: Tuple[float, float] = (6, 5)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a styled confusion matrix with green diagonal and red off-diagonal.

    Parameters
    ----------
    cm: np.ndarray
        confusion matrix
        Created as: confusion_matrix(y_test, y_pred, labels=class_names)
    class_names : Sequence[str]
        List or array of class names in correct order
    title, title_fontsize, label_fontsize, tick_fontsize, cell_fontsize
        styling
    figsize : tuple, default (10, 8)
        Figure size in inches
    
    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # Check cm shpae and the class names
    n_classes = cm.shape[0]
    if cm.shape != (n_classes, n_classes):
        raise ValueError("Confusion matrix must be square")
    if len(class_names) != n_classes:
        raise ValueError("Number of class_names must match matrix size")

    fig, ax = plt.subplots(figsize=figsize)

    # Light background heatmap 
    sns.heatmap(cm, annot=False, fmt='d', cmap='Greys', alpha=0.1,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar=False, linewidths=0.5, linecolor='lightgray')

    # Colored overlay + text
    for i in range(n_classes):
        for j in range(n_classes):
            color = 'green' if i == j else 'pink'
            text_color = 'darkgreen' if i == j else 'darkred'
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, 
                                       color=color, alpha=0.3, lw=0))
            ax.text(j + 0.5, i + 0.5, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=cell_fontsize,
                    color=text_color)

    # Labels and title
    ax.set_xlabel('Predicted', fontsize=label_fontsize)
    ax.set_ylabel('True', fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=20, color="#708090")
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    plt.tight_layout()
    return fig, ax
    