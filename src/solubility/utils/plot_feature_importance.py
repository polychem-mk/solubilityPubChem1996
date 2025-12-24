import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple

def plot_feature_importance(
    feat_imp: pd.DataFrame,
    top_n: int = 20,
    *,
    title_fontsize: int = 26,
    label_fontsize: int = 22,
    tick_fontsize: int = 16,
    bar_color: str = "#B0C4DE",
    figsize: Tuple[float, float] = (10, 8),
    asc: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot horizontal bar chart of top N most important features.

    Parameters
    ----------
    feat_imp : pd.DataFrame
        Must contain columns 'feature' (str) and 'importance' (float).
        Created as:
        pd.DataFrame({
            'feature': model.feature_names_in_,
            'importance': model.feature_importances_
        })
    top_n : int, default 20
        Number of top features to display
    title_fontsize, label_fontsize, tick_fontsize, bar_color
        styling
    figsize : tuple, default (10, 8)
        Figure size in inches
    asc: bool, default False
        Display horizontal bars in ascending order        

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        Use fig.savefig("path.png", dpi=300, bbox_inches="tight") to save 
    """
    if 'feature' not in feat_imp.columns or 'importance' not in feat_imp.columns:
        raise ValueError("feat_imp must contain 'feature' and 'importance' columns")

    data = feat_imp.head(top_n).copy()
    data = data.sort_values('importance', ascending=asc)

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=data,
        y='feature',
        x='importance',
        color=bar_color,
        ax=ax
    )

    # Styling 
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=title_fontsize, pad=20, color="#708090")
    ax.set_xlabel("Importance", fontsize=label_fontsize)
    ax.set_ylabel("")
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    plt.tight_layout()
    return fig, ax
