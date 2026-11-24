"""Generate vector plots from measured probabilities and contributions."""

import os
from .paths import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from .constants import CORE_MODELS


def save_figures(out, y_test, scores, probabilities, shap_data):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    colors = ["#295a84", "#b45c32", "#478369", "#82659d", "#be8d27"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    for name, color in zip(CORE_MODELS, colors):
        fpr, tpr, _ = roc_curve(y_test, probabilities[name])
        ax.plot(fpr, tpr, label=f"{name} ({scores[name]['roc_auc']:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.savefig(figures / "roc_curves.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    for ax, name, cmap in zip(
        axes, ["Logistic Regression", "Hybrid MiniLM + XGBoost"], ["Blues", "Greens"]
    ):
        matrix = np.array(
            [[scores[name]["tn"], scores[name]["fp"]], [scores[name]["fn"], scores[name]["tp"]]]
        )
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=len(y_test[y_test == 0]))
        for (row, col), number in np.ndenumerate(matrix):
            ax.text(
                col,
                row,
                str(number),
                ha="center",
                va="center",
                color="white" if number > matrix.max() / 2 else "black",
                fontsize=12,
            )
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["No churn", "Churn"],
            yticklabels=["No churn", "Churn"],
            xlabel="Predicted class",
            ylabel="Observed class",
            title=name,
        )
    fig.savefig(figures / "confusion_matrices.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    names = ["Logistic\nregression", "XGBoost", "FNN", "Deep FNN", "MiniLM +\nXGBoost"]
    locations = np.arange(len(names))
    for offset, metric, color in [
        (-0.24, "accuracy", "#295a84"),
        (0, "f1", "#478369"),
        (0.24, "roc_auc", "#b8862e"),
    ]:
        bars = ax.bar(
            locations + offset,
            [scores[name][metric] for name in CORE_MODELS],
            width=0.23,
            label={"accuracy": "Accuracy", "f1": "Churn F1", "roc_auc": "ROC-AUC"}[metric],
            color=color,
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set(xticks=locations, xticklabels=names, ylim=(0, 1), ylabel="Held-out score")
    ax.legend(loc="upper center", ncol=3, frameon=False)
    fig.savefig(figures / "evaluation_metrics.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), constrained_layout=True)
    for ax, name in zip(axes, ["XGBoost", "Hybrid MiniLM + XGBoost"]):
        ranked = (
            shap_data[shap_data.model == name]
            .sort_values("mean_absolute_group_contribution")
            .tail(8)
        )
        ax.barh(ranked.feature, ranked.mean_absolute_group_contribution, color="#295a84")
        ax.set(title=name, xlabel="Mean absolute grouped SHAP value\n(raw log-odds margin)")
        ax.tick_params(axis="y", labelsize=8)
    fig.savefig(figures / "shap_contributions.pdf")
    plt.close(fig)
