from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


JAPANESE_FONT_CANDIDATES = (
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "Yu Gothic",
    "Meiryo",
    "IPAexGothic",
    "IPAGothic",
)

JAPANESE_FONT_FILE_HINTS = (
    "notosanscjk",
    "notosansjp",
    "ipaexg",
    "ipag",
    "yugoth",
    "meiryo",
)


def _looks_like_japanese_font(path: str) -> bool:
    normalized = str(Path(path)).lower().replace("-", "").replace("_", "")
    return any(hint in normalized for hint in JAPANESE_FONT_FILE_HINTS)


def _register_system_japanese_fonts() -> set[str]:
    """Register Japanese fonts directly from system files, bypassing stale MPL caches."""
    registered: set[str] = set()
    for path in font_manager.findSystemFonts():
        if not _looks_like_japanese_font(path):
            continue
        try:
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
        except (OSError, RuntimeError, ValueError):
            continue
        if family:
            registered.add(family)
    return registered


def configure_matplotlib_font(enable_japanese_support: bool = False) -> str | None:
    """Use an installed Japanese-capable font only when explicitly enabled."""
    if not enable_japanese_support:
        return None

    available = {entry.name for entry in font_manager.fontManager.ttflist}
    if not any(candidate in available for candidate in JAPANESE_FONT_CANDIDATES):
        available.update(_register_system_japanese_fonts())

    for candidate in JAPANESE_FONT_CANDIDATES:
        if candidate not in available:
            continue
        plt.rcParams["font.family"] = "sans-serif"
        fallback = [
            name
            for name in plt.rcParams.get("font.sans-serif", [])
            if name != candidate
        ]
        plt.rcParams["font.sans-serif"] = [candidate, *fallback]
        plt.rcParams["axes.unicode_minus"] = False
        return candidate
    return None


def plot_regression(y_true, predictions: dict[str, np.ndarray], target: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    y_true_arr = np.asarray(y_true)
    all_values = [y_true_arr]
    for model_name, values in predictions.items():
        pred_arr = np.asarray(values)
        all_values.append(pred_arr)
        ax.scatter(y_true_arr, pred_arr, alpha=0.65, label=model_name)
    combined = np.concatenate(all_values)
    finite = combined[np.isfinite(combined)]
    if finite.size:
        lower = float(finite.min())
        upper = float(finite.max())
        if lower == upper:
            lower -= 1.0
            upper += 1.0
        ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_classification(y_true, predictions: dict[str, np.ndarray], target: str):
    y_true_arr = np.asarray(y_true)
    best_name = max(
        predictions,
        key=lambda name: accuracy_score(y_true_arr, np.asarray(predictions[name])),
    )
    best_pred = np.asarray(predictions[best_name])
    labels = np.unique(np.concatenate([y_true_arr, best_pred]))
    matrix = confusion_matrix(y_true_arr, best_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, aspect="auto")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Best confusion matrix: {best_name} ({target})")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    return fig
