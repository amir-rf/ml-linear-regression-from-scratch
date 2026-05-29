from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_regression_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R^2": r2,
    }

def plot_loss_curve(loss_history, output_path="figures/loss_curve.png"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Mean square error")
    plt.title("Training loss curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi = 300)
    plt.close()


def plot_prediction_vs_actual(y_true, y_pred, output_path="figures/predicted_vs_actual.png"):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title("Predicted vs actual values")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residual_histogram(y_true, y_pred, output_path="figures/residual_histogram.png"):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    residual = y_true - y_pred

    plt.figure(figsize=(8, 5))
    plt.hist(residual, bins=50)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


