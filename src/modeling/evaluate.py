"""
Evaluation utilities.

We report multiple regression metrics:
- R2 (good for relative fit)
- MAE (interpretable average error in BDT)
- RMSE (penalizes large errors)
"""

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Predicted target values.

    Returns:
        Metrics dataclass.
    """
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }
