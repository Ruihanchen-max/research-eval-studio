"""Metric functions and grouped evaluation for ResearchEval Studio."""

from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[Iterable[float], np.ndarray]


def _validated_pair(y_true: ArrayLike, y_pred: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Convert inputs to finite one-dimensional float arrays of equal length."""
    try:
        true_values = np.asarray(y_true, dtype=np.float64)
        pred_values = np.asarray(y_pred, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true 和 y_pred 必须是数值序列。") from exc

    if true_values.ndim != 1 or pred_values.ndim != 1:
        raise ValueError("y_true 和 y_pred 必须是一维序列。")
    if true_values.size == 0 or pred_values.size == 0:
        raise ValueError("空数据无法计算评价指标。")
    if true_values.shape != pred_values.shape:
        raise ValueError("y_true 和 y_pred 的样本数量必须相同。")
    if not np.isfinite(true_values).all() or not np.isfinite(pred_values).all():
        raise ValueError("y_true 和 y_pred 不能包含 NaN 或无限值。")

    return true_values, pred_values


def _finite_metric(value: float, name: str) -> float:
    if not np.isfinite(value):
        raise ValueError(f"{name} 计算发生数值溢出，无法提供有效结果。")
    return float(value)


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean absolute error: mean(abs(y_true - y_pred))."""
    true_values, pred_values = _validated_pair(y_true, y_pred)
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.mean(np.abs(true_values - pred_values))
    return _finite_metric(float(value), "MAE")


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return root mean squared error: sqrt(mean((y_true - y_pred) ** 2))."""
    true_values, pred_values = _validated_pair(y_true, y_pred)
    with np.errstate(over="ignore", invalid="ignore"):
        differences = true_values - pred_values
        value = np.sqrt(np.mean(np.square(differences)))
    return _finite_metric(float(value), "RMSE")


def lin_concordance_correlation_coefficient(
    y_true: ArrayLike, y_pred: ArrayLike
) -> float:
    """Return Lin's CCC using population moments after common-scale normalization.

    CCC = 2 * cov(y_true, y_pred) /
          (var(y_true) + var(y_pred) + (mean(y_true) - mean(y_pred)) ** 2).

    Dividing both arrays by their shared maximum absolute value leaves CCC
    unchanged while avoiding squaring extremely small or large raw values.
    """
    true_values, pred_values = _validated_pair(y_true, y_pred)
    if true_values.size < 2:
        raise ValueError("CCC 至少需要 2 个样本；当前组只有 1 个。")

    scale = max(float(np.max(np.abs(true_values))), float(np.max(np.abs(pred_values))))
    if scale == 0.0:
        raise ValueError("两序列完全相同且为常数，CCC 分母为 0，无法定义。")

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled_true = true_values / scale
        scaled_pred = pred_values / scale
        true_mean = np.mean(scaled_true)
        pred_mean = np.mean(scaled_pred)
        centered_true = scaled_true - true_mean
        centered_pred = scaled_pred - pred_mean
        covariance = np.mean(centered_true * centered_pred)
        true_variance = np.mean(np.square(centered_true))
        pred_variance = np.mean(np.square(centered_pred))
        mean_difference = true_mean - pred_mean
        denominator = true_variance + pred_variance + np.square(mean_difference)

        if not np.isfinite(denominator) or not np.isfinite(covariance):
            raise ValueError("CCC 计算发生数值溢出，无法提供有效结果。")
        if denominator == 0.0:
            raise ValueError("两序列完全相同且为常数，CCC 分母为 0，无法定义。")
        value = 2.0 * covariance / denominator

    return _finite_metric(float(value), "CCC")


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, object]:
    """Calculate metrics while preserving valid MAE/RMSE when CCC is undefined."""
    true_values, pred_values = _validated_pair(y_true, y_pred)
    mae = mean_absolute_error(true_values, pred_values)
    rmse = root_mean_squared_error(true_values, pred_values)

    ccc: Optional[float]
    ccc_note = ""
    try:
        ccc = lin_concordance_correlation_coefficient(true_values, pred_values)
    except ValueError as exc:
        ccc = None
        ccc_note = str(exc)

    return {
        "n_samples": int(true_values.size),
        "mae": mae,
        "rmse": rmse,
        "ccc": ccc,
        "ccc_note": ccc_note,
    }


def evaluate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Evaluate pooled records per model and separately per model/subject.

    The overall row pools every observation for a model. Subject rows are
    independent calculations; this function does not average subject metrics.
    Missing group labels and empty frames are rejected because pandas groupby
    would otherwise drop missing keys or produce a misleading empty result.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("评估数据必须是 pandas DataFrame。")

    required = {"model", "subject_id", "y_true", "y_pred"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("评估数据缺少字段：" + ", ".join(missing))
    if data.empty:
        raise ValueError("评估数据不包含任何数据行，无法生成指标。")

    for column in ("model", "subject_id"):
        labels = data[column].astype("string").str.strip()
        missing_mask = data[column].isna() | labels.eq("").fillna(False)
        missing_count = int(missing_mask.sum())
        if missing_count:
            raise ValueError(
                f"评估数据字段 {column} 包含 {missing_count} 个缺失或空白分组键；"
                "请先修正数据，不会跳过这些记录。"
            )

    rows = []
    for model, model_group in data.groupby("model", sort=True):
        pooled = compute_metrics(model_group["y_true"], model_group["y_pred"])
        rows.append(
            {
                "scope": "overall_pooled",
                "model": str(model),
                "subject_id": "",
                **pooled,
            }
        )

        for subject_id, subject_group in model_group.groupby("subject_id", sort=True):
            subject_metrics = compute_metrics(
                subject_group["y_true"], subject_group["y_pred"]
            )
            rows.append(
                {
                    "scope": "subject",
                    "model": str(model),
                    "subject_id": str(subject_id),
                    **subject_metrics,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "scope",
            "model",
            "subject_id",
            "n_samples",
            "mae",
            "rmse",
            "ccc",
            "ccc_note",
        ],
    )
