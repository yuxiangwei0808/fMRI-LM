"""Metric summary helpers for instruction-tuning runs."""

import math


_ERROR_METRICS = {"mae", "mse", "rmse"}
_CLASSIFICATION_PRIORITY = (
    "accuracy",
    "balanced_accuracy",
    "auc",
    "roc_auc",
    "f1",
    "avg_field_accuracy",
    "overall_accuracy",
)
_REGRESSION_PRIORITY = ("mae", "rmse", "mse", "r2", "pearson", "accuracy")


def _finite_number(value):
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def _mean(values):
    values = [float(v) for v in values if _finite_number(v)]
    return sum(values) / len(values) if values else None


def _dataset_info(data_loader_val_test, data_name):
    if not data_loader_val_test or data_name not in data_loader_val_test:
        return None
    return data_loader_val_test[data_name].get("info")


def _score_from_metric(metric_name, value, is_regression):
    if not _finite_number(value):
        return None
    metric_key = str(metric_name).lower()
    score = -float(value) if is_regression and metric_key in _ERROR_METRICS else float(value)
    score_name = f"neg_{metric_key}" if is_regression and metric_key in _ERROR_METRICS else metric_key
    task_type = "regression" if is_regression else "classification"
    return {"score": score, "metric": score_name, "task_type": task_type}


def _single_task_record(metrics, data_info=None):
    if not isinstance(metrics, dict):
        return None
    is_regression = bool(data_info.get("is_regression", False)) if isinstance(data_info, dict) else False
    priority = _REGRESSION_PRIORITY if is_regression else _CLASSIFICATION_PRIORITY
    for metric_name in priority:
        if metric_name in metrics:
            record = _score_from_metric(metric_name, metrics[metric_name], is_regression)
            if record is not None:
                return record
    for metric_name, value in metrics.items():
        record = _score_from_metric(metric_name, value, is_regression)
        if record is not None:
            return record
    return None


def _multi_task_record(metrics, data_info=None):
    details = metrics.get("details") if isinstance(metrics, dict) else None
    if not isinstance(details, dict):
        return None

    target_records = []
    for target_name, target_metrics in details.items():
        target_info = data_info.get(target_name, {}) if isinstance(data_info, dict) else {}
        record = _single_task_record(target_metrics, target_info)
        if record is not None:
            target_records.append(record)

    avg_score = _mean([record["score"] for record in target_records])
    if avg_score is None:
        return None

    task_types = {record["task_type"] for record in target_records}
    task_type = task_types.pop() if len(task_types) == 1 else "mixed"
    return {"score": avg_score, "metric": "avg_target_score", "task_type": task_type}


def _dataset_record(metrics, data_info=None, open_ended=False):
    if not isinstance(metrics, dict) or not metrics:
        return None

    if open_ended:
        for metric_name in ("avg_field_accuracy", "overall_accuracy"):
            if metric_name in metrics:
                return _score_from_metric(metric_name, metrics[metric_name], is_regression=False)

    if "details" in metrics:
        record = _multi_task_record(metrics, data_info)
        if record is not None:
            return record

    if "avg_accuracy" in metrics:
        return _score_from_metric("avg_accuracy", metrics["avg_accuracy"], is_regression=False)

    return _single_task_record(metrics, data_info)


def summarize_best_dataset_scores(best_metrics_per_dataset, data_loader_val_test=None, best_epoch_per_dataset=None, open_ended=False):
    """Summarize the average of each dataset's own best validation score.

    Scores are normalized to a higher-is-better direction. For regression error
    metrics such as MAE/MSE/RMSE, the stored score is the negative error.
    """
    best_epoch_per_dataset = best_epoch_per_dataset or {}
    per_dataset = {}
    all_scores = []
    regression_scores = []
    classification_scores = []

    for data_name, metrics in best_metrics_per_dataset.items():
        data_info = _dataset_info(data_loader_val_test, data_name)
        record = _dataset_record(metrics, data_info=data_info, open_ended=open_ended)
        if record is None:
            continue
        record = dict(record)
        record["epoch"] = best_epoch_per_dataset.get(data_name)
        per_dataset[data_name] = record
        all_scores.append(record["score"])
        if record["task_type"] == "regression":
            regression_scores.append(record["score"])
        elif record["task_type"] == "classification":
            classification_scores.append(record["score"])

    return {
        "avg_best_per_dataset_metric": _mean(all_scores),
        "avg_best_per_dataset_metric_regression": _mean(regression_scores),
        "avg_best_per_dataset_metric_classification": _mean(classification_scores),
        "best_per_dataset_scores": per_dataset,
    }
