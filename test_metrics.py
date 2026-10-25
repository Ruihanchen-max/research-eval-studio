from io import StringIO
import math

import numpy as np
import pandas as pd
import pytest

from metrics import (
    compute_metrics,
    evaluate_dataset,
    lin_concordance_correlation_coefficient,
    mean_absolute_error,
    root_mean_squared_error,
)


def test_mae_and_rmse_known_values():
    truth = [1.0, 2.0, 3.0]
    prediction = [1.0, 2.0, 4.0]

    assert mean_absolute_error(truth, prediction) == pytest.approx(1.0 / 3.0)
    assert root_mean_squared_error(truth, prediction) == pytest.approx(math.sqrt(1.0 / 3.0))


def test_ccc_known_value_uses_population_moments():
    truth = [1.0, 2.0, 3.0]
    prediction = [1.0, 2.0, 4.0]

    assert lin_concordance_correlation_coefficient(truth, prediction) == pytest.approx(
        6.0 / 7.0
    )


@pytest.mark.parametrize(
    "truth,prediction",
    [
        ([], []),
        ([1.0], [1.0, 2.0]),
        ([1.0, np.nan], [1.0, 2.0]),
        ([1.0, np.inf], [1.0, 2.0]),
        ([[1.0, 2.0]], [[1.0, 2.0]]),
    ],
)
def test_metric_functions_reject_invalid_inputs(truth, prediction):
    with pytest.raises(ValueError):
        mean_absolute_error(truth, prediction)


def test_single_sample_keeps_mae_rmse_and_explains_undefined_ccc():
    result = compute_metrics([2.0], [3.0])

    assert result["n_samples"] == 1
    assert result["mae"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(1.0)
    assert result["ccc"] is None
    assert "至少需要 2 个样本" in result["ccc_note"]


def test_identical_constant_sequences_have_undefined_ccc():
    with pytest.raises(ValueError, match="CCC 分母为 0"):
        lin_concordance_correlation_coefficient([2.0, 2.0], [2.0, 2.0])


def test_distinct_constant_sequences_have_zero_ccc():
    assert lin_concordance_correlation_coefficient([2.0, 2.0], [3.0, 3.0]) == 0.0


def test_identical_nonconstant_sequences_have_perfect_ccc():
    assert lin_concordance_correlation_coefficient([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_overall_pooled_metric_is_not_subject_average():
    data = pd.DataFrame(
        {
            "model": ["A", "A", "A", "A"],
            "subject_id": ["S1", "S1", "S1", "S2"],
            "time": [1, 2, 3, 1],
            "y_true": [0.0, 0.0, 0.0, 10.0],
            "y_pred": [0.0, 0.0, 0.0, 18.0],
        }
    )

    result = evaluate_dataset(data)
    overall_mae = result.loc[result["scope"] == "overall_pooled", "mae"].iloc[0]
    subject_maes = result.loc[result["scope"] == "subject", "mae"].tolist()

    assert overall_mae == pytest.approx(2.0)
    assert subject_maes == pytest.approx([0.0, 8.0])
    assert sum(subject_maes) / len(subject_maes) == pytest.approx(4.0)


def test_single_observation_and_missing_model_subject_pair_are_handled():
    data = pd.DataFrame(
        {
            "model": ["A", "B"],
            "subject_id": ["S01", "S02"],
            "time": [1.0, 1.0],
            "y_true": [1.0, 5.0],
            "y_pred": [1.5, 5.0],
        }
    )

    result = evaluate_dataset(data)
    singleton = result.loc[
        (result["scope"] == "subject")
        & (result["model"] == "A")
        & (result["subject_id"] == "S01")
    ].iloc[0]

    assert singleton["n_samples"] == 1
    assert singleton["mae"] == pytest.approx(0.5)
    assert singleton["rmse"] == pytest.approx(0.5)
    assert pd.isna(singleton["ccc"])
    assert "至少需要 2 个样本" in singleton["ccc_note"]
    assert not (
        (result["scope"] == "subject")
        & (result["model"] == "A")
        & (result["subject_id"] == "S02")
    ).any()


def test_float_precision_survives_csv_export_round_trip():
    result = compute_metrics([0.1, 0.2, 0.3], [0.1, 0.2, 0.30000000000000004])
    exported = pd.DataFrame([result]).to_csv(index=False, float_format="%.17g")
    restored = pd.read_csv(StringIO(exported), float_precision="round_trip").iloc[0]

    assert restored["mae"] == result["mae"]
    assert restored["rmse"] == result["rmse"]


def test_extreme_finite_values_raise_instead_of_returning_overflow():
    with pytest.raises(ValueError, match="溢出"):
        mean_absolute_error([1e308], [-1e308])
    with pytest.raises(ValueError, match="溢出"):
        root_mean_squared_error([1e308], [-1e308])
    assert lin_concordance_correlation_coefficient(
        [1e308, -1e308], [0.0, 0.0]
    ) == pytest.approx(0.0)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_ccc_is_stable_for_tiny_and_huge_finite_scales(scale):
    truth = [scale, 2.0 * scale, 3.0 * scale]
    prediction = [scale, 2.0 * scale, 4.0 * scale]

    assert lin_concordance_correlation_coefficient(truth, prediction) == pytest.approx(
        6.0 / 7.0
    )


def test_ccc_is_stable_for_subnormal_nonconstant_values():
    smallest = np.nextafter(0.0, 1.0)
    values = [smallest, 2.0 * smallest, 3.0 * smallest]

    assert lin_concordance_correlation_coefficient(values, values) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "column,bad_value",
    [
        ("model", None),
        ("model", " "),
        ("subject_id", None),
        ("subject_id", pd.NA),
        ("subject_id", " "),
    ],
)
def test_evaluate_dataset_rejects_missing_group_keys_without_dropping_rows(
    column, bad_value
):
    data = pd.DataFrame(
        {
            "model": ["A", "A"],
            "subject_id": ["S01", "S02"],
            "time": [1, 1],
            "y_true": [1.0, 2.0],
            "y_pred": [1.1, 2.1],
        }
    )
    data[column] = data[column].astype(object)
    data.loc[1, column] = bad_value

    with pytest.raises(ValueError, match=f"字段 {column}.*不会跳过"):
        evaluate_dataset(data)


def test_evaluate_dataset_rejects_empty_dataframe():
    empty = pd.DataFrame(
        columns=["model", "subject_id", "y_true", "y_pred"]
    )

    with pytest.raises(ValueError, match="不包含任何数据行"):
        evaluate_dataset(empty)
