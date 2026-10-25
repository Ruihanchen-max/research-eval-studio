from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import pytest

from validation import (
    MAX_DATA_ROWS,
    MAX_MODEL_SUBJECT_GROUPS,
    MAX_UPLOAD_SIZE_MB,
    DataValidationError,
    read_csv,
    validate_data,
    validate_resource_limits,
    validate_upload_size,
)


def valid_data():
    return pd.DataFrame(
        {
            "model": ["Model A", "Model A"],
            "subject_id": ["S01", "S01"],
            "time": [1, 2],
            "y_true": [0.5, 1.0],
            "y_pred": [0.4, 0.9],
        }
    )


def test_valid_data_is_normalized_and_keeps_extra_columns():
    source = valid_data()
    source["note"] = ["first", "second"]

    cleaned = validate_data(source)

    assert cleaned["model"].tolist() == ["Model A", "Model A"]
    assert cleaned["time"].tolist() == [1.0, 2.0]
    assert cleaned["note"].tolist() == ["first", "second"]


def test_missing_required_fields_are_reported():
    with pytest.raises(DataValidationError, match="缺少必需字段"):
        validate_data(pd.DataFrame({"model": ["A"]}))


@pytest.mark.parametrize("column", ["time", "y_true", "y_pred"])
@pytest.mark.parametrize(
    "bad_value,expected",
    [
        ("not-a-number", "非数值内容"),
        (None, "缺失值"),
        (np.nan, "缺失值"),
        (float("inf"), "无限值"),
        (float("-inf"), "无限值"),
    ],
)
def test_invalid_numeric_values_are_reported(column, bad_value, expected):
    source = valid_data()
    source[column] = source[column].astype(object)
    source.loc[0, column] = bad_value

    with pytest.raises(DataValidationError, match=expected):
        validate_data(source)


def test_duplicate_model_subject_time_is_rejected():
    source = valid_data()
    source.loc[1, "time"] = 1

    with pytest.raises(DataValidationError, match="重复记录"):
        validate_data(source)


def test_blank_subject_is_rejected():
    source = valid_data()
    source.loc[0, "subject_id"] = " "

    with pytest.raises(DataValidationError, match="subject_id"):
        validate_data(source)


def test_csv_reading_preserves_string_subject_ids_and_leading_zeroes():
    source = StringIO(
        "model,subject_id,time,y_true,y_pred\n"
        "Model 1,001,1,2.0,2.1\n"
        "Model 1,participant-x,2,3.0,3.1\n"
    )

    cleaned = validate_data(read_csv(source))

    assert cleaned["subject_id"].tolist() == ["001", "participant-x"]


def test_csv_reading_preserves_na_like_identifier_text():
    source = StringIO(
        "model,subject_id,time,y_true,y_pred\n"
        "NA,NULL,1,2.0,2.1\n"
        "N/A,NA,2,3.0,3.1\n"
    )

    cleaned = validate_data(read_csv(source))

    assert cleaned["model"].tolist() == ["NA", "N/A"]
    assert cleaned["subject_id"].tolist() == ["NULL", "NA"]


def test_utf8_bom_csv_is_supported():
    content = (
        "model,subject_id,time,y_true,y_pred\nA,S01,1,2.0,2.1\n"
    ).encode("utf-8-sig")

    cleaned = validate_data(read_csv(BytesIO(content)))

    assert cleaned.loc[0, "model"] == "A"
    assert cleaned.loc[0, "subject_id"] == "S01"


@pytest.mark.parametrize("missing_token", ["NA", "NULL", "N/A", "NaN"])
def test_na_like_tokens_are_missing_in_numeric_columns(missing_token):
    source = StringIO(
        "model,subject_id,time,y_true,y_pred\n"
        f"A,S01,1,2.0,{missing_token}\n"
    )

    with pytest.raises(DataValidationError, match="y_pred 包含 1 个缺失值"):
        validate_data(read_csv(source))


@pytest.mark.parametrize(
    "malformed_csv,expected_message",
    [
        (
            'model,subject_id,time,y_true,y_pred\nA,"S01,1,2,3\n',
            "CSV 格式错误",
        ),
        (
            "model,subject_id,time,y_true,y_pred\nA,S01,1,2,3,extra\n",
            "CSV 列数与表头不一致",
        ),
    ],
)
def test_malformed_csv_is_rejected_with_actionable_message(
    malformed_csv, expected_message
):
    with pytest.raises(DataValidationError, match=expected_message):
        read_csv(StringIO(malformed_csv))


def test_unsupported_csv_encoding_has_actionable_message():
    content = (
        b"model,subject_id,time,y_true,y_pred\n"
        b"A,S01,1,2,\xff\n"
    )

    with pytest.raises(DataValidationError, match="另存为 UTF-8"):
        read_csv(BytesIO(content))


def test_missing_columns_from_csv_are_reported():
    with pytest.raises(DataValidationError, match="缺少必需字段"):
        validate_data(read_csv(StringIO("model\nA\n")))


def test_short_csv_row_reports_missing_numeric_value():
    source = StringIO(
        "model,subject_id,time,y_true,y_pred\nA,S01,1,2.0\n"
    )

    with pytest.raises(DataValidationError, match="y_pred 包含 1 个缺失值"):
        validate_data(read_csv(source))


def test_empty_and_header_only_csvs_have_clear_errors():
    with pytest.raises(DataValidationError, match="CSV 文件为空"):
        read_csv(StringIO(""))

    header_only = read_csv(StringIO(",".join(
        ["model", "subject_id", "time", "y_true", "y_pred"]
    )))
    with pytest.raises(DataValidationError, match="不包含任何数据行"):
        validate_data(header_only)


def test_upload_size_limit_accepts_boundary_and_rejects_oversized_file():
    validate_upload_size(MAX_UPLOAD_SIZE_MB * 1024 * 1024)

    with pytest.raises(DataValidationError, match="超过 10 MB 限制"):
        validate_upload_size(MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1)


def test_resource_limit_rejects_too_many_rows():
    data = pd.DataFrame(
        {
            "model": ["A"] * (MAX_DATA_ROWS + 1),
            "subject_id": ["S01"] * (MAX_DATA_ROWS + 1),
        }
    )

    with pytest.raises(DataValidationError, match="超过演示版上限"):
        validate_resource_limits(data)


def test_resource_limit_rejects_too_many_model_subject_groups():
    data = pd.DataFrame(
        {
            "model": ["A"] * (MAX_MODEL_SUBJECT_GROUPS + 1),
            "subject_id": [
                f"S{index:04d}" for index in range(MAX_MODEL_SUBJECT_GROUPS + 1)
            ],
        }
    )

    with pytest.raises(DataValidationError, match="模型与被试组合数"):
        validate_resource_limits(data)
