"""CSV parsing and schema/data-quality validation."""

import warnings
from typing import List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("model", "subject_id", "time", "y_true", "y_pred")
NUMERIC_COLUMNS = ("time", "y_true", "y_pred")
RECORD_KEY = ("model", "subject_id", "time")
NUMERIC_MISSING_TOKENS = frozenset(
    {"na", "n/a", "null", "nan", "none", "#na", "#n/a"}
)

# Conservative limits for the public demo. The file-size limit is also set in
# .streamlit/config.toml so oversized uploads are rejected before parsing.
MAX_UPLOAD_SIZE_MB = 10
MAX_DATA_ROWS = 50_000
MAX_MODEL_SUBJECT_GROUPS = 2_000


class DataValidationError(ValueError):
    """Raised when uploaded data cannot be safely evaluated."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("；".join(errors))


def validate_upload_size(size_bytes: int) -> None:
    """Reject uploaded files above the app's configured size limit."""
    if size_bytes > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise DataValidationError(
            [
                f"文件大小超过 {MAX_UPLOAD_SIZE_MB} MB 限制；"
                "请缩小文件或先筛选需要评估的数据。"
            ]
        )


def validate_resource_limits(data: pd.DataFrame) -> None:
    """Bound row and group counts for predictable demo-app resource use."""
    if len(data) > MAX_DATA_ROWS:
        raise DataValidationError(
            [
                f"数据行数为 {len(data):,}，超过演示版上限 {MAX_DATA_ROWS:,}；"
                "请先筛选数据后重试。"
            ]
        )

    group_count = int(data[["model", "subject_id"]].drop_duplicates().shape[0])
    if group_count > MAX_MODEL_SUBJECT_GROUPS:
        raise DataValidationError(
            [
                f"模型与被试组合数为 {group_count:,}，超过演示版上限 "
                f"{MAX_MODEL_SUBJECT_GROUPS:,}；请先筛选数据后重试。"
            ]
        )


def read_csv(source) -> pd.DataFrame:
    """Read UTF-8 CSVs, preserving identifiers and rejecting malformed rows."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            return pd.read_csv(
                source,
                dtype={"model": "string", "subject_id": "string"},
                encoding="utf-8-sig",
                keep_default_na=False,
                index_col=False,
                on_bad_lines="error",
            )
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(["CSV 文件为空或缺少表头。"]) from exc
    except UnicodeDecodeError as exc:
        raise DataValidationError(
            ["无法按 UTF-8 解码 CSV；请将文件另存为 UTF-8 或 UTF-8 BOM 编码后重试。"]
        ) from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(
            ["CSV 格式错误：请检查引号是否闭合，以及每行字段数是否与表头一致。"]
        ) from exc
    except pd.errors.ParserWarning as exc:
        raise DataValidationError(
            ["CSV 列数与表头不一致；请检查是否存在多余分隔符或字段。"]
        ) from exc


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the required columns, preserving extra columns."""
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError(["输入必须是 CSV 表格数据。"])

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise DataValidationError(
            ["缺少必需字段：" + ", ".join(missing_columns)]
        )
    if data.empty:
        raise DataValidationError(["CSV 不包含任何数据行。"])

    cleaned = data.copy()
    errors = []

    for column in ("model", "subject_id"):
        blank_mask = cleaned[column].isna()
        blank_mask = blank_mask | (
            cleaned[column].astype("string").str.strip().eq("").fillna(False)
        )
        count = int(blank_mask.sum())
        if count:
            errors.append(f"字段 {column} 包含 {count} 个缺失或空白值。")
        cleaned[column] = cleaned[column].astype("string").str.strip()

    for column in NUMERIC_COLUMNS:
        original = cleaned[column]
        text_values = original.astype("string").str.strip()
        normalized_text = text_values.str.casefold()
        blank_mask = (
            original.isna()
            | text_values.eq("").fillna(False)
            | normalized_text.isin(NUMERIC_MISSING_TOKENS).fillna(False)
        )
        numeric = pd.to_numeric(original, errors="coerce")
        invalid_mask = numeric.isna() & ~blank_mask

        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            examples = original.loc[invalid_mask].astype(str).head(3).tolist()
            errors.append(
                f"字段 {column} 包含 {invalid_count} 个非数值内容"
                f"（示例：{', '.join(examples)}）。"
            )

        blank_count = int(blank_mask.sum())
        if blank_count:
            errors.append(f"字段 {column} 包含 {blank_count} 个缺失值。")

        finite_values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        infinite_count = int((numeric.notna().to_numpy() & ~np.isfinite(finite_values)).sum())
        if infinite_count:
            errors.append(f"字段 {column} 包含 {infinite_count} 个无限值。")

        cleaned[column] = numeric

    duplicate_mask = cleaned.duplicated(subset=list(RECORD_KEY), keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        examples = (
            cleaned.loc[duplicate_mask, list(RECORD_KEY)]
            .drop_duplicates()
            .head(3)
            .astype(str)
            .agg(" / ".join, axis=1)
            .tolist()
        )
        errors.append(
            f"发现 {duplicate_count} 行重复记录；"
            "model、subject_id、time 组合必须唯一"
            + (f"（示例：{'; '.join(examples)}）。" if examples else "。")
        )

    if errors:
        raise DataValidationError(errors)

    return cleaned
