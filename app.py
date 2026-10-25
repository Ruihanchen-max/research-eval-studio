"""Streamlit interface for ResearchEval Studio v0.1."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from exporting import export_results_csv, spreadsheet_safe_copy
from metrics import evaluate_dataset
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


DEMO_PATH = Path(__file__).with_name("demo.csv")
PREVIEW_ROWS = 1_000
MAX_CHART_POINTS = 2_000


def _display_metrics_table(frame: pd.DataFrame, include_subject: bool) -> pd.DataFrame:
    columns = ["model"]
    if include_subject:
        columns.append("subject_id")
    columns.extend(["n_samples", "mae", "rmse", "ccc", "ccc_note"])
    display = frame.loc[:, columns].copy()
    display = display.rename(
        columns={
            "model": "模型",
            "subject_id": "被试",
            "n_samples": "样本数",
            "mae": "MAE",
            "rmse": "RMSE",
            "ccc": "CCC",
            "ccc_note": "CCC 说明",
        }
    )
    return display


def main() -> None:
    st.set_page_config(
        page_title="ResearchEval Studio",
        page_icon="📊",
        layout="wide",
    )
    st.title("ResearchEval Studio")
    st.markdown("**A Reproducible Machine Learning Experiment Evaluation Toolkit**")
    st.caption(
        "面向表格预测结果的轻量评估工具：检查数据、计算模型指标、"
        "查看分组曲线并导出结果。 / Validate prediction CSVs, compare models, "
        "inspect subject-level results, and export metrics."
    )

    st.markdown("### 1. 导入数据 | Import data")
    st.caption(
        f"CSV 必须含 model、subject_id、time、y_true、y_pred 五列；"
        f"单文件上限 {MAX_UPLOAD_SIZE_MB} MB，最多 {MAX_DATA_ROWS:,} 行。"
        " / Upload one UTF-8 CSV using the required columns."
    )
    st.warning(
        "演示用途：请仅使用模拟、公开或已获授权的数据。"
        "本地运行时文件由本机应用处理；若部署到云端，上传内容会传输到托管服务器处理。"
        "请勿上传敏感、机密或未经授权的研究数据。工具不保证科研结论或临床决策。"
        " / Demo only. Use synthetic, public, or authorized data. Cloud uploads are "
        "processed by the hosting server. Do not upload sensitive or confidential data."
    )

    uploaded_file = st.file_uploader(
        "上传预测结果 CSV | Upload prediction CSV",
        type=["csv"],
        help=(
            f"最多 {MAX_UPLOAD_SIZE_MB} MB、{MAX_DATA_ROWS:,} 行和 "
            f"{MAX_MODEL_SUBJECT_GROUPS:,} 个模型-被试组合。"
        ),
    )

    try:
        if uploaded_file is None:
            source = read_csv(DEMO_PATH)
            st.caption(
                "当前展示内置合成数据 demo.csv（30 条记录；不含真实科研数据）。"
                "上传文件可切换数据。 / Synthetic demo data is loaded by default."
            )
        else:
            validate_upload_size(uploaded_file.size)
            source = read_csv(uploaded_file)
        data = validate_data(source)
        validate_resource_limits(data)
    except DataValidationError as exc:
        st.error("数据校验失败：\n" + "\n".join(f"- {error}" for error in exc.errors))
        st.stop()
    except Exception:
        st.error(
            "无法读取 CSV。请确认文件为有效的 UTF-8 CSV，检查分隔符、表头和引号后重试。"
            " / Could not read the CSV. Check its UTF-8 encoding, delimiter, header, and quotes."
        )
        st.stop()

    st.markdown("### 2. 数据预览 | Data preview")
    summary = st.columns(3)
    summary[0].metric("有效记录 | Rows", f"{len(data):,}")
    summary[1].metric("模型 | Models", f"{data['model'].nunique():,}")
    summary[2].metric("被试 | Subjects", f"{data['subject_id'].nunique():,}")
    if len(data) > PREVIEW_ROWS:
        st.caption(f"为保持页面响应，仅预览前 {PREVIEW_ROWS:,} 行；指标使用全部记录。")
    st.caption(
        "表格工具栏也能导出 CSV，因此表格副本中的危险文本会显示前导单引号；"
        "计算、分组和选择器使用原标签。 / Table copies protect formula-like text; "
        "analysis and selectors keep the original labels."
    )
    st.dataframe(
        spreadsheet_safe_copy(data.head(PREVIEW_ROWS)),
        use_container_width=True,
        hide_index=True,
    )

    try:
        results = evaluate_dataset(data)
    except ValueError as exc:
        st.error(f"无法计算评估指标：{exc}")
        st.stop()
    overall = results.loc[results["scope"] == "overall_pooled"]
    by_subject = results.loc[results["scope"] == "subject"]

    st.markdown("### 3. 模型总体指标 | Pooled model-level metrics")
    st.caption(
        "整体指标将每个模型的所有观测拼接后重新计算，不是分被试指标的平均值。"
        " / Each model is scored across all its observations; this is not the mean of subject scores."
    )
    st.dataframe(
        spreadsheet_safe_copy(_display_metrics_table(overall, include_subject=False)),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 4. 分被试指标 | Subject-level metrics")
    st.dataframe(
        spreadsheet_safe_copy(_display_metrics_table(by_subject, include_subject=True)),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 5. 导出结果 | Export")
    download_col, download_note = st.columns([1, 3])
    with download_col:
        downloaded = st.download_button(
            label="下载评估结果 CSV | Download CSV",
            data=export_results_csv(results),
            file_name="research_eval_results.csv",
            mime="text/csv",
            help="数值保留完整精度；可能被识别为公式的文本会添加前导单引号。",
        )
    with download_note:
        st.caption(
            "导出文件包含整体拼接行和分被试行，与上方结果表对应；"
            "浮点数以完整精度保存。危险文本会添加前导单引号，内部标签不变。"
            " / Same metric values; formula-like text is prefixed with an apostrophe for export."
        )
        if downloaded:
            st.success("评估结果 CSV 已下载。 / Results CSV downloaded.")

    st.markdown("### 6. 预测曲线 | Prediction curves")
    st.caption("按时间点排序显示真实值与预测值。 / Values are sorted by time.")
    models = sorted(data["model"].astype(str).unique().tolist())
    model_column, subject_column = st.columns(2)
    with model_column:
        selected_model = st.selectbox("选择模型 | Model", models)
    model_data = data.loc[data["model"].astype(str) == selected_model]
    subjects = sorted(model_data["subject_id"].astype(str).unique().tolist())
    with subject_column:
        selected_subject = st.selectbox("选择被试 | Subject", subjects)
    series = model_data.loc[
        model_data["subject_id"].astype(str) == selected_subject,
        ["time", "y_true", "y_pred"],
    ].sort_values("time")
    if len(series) > MAX_CHART_POINTS:
        sample_indices = np.unique(
            np.linspace(0, len(series) - 1, MAX_CHART_POINTS, dtype=int)
        )
        series = series.iloc[sample_indices]
        st.caption(
            f"曲线为保持响应从 {len(model_data.loc[model_data['subject_id'].astype(str) == selected_subject]):,} "
            f"个观测中均匀抽取 {len(series):,} 点；表格指标仍使用全部观测。"
        )
    chart_data = series.set_index("time").rename(
        columns={"y_true": "真实值", "y_pred": "预测值"}
    )
    st.line_chart(
        chart_data,
        use_container_width=True,
        height=360,
        y_label="数值 | Value",
    )


if __name__ == "__main__":
    main()
