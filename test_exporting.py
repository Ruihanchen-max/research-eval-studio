import csv
from io import BytesIO, StringIO

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from exporting import export_results_csv, spreadsheet_safe_copy
from metrics import evaluate_dataset
from validation import read_csv, validate_data


def exported_cells(value):
    content = export_results_csv(pd.DataFrame({"model": [value]}))
    return list(csv.reader(StringIO(content.decode("utf-8-sig"))))


@pytest.mark.parametrize(
    "value",
    [
        "=1+1", "+SUM(1,2)", "-1+2", "@SUM(1,2)",
        "   =1+1", "\t=1+1", " \r\n+1", "\x00-1", "\ufeff@x",
        "\u200b=1+1", "\u00a0\u2003=1+1", "\tplain", "\nplain",
        "＝1+1", "＋1", "－1", "＠x", '=HYPERLINK("example","label")',
        '=1,"\n=2',
    ],
)
def test_formula_like_text_is_prefixed_without_deleting_characters(value):
    assert exported_cells(value) == [["model"], ["'" + value]]


@pytest.mark.parametrize(
    "value",
    ["Model A", "Model-1", "001", "participant-x", "NA", "", "  model", "'literal"],
)
def test_ordinary_text_and_identifiers_are_unchanged(value):
    assert exported_cells(value) == [["model"], [value]]


def test_delimiters_quotes_and_newlines_cannot_create_additional_cells():
    value = 'Model,one;"quoted"\n=1+1'
    assert exported_cells(value) == [["model"], [value]]


def test_table_toolbar_copy_protects_headers_and_values_without_mutating_input():
    source = pd.DataFrame({" =extra": ["\ufeff@value"], "metric": [-0.5]})
    before = source.copy(deep=True)
    protected = spreadsheet_safe_copy(source)
    assert protected.columns.tolist() == ["' =extra", "metric"]
    assert protected.iloc[0, 0] == "'\ufeff@value"
    assert protected.iloc[0, 1] == -0.5
    assert_frame_equal(source, before)


def test_escaped_extra_headers_remain_unique_for_streamlit_tables():
    frame = pd.DataFrame({"=extra": [1], "'=extra": [2], "'=extra [2]": [3]})
    protected = spreadsheet_safe_copy(frame)
    assert protected.columns.is_unique
    assert protected.iloc[0].tolist() == [1, 2, 3]
    assert frame.columns.tolist() == ["=extra", "'=extra", "'=extra [2]"]


def test_export_preserves_grouping_metric_values_and_source_frame():
    source = pd.DataFrame(
        {
            "model": ["=model", "=model", "Model-1", "Model-1"],
            "subject_id": ["@subject", "@subject", "001", "001"],
            "time": [1, 2, 1, 2],
            "y_true": [1.0, 2.0, 1.0, 2.0],
            "y_pred": [2.0, 1.0, 1.0, 2.0000000000000004],
        }
    )
    source_before = source.copy(deep=True)
    results = evaluate_dataset(validate_data(source))
    results_before = results.copy(deep=True)
    content = export_results_csv(results)
    restored = pd.read_csv(
        BytesIO(content), keep_default_na=False,
        dtype={"model": "string", "subject_id": "string"},
        float_precision="round_trip",
    )

    assert content.startswith(b"\xef\xbb\xbf")
    assert_frame_equal(source, source_before)
    assert_frame_equal(results, results_before)
    assert set(results["model"]) == {"=model", "Model-1"}
    assert set(restored["model"]) == {"'=model", "Model-1"}
    assert "'@subject" in restored["subject_id"].tolist()
    assert "001" in restored["subject_id"].tolist()
    numeric = ["n_samples", "mae", "rmse", "ccc"]
    assert_frame_equal(restored[numeric], results[numeric], check_dtype=False)
    assert (restored["ccc"] < 0).any()


def test_demo_export_has_same_results_and_undefined_ccc_note_is_preserved():
    results = evaluate_dataset(validate_data(read_csv("demo.csv")))
    content = export_results_csv(results)
    restored = pd.read_csv(
        BytesIO(content), keep_default_na=False, float_precision="round_trip"
    )
    assert_frame_equal(restored, results, check_dtype=False)

    single = evaluate_dataset(pd.DataFrame({
        "model": ["A"], "subject_id": ["001"], "y_true": [1.0], "y_pred": [1.0],
    }))
    cells = list(csv.DictReader(StringIO(export_results_csv(single).decode("utf-8-sig"))))
    assert cells[1]["subject_id"] == "001"
    assert cells[1]["ccc"] == ""
    assert cells[1]["ccc_note"] == single.iloc[1]["ccc_note"]
