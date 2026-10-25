from pathlib import Path
import tomllib

from streamlit.testing.v1 import AppTest

import metrics
import pandas as pd
import validation
from validation import MAX_UPLOAD_SIZE_MB


def test_app_renders_synthetic_demo_and_public_demo_notice():
    app_test = AppTest.from_file("app.py").run()

    assert not app_test.exception
    assert any("ResearchEval Studio" in title.value for title in app_test.title)
    assert any("演示用途" in message.value for message in app_test.warning)
    assert any("有效记录" in item.label for item in app_test.metric)
    assert any("模型总体指标" in item.value for item in app_test.markdown)


def test_streamlit_upload_config_matches_application_limit():
    test_path = Path(__file__).resolve()
    repo_root = test_path.parent.parent if test_path.parent.name == "tests" else test_path.parent
    config_path = repo_root / ".streamlit" / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["server"]["maxUploadSize"] == MAX_UPLOAD_SIZE_MB
    assert config["browser"]["gatherUsageStats"] is False


def test_app_explains_metric_overflow_to_user(monkeypatch):
    def raise_metric_error(_data):
        raise ValueError("MAE 计算发生数值溢出，无法提供有效结果。")

    monkeypatch.setattr(metrics, "evaluate_dataset", raise_metric_error)

    app_test = AppTest.from_file("app.py").run()

    assert not app_test.exception
    assert any("无法计算评估指标" in message.value for message in app_test.error)


def test_all_table_csv_toolbars_get_protected_copies_and_selectors_keep_labels(monkeypatch):
    data = pd.DataFrame({
        "model": ["=model", "=model"], "subject_id": ["@subject", "@subject"],
        "time": [1, 2], "y_true": [1.0, 2.0], "y_pred": [2.0, 1.0],
        "=extra": ["=1+1", "ordinary"],
    })
    before = data.copy(deep=True)
    monkeypatch.setattr(validation, "read_csv", lambda _source: data)
    app_test = AppTest.from_file("app.py").run()

    assert not app_test.exception
    assert len(app_test.dataframe) == 3
    for table in app_test.dataframe:
        assert table.value.iloc[0, 0] == "'=model"
    assert "'=extra" in app_test.dataframe[0].value.columns
    assert app_test.dataframe[0].value.iloc[0]["'=extra"] == "'=1+1"
    assert app_test.selectbox[0].value == "=model"
    assert app_test.selectbox[1].value == "@subject"
    pd.testing.assert_frame_equal(data, before)
