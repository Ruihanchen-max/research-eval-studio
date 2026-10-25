# ResearchEval Studio v0.1

[中文说明](#中文说明) · [English](#english)

**A Reproducible Machine Learning Experiment Evaluation Toolkit**

![ResearchEval Studio demo interface](screenshots/research-eval-studio-demo.png)

![ResearchEval Studio metrics, export, and prediction curve](screenshots/research-eval-studio-results.png)

> The screenshot uses the bundled synthetic demo only. This project is a technical demonstration; it has not been deployed online and is not a validated clinical or research decision system.

## 中文说明

ResearchEval Studio 将一份预测结果 CSV 转换为经过数据检查、计算口径明确、可下载的回归评估结果。适合需要重复整理模型预测结果的科研人员、机器学习工程师和数据分析人员。项目内的 `demo.csv` 是人工构造的合成示例，不含真实科研数据。

### 功能

- 校验必需字段、数值、缺失/无限值和重复记录，并保留字符串被试编号中的前导零。
- 计算每个模型的整体拼接指标，以及每个模型、每个被试的独立指标。
- 展示数据摘要、表格和真实值/预测值曲线；导出完整精度 CSV。
- 用 pytest 和 Streamlit AppTest 验证关键指标、数据边界和界面行为。

整体指标是将同一模型的所有观测拼接后重新计算，不是分被试指标的平均值。当前不计算被试指标的平均值。

### 输入与输出

CSV 必须包含以下列：

| 字段 | 含义 | 要求 |
| --- | --- | --- |
| `model` | 模型名称 | 非空文本 |
| `subject_id` | 被试编号 | 非空文本；如 `001` 会保留前导零 |
| `time` | 时间点 | 有限数值 |
| `y_true` | 真实值 | 有限数值 |
| `y_pred` | 预测值 | 有限数值 |

`model + subject_id + time` 必须唯一。时间列不接受日期字符串。输出包含 `overall_pooled`（模型整体拼接）和 `subject`（分模型、分被试）两种 scope，字段包括样本数、MAE、RMSE、CCC 及未定义 CCC 的说明。

### 指标定义

设真实值为 `yᵢ`、预测值为 `pᵢ`、样本数为 `n`：

- **MAE** = `(1/n) Σ |yᵢ - pᵢ|`
- **RMSE** = `sqrt((1/n) Σ (yᵢ - pᵢ)²)`
- **Lin's CCC** = `2 cov(y,p) / [var(y) + var(p) + (mean(y)-mean(p))²]`

方差和协方差采用总体矩，分母为 `n`（`ddof=0`）。CCC 至少需要两个观测；相同常数序列的 CCC 未定义。此时 MAE/RMSE 仍会计算，CCC 留空并说明原因。

### 本地安装与运行（macOS）

需要 Python 3.12。请从项目根目录执行：

```bash
python3.12 -m venv .venv-py312
source .venv-py312/bin/activate
python -m pip install -r requirements.txt
python -m pip check
streamlit run app.py
```

应用会默认加载 `demo.csv`。退出时在终端按 `Control-C`，然后运行 `deactivate` 退出虚拟环境。

测试与代码导入检查：

```bash
python -m pytest -q
python -m compileall -q app.py metrics.py validation.py exporting.py tests
python -c "import app, metrics, validation, exporting; print('imports: ok')"
```

`requirements.txt` 通过 `-r requirements-py312.lock` 引用同一份完整固定依赖（包含 pytest），本地、CI 和公开部署入口使用相同版本。CI 在 Ubuntu / Python 3.12 上分别通过两个文件入口安装并运行测试；工作流尚未在远端执行。

未来部署时，入口选择 `app.py`、Python 3.12；上传限制已写入 `.streamlit/config.toml`。安装和部署步骤参考 [Streamlit 官方部署文档](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)。目前没有在线演示链接，也没有执行云端部署。

### CSV 导出安全

主下载在导出副本中对危险文本添加前导单引号。Streamlit 表格自带独立 CSV 下载工具栏，因此三个表格的展示副本（含额外列表头）也受同样保护；内部标签、分组、选择器和数值指标不变。检测 `= + - @` 及全角形式，识别其前面的 Unicode 空白和控制/格式字符；以控制/格式字符开头的文本也受保护。所有 CSV 单元格加双引号，内部双引号由 CSV 序列化器转义，原有字符不删除。普通编号如 `001`、`Model-1` 不变，文本 `-1` 会被保护，真正的负数指标仍是数值。额外列表头若在转义后重名，仅在展示/导出副本中追加编号以维持唯一。

CSV 不携带列类型，不同电子表格及再次另存/打开的行为存在差异；这不是通用安全保证。程序读取时会保留保护单引号，不应自动去除后再作为可信内容打开。电子表格仍可能把普通编号 `001` 转为数字，请按文本列导入。未实际验证 Excel/LibreOffice 中公式是否执行。参见 [OWASP CSV Injection](https://owasp.org/www-community/attacks/CSV_Injection)。

### 演示范围与数据提示

此项目仅用于技术演示。请使用模拟、公开或获得授权的数据。当前演示上限为单文件 10 MB、50,000 行和 2,000 个模型-被试组合；页面最多预览 1,000 行，长曲线最多均匀抽样显示 2,000 个点，指标仍用全部通过校验的记录计算。

本地运行时，应用在本机处理上传文件；如部署在 Streamlit Community Cloud，上传文件会传输到托管服务器处理。项目不提供敏感数据隔离、安全删除或隐私保护保证。请勿上传敏感、机密或未经授权的研究数据。工具不保证临床决策或科研结论。

当前只支持数值型回归预测，不支持分类指标、日期时间解析、置信区间、显著性检验、多用户权限、数据库、SLA 或大规模数据性能保证。

### 许可证与公开状态

本项目采用 [MIT License](LICENSE)。许可证允许在保留版权和许可声明的前提下使用、修改和分发代码；软件按“现状”提供，不包含适用于特定科研或临床用途的保证。

## English

ResearchEval Studio turns a prediction-result CSV into validated, clearly scoped regression metrics and a downloadable result table. It is intended for researchers, machine-learning engineers, and analysts who repeatedly organize model predictions. The bundled `demo.csv` contains synthetic examples only; it has no real research records.

### What it does

- Validates required columns, numeric values, missing/non-finite values, and duplicate records while preserving string identifiers such as `001`.
- Calculates metrics pooled across all observations for each model and separately for each model-subject pair.
- Shows a data summary, result tables, and true-versus-predicted curves; exports full-precision CSV.
- Uses pytest and Streamlit AppTest for core metrics, input boundaries, and interface behavior.

A pooled model score is recalculated from all observations for that model. It is not the average of subject-level scores; that average is not implemented.

### Input and output

The CSV must contain `model`, `subject_id`, `time`, `y_true`, and `y_pred`. `model` and `subject_id` must be non-empty text; the remaining fields must be finite numeric values. The combination `model + subject_id + time` must be unique. Date strings are not supported in `time`.

The export includes `overall_pooled` rows and per-model/per-subject `subject` rows, with sample count, MAE, RMSE, CCC, and a note when CCC is undefined. Variance and covariance use population moments (`ddof=0`). CCC requires at least two observations and is undefined for identical constant sequences.

### Run locally on macOS

Use Python 3.12 and run these commands from the project root:

```bash
python3.12 -m venv .venv-py312
source .venv-py312/bin/activate
python -m pip install -r requirements.txt
python -m pip check
streamlit run app.py
```

The app loads `demo.csv` by default. Press `Control-C` in the terminal to stop it, then run `deactivate`.

Run checks with `python -m pytest -q`, `python -m compileall -q app.py metrics.py validation.py exporting.py tests`, and `python -c "import app, metrics, validation, exporting; print('imports: ok')"`.

`requirements.txt` includes `requirements-py312.lock` using `-r`, so local installation, CI, and the deployment entry point use one complete pinned dependency set (including pytest). The Ubuntu / Python 3.12 CI matrix installs through each entry point and runs the tests; the remote workflow has not yet been executed.

For future deployment, select `app.py` and Python 3.12. Upload limits are configured in `.streamlit/config.toml`. See the [official Streamlit deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy). There is no online demo, and cloud deployment has not been performed.

### CSV export safety

The main download prefixes risky text with an apostrophe on an export copy. All three table copies, including extra column headers, also receive protection because Streamlit has independent CSV download toolbars. Internal labels, grouping, selectors, and numeric metrics stay unchanged. Detection covers `= + - @`, full-width variants, leading Unicode whitespace, and control/format prefixes. CSV quoting preserves delimiters, quotes, and line breaks. Ordinary IDs such as `001` and `Model-1` are unchanged; text `-1` is protected, while negative numeric metrics remain numbers. Extra headers that collide after escaping receive a uniqueness suffix on the copy only.

CSV has no column-type metadata. Protection is not universal across spreadsheet software or save/reopen cycles. Programmatic readers retain the apostrophe; do not automatically strip it and treat the result as trusted. Import identifier columns as text to avoid spreadsheet conversion of `001`. Formula execution in Excel/LibreOffice has not been tested.

### Demo safety and limitations

This is a technical demo. Use synthetic, public, or authorized data only. The demo limits each file to 10 MB, the dataset to 50,000 rows, and model-subject combinations to 2,000. The page previews up to 1,000 rows and samples long curves to at most 2,000 points; metrics still use every validated row.

When run locally, uploaded files are processed on the local machine. If deployed to Streamlit Community Cloud, uploads are transmitted to the hosting server for processing. This project does not provide guarantees for sensitive-data isolation, secure deletion, or privacy protection. Do not upload sensitive, confidential, or unauthorized research data. The tool does not guarantee clinical decisions or research conclusions.

The current version supports numeric regression predictions only. It does not support classification metrics, date parsing, confidence intervals, significance testing, multi-user access control, databases, an SLA, or performance guarantees for large datasets.

### License and publication status

This project is released under the [MIT License](LICENSE). The license permits use, modification, and redistribution when the copyright and license notice are retained. The software is provided "as is" and carries no guarantee for a particular research or clinical use.
