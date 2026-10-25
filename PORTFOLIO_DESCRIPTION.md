# ResearchEval Studio — Portfolio Description

## 中文项目介绍

ResearchEval Studio 是一个面向科研人员、机器学习工程师和数据分析人员的预测结果评估演示工具。用户提供符合约定格式的 CSV，应用检查数据质量，按模型计算整体拼接指标，并按模型与被试分别计算指标；随后查看真实值/预测值曲线并下载完整精度结果 CSV。

项目旨在减少重复的手工 CSV 整理与指标汇总工作，并展示从科研代码到可复现小型软件的工程实践：独立的数据验证和指标模块、具有手算预期值的数学测试、虚拟环境与固定依赖、Streamlit 交互页面和 Python 3.12 自动化测试工作流。内置 `demo.csv` 仅含人工构造数据。

## English project description

ResearchEval Studio is a demonstration toolkit for researchers, machine-learning engineers, and data analysts who evaluate prediction results. Users provide a CSV in the documented schema. The app checks data quality, calculates pooled metrics per model and separate metrics per model-subject pair, displays true-versus-predicted curves, and exports a full-precision CSV result table.

The project reduces repetitive CSV cleanup and metric aggregation while demonstrating reproducible software engineering for research workflows: separated validation and metric modules, mathematical tests with hand-checkable expected values, isolated environments and pinned dependencies, a Streamlit interface, and a Python 3.12 automated test workflow. The bundled `demo.csv` contains synthetic data only.

## 核心技术 | Core technology

- Python 3.12
- Pandas and NumPy for tabular validation and metric calculations
- Streamlit for the interactive application
- pytest and Streamlit AppTest for automated checks
- Git and GitHub Actions workflow configuration for reproducibility

## 输入和输出 | Inputs and outputs

**Input:** a UTF-8 CSV with `model`, `subject_id`, `time`, `y_true`, and `y_pred`. Identifiers are text; the three numeric fields must contain finite values. The key `model + subject_id + time` must be unique. The current demo accepts at most 10 MB, 50,000 rows, and 2,000 model-subject groups.

**Output:** data-quality feedback, a bounded data preview, pooled model-level and separate subject-level MAE/RMSE/Lin's CCC, selected time-series curves, and a full-precision CSV containing the same two result scopes. Formula-like text is protected at export; internal grouping and numeric metrics remain unchanged.

The pooled model score is recalculated from all observations and must not be interpreted as the mean of subject-level scores. Subject-score averaging is not implemented.

## 解决的问题 | Problem addressed

Manual result aggregation can cause repeated work, inconsistent metric definitions, accidental input errors, and unclear grouping scope. This tool offers a small repeatable path from a validated prediction CSV to documented metrics and downloadable results. It supports numeric regression evaluation only; it does not determine whether a model or scientific conclusion is valid.

## 适用对象 | Potential users

- Research groups that repeatedly evaluate numeric prediction outputs.
- ML engineers and analysts who need a transparent CSV review and metric-export step.
- Small R&D teams exploring an internal workflow prototype before specifying custom requirements.

These are potential audiences only. No customers, paid engagements, testimonials, public deployment, or commercial outcomes are claimed.

## 可评估的定制方向 | Possible scoped customization

After agreeing on data definitions, metric formulas, outputs, data permissions, and acceptance criteria, the code could be adapted to:

- Map a documented CSV schema or additional grouping fields.
- Add explicitly defined regression measures or aggregation tables.
- Change chart labels, sorting, or report output formats.
- Integrate authorized upstream research-script outputs.
- Add regression tests and installation notes for a specified Python environment.

Any customization requires a separate assessment of data governance, expected row counts, target environment, and acceptance criteria. This project does not establish that such work has been contracted or delivered.

## 当前不支持 | Not currently supported

- Classification metrics, date/time parsing, confidence intervals, significance tests, or scientific interpretation.
- Login, roles, database, multi-tenant isolation, secure deletion, audit trails, or sensitive-data guarantees.
- Large-scale performance guarantees, SLA, supported customer environment, or managed cloud hosting.
- Clinical decision support or guarantees about research conclusions.

The tool is a local/demo MVP. If it is deployed to a cloud host, uploaded files are transmitted to that host for processing; users must not submit sensitive, confidential, or unauthorized data. No secure isolation or deletion behavior is claimed.

## 商业展示边界 | Commercial presentation boundary

The current code can serve as a free technical demonstration and a starting point for narrowly scoped custom development after requirements and data permissions are agreed. It is not a mature commercial product and has no verified paying customers or production deployments.
