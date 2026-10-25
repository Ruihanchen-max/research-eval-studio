"""CSV serialization at the download boundary; never mutate analysis data."""

import csv
import unicodedata

import pandas as pd


FORMULA_PREFIXES = frozenset("=+-@＝＋－＠")


def _escape_spreadsheet_text(value):
    """Prefix risky text with an apostrophe, preserving every original character.

    Inspect beyond leading Unicode whitespace/control/format characters. Leading
    control/format characters are also protected on their own, including tabs,
    newlines, BOM, and zero-width characters. Numeric values are never escaped.
    """
    if not isinstance(value, str):
        return value

    leading_control = False
    for character in value:
        category = unicodedata.category(character)
        if character.isspace() or category in {"Cc", "Cf"}:
            leading_control |= category in {"Cc", "Cf"}
            continue
        if leading_control or character in FORMULA_PREFIXES:
            return "'" + value
        return value
    return "'" + value if leading_control else value


def spreadsheet_safe_copy(frame: pd.DataFrame) -> pd.DataFrame:
    """Protect a copy, including headers, for CSV or a table's CSV toolbar.

    Streamlit's dataframe toolbar exports its displayed values independently of
    the main download button, so the displayed copy needs the same protection.
    """
    protected = frame.map(_escape_spreadsheet_text)
    columns = []
    used = set()
    for column in frame.columns:
        base = _escape_spreadsheet_text(column)
        label = base
        suffix = 2
        while label in used:
            label = f"{base} [{suffix}]"
            suffix += 1
        columns.append(label)
        used.add(label)
    protected.columns = columns
    return protected


def export_results_csv(results: pd.DataFrame) -> bytes:
    """Return UTF-8 BOM CSV with quoted cells and protected textual values.

    Intended for spreadsheet viewing of evaluation results with fixed headers.
    CSV has no text-type metadata: downstream software may remove protection.
    """
    exported = spreadsheet_safe_copy(results)
    return exported.to_csv(
        index=False,
        float_format="%.17g",
        quoting=csv.QUOTE_ALL,
        doublequote=True,
        lineterminator="\r\n",
    ).encode("utf-8-sig")
