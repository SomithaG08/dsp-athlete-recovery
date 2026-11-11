"""Run the project's Great Expectations checkpoints on one dataframe.

Used by the Airflow ingestion DAG and the Streamlit validation page, so both apply
exactly the same rules. The rules themselves live in the GX suites
(see validation/build_suites.py); this module only runs them and reads the results.

Two passes (see build_suites.py for why):
  1. athlete_schema_checkpoint on the raw frame   -> schema + type errors
  2. athlete_checkpoint on the typed frame        -> completeness, validity, consistency
"""
import os
import shutil
from pathlib import Path

import pandas as pd

from scripts.data_quality import calculate_criticality, categorize_expectation, summarize_validation_errors

DEFAULT_GX_DIR = Path(__file__).resolve().parents[1] / "gx" / "gx"
SCHEMA_CHECKPOINT = "athlete_schema_checkpoint"
CONTENT_CHECKPOINT = "athlete_checkpoint"
NUMERIC_COLUMNS = [
    "Athlete_ID", "Day", "Day_of_Week", "Week", "Age", "Training_Duration_Min",
    "Training_Intensity", "Sleep_Duration_Hours", "Caffeine_Intake_mg", "Resting_Heart_Rate",
    "HRV_ms", "Mood_Score", "Muscle_Soreness", "Energy_Level", "Recovery_Score",
]
_CONTEXTS = {}


# Every run adds a result to the GX validation store, and Data Docs re-reads the whole store to
# rebuild its index. Left unchecked, validation slows from ~2 s to over a minute after a few
# hundred files, so only the most recent results are kept (older ones are moved, not deleted).
KEEP_VALIDATIONS = int(os.getenv("GX_KEEP_VALIDATIONS", "50"))


def prune_validation_history(context_root, keep=KEEP_VALIDATIONS):
    """Move all but the newest `keep` results per suite to uncommitted/validations_archive."""
    store = Path(context_root) / "uncommitted" / "validations"
    archive = Path(context_root) / "uncommitted" / "validations_archive"
    if keep <= 0 or not store.is_dir():
        return 0
    moved = 0
    for suite_dir in store.iterdir():
        if not suite_dir.is_dir():
            continue
        runs = sorted(p for p in suite_dir.glob("*/*") if p.is_dir())  # <batch>/<run time>, sortable
        for old in runs[:-keep]:
            target = archive / old.relative_to(store)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(target))
                moved += 1
            except (OSError, shutil.Error):
                continue  # a parallel ingestion run already moved it
    return moved


def get_context(context_root_dir=None):
    import great_expectations as gx
    root = str(context_root_dir or os.getenv("GX_CONTEXT_ROOT_DIR", DEFAULT_GX_DIR))
    if root not in _CONTEXTS:
        _CONTEXTS[root] = gx.get_context(context_root_dir=root)
    return _CONTEXTS[root]


def _to_number(value):
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value          # keep the text so the type expectation can flag this row
    return value


def prepare_frames(df: pd.DataFrame):
    """Return (schema_frame, content_frame).

    schema_frame keeps unparseable text inside numeric columns (row-level type errors).
    content_frame converts numeric columns to numbers; unparseable values become null.
    """
    schema_frame = df.copy()
    content_frame = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].dtype == object:
            schema_frame[col] = df[col].map(_to_number)
            content_frame[col] = pd.to_numeric(df[col], errors="coerce")
    return schema_frame.reset_index(drop=True), content_frame.reset_index(drop=True)


def docs_url(context, result_identifier):
    """Turn the file:// Data Docs link of one validation run into an nginx URL."""
    base = os.getenv("DATA_DOCS_BASE_URL", "http://localhost:8088").rstrip("/")
    try:
        urls = context.get_docs_sites_urls(resource_identifier=result_identifier)
        site_url = urls[0]["site_url"] if urls else ""
    except Exception:
        site_url = ""
    marker = "local_site/"
    if marker in site_url:
        return f"{base}/{site_url.split(marker, 1)[1]}"
    return base + "/"


def _read_results(checkpoint_result):
    identifier, validation = next(iter(checkpoint_result.run_results.items()))
    return identifier, validation


def run_validation(df: pd.DataFrame, context_root_dir=None) -> dict:
    """Validate one file. Returns invalid rows, error counts per category and report links."""
    context = get_context(context_root_dir)
    prune_validation_history(context_root_dir or os.getenv("GX_CONTEXT_ROOT_DIR", DEFAULT_GX_DIR))
    schema_frame, content_frame = prepare_frames(df)
    total = len(content_frame)
    invalid = set()
    row_errors = {}
    error_counts = {}
    category_counts = {"completeness": 0, "validity": 0, "consistency": 0, "schema": 0, "type": 0}
    missing_columns = 0
    reports = {}
    evaluated = passed = 0
    # (row, column) cells with text in a numeric column. The content pass turns that text into null,
    # so the null rule would flag the same cell again: it is reported once, as a type error.
    type_cells = set()

    for checkpoint_name, frame in ((SCHEMA_CHECKPOINT, schema_frame), (CONTENT_CHECKPOINT, content_frame)):
        result = context.checkpoints.get(checkpoint_name).run(batch_parameters={"dataframe": frame})
        identifier, validation = _read_results(result)
        reports[checkpoint_name] = docs_url(context, identifier)
        stats = validation.statistics
        evaluated += int(stats.get("evaluated_expectations", 0))
        passed += int(stats.get("successful_expectations", 0))
        for item in validation.results:
            if item.success:
                continue
            exp_type = item.expectation_config.type
            kwargs = item.expectation_config.kwargs
            column = kwargs.get("column") or "+".join(kwargs.get("column_list") or []) or "table"
            category = categorize_expectation(exp_type)
            res = item.result or {}
            if exp_type == "expect_table_columns_to_match_set":
                mismatched = (res.get("details") or {}).get("mismatched") or {}
                count = len(mismatched.get("missing") or []) + len(mismatched.get("unexpected") or []) or 1
                missing_columns += count
                category_counts["schema"] += count
                error_counts[exp_type] = error_counts.get(exp_type, 0) + count
                continue
            referenced = [kwargs["column"]] if kwargs.get("column") else list(kwargs.get("column_list") or [])
            if any(c not in frame.columns for c in referenced):
                continue  # rule on an absent column: already reported as a schema error
            indices = res.get("unexpected_index_list")
            if indices is None:
                # Column-level failure (e.g. the column is missing): no row can be trusted.
                indices = list(range(total))
            indices = [i for i in indices if isinstance(i, int)]
            count = int(res.get("unexpected_count") or len(indices))
            if category == "type" and kwargs.get("column"):
                type_cells.update((i, kwargs["column"]) for i in indices)
            elif category == "completeness" and kwargs.get("column"):
                kept = [i for i in indices if (i, kwargs["column"]) not in type_cells]
                if len(kept) != len(indices):
                    indices, count = kept, len(kept)
                if not indices:
                    continue
            error_counts[exp_type] = error_counts.get(exp_type, 0) + count
            category_counts[category] = category_counts.get(category, 0) + count
            for i in indices:
                invalid.add(i)
                row_errors.setdefault(i, []).append(f"{column}:{category}")

    if missing_columns:
        invalid = set(range(total))      # a file with the wrong schema is rejected as a whole
    invalid_pct = round(len(invalid) / total * 100, 2) if total else 0.0
    return {
        "total_rows": total,
        "invalid_row_indices": sorted(invalid),
        "row_errors": {int(k): ";".join(v) for k, v in row_errors.items()},
        "error_counts": error_counts,
        "category_counts": category_counts,
        "summary": summarize_validation_errors(error_counts, missing_columns=missing_columns),
        "missing_columns": missing_columns,
        "invalid_percentage": invalid_pct,
        "criticality": calculate_criticality(invalid_pct, missing_columns=bool(missing_columns)),
        "data_docs_path": reports.get(CONTENT_CHECKPOINT, ""),
        "schema_report_url": reports.get(SCHEMA_CHECKPOINT, ""),
        "expectations_evaluated": evaluated,
        "expectations_passed": passed,
        "success": not invalid,
    }
