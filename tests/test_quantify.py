"""Tests for quantify.py's animal_id dtype coercion (coerce_id_column) and
build_master_table()'s merges, which previously crashed when summary_table.csv's
animal_id was str-typed and the cohort file's was int64 (e.g. via
cohort_loader.load_cohort_excel(), or a plain numeric-looking CSV column)."""

from __future__ import annotations

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import quantify  # noqa: E402


def test_coerce_id_column_returns_copy_and_casts():
    df = pd.DataFrame({"animal_id": [142, 143], "x": [1, 2]})
    out = quantify.coerce_id_column(df)

    assert out is not df
    assert out["animal_id"].dtype == object
    assert list(out["animal_id"]) == ["142", "143"]
    # Original untouched.
    assert df["animal_id"].dtype != object


def _setup_project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    (results / "comparison").mkdir(parents=True)
    (results / "shared").mkdir()

    # One row uses a non-numeric animal_id ("control_X"), which forces pandas'
    # default pd.read_csv() to infer the whole animal_id column as object/str
    # dtype — exactly how a real messy metadata column ends up str-typed even
    # though most of its values ("142", "143") look numeric. The int64-typed
    # cohort file below uses the same underlying "142"/"143" identifiers.
    summary = pd.DataFrame({
        "stem": ["s1_d1", "s1_d2", "s2_d1", "s2_d2", "s3_d1"],
        "animal_id": ["142", "142", "143", "143", "control_X"],
        "context": ["A", "B", "A", "B", "A"],
        "day": [1, 1, 1, 1, 1],
        "state_0_frac": [0.7, 0.3, 0.6, 0.4, 0.5],
        "state_1_frac": [0.3, 0.7, 0.4, 0.6, 0.5],
    })
    summary.to_csv(results / "comparison" / "summary_table.csv", index=False)

    with open(results / "shared" / "cluster_info.json", "w") as f:
        json.dump({"n_clusters": 2}, f)

    monkeypatch.setattr(quantify, "_RES", lambda: __import__("pathlib").Path(results))
    return results


def test_build_master_table_succeeds_with_mixed_animal_id_dtypes(tmp_path, monkeypatch):
    results = _setup_project(tmp_path, monkeypatch)

    # Cohort's animal_id is written as bare integers -> pandas infers int64,
    # exactly like cohort_loader.load_cohort_excel()'s explicit int cast.
    cohort_csv = tmp_path / "cohort.csv"
    pd.DataFrame({
        "animal_id": [142, 143],
        "cohort_label": ["groupA", "groupA"],
        "sex": ["M", "F"],
    }).to_csv(cohort_csv, index=False)

    out_dir = tmp_path / "quant_out"
    master = quantify.build_master_table(cohort_path=str(cohort_csv), out_dir=str(out_dir))

    assert set(master["animal_id"]) == {"142", "143", "control_X"}
    # Cohort columns populated for matching animals, correctly NaN for the
    # one animal ("control_X") that has no cohort row at all.
    matched = master.set_index("animal_id").loc[["142", "143"]]
    assert matched["cohort_label"].notna().all()
    assert set(matched["cohort_label"]) == {"groupA"}
    assert matched["sex"].loc["142"] == "M"
    assert pd.isna(master.set_index("animal_id")["cohort_label"].loc["control_X"])
    assert master.set_index("animal_id")["cohort_label"].loc["control_X"] != master.set_index("animal_id")["cohort_label"].loc["control_X"]  # NaN


def test_build_master_table_no_cohort_file(tmp_path, monkeypatch):
    results = _setup_project(tmp_path, monkeypatch)
    out_dir = tmp_path / "quant_out_no_cohort"

    master = quantify.build_master_table(cohort_path=None, out_dir=str(out_dir))

    assert set(master["animal_id"]) == {"142", "143", "control_X"}
    assert master["fear_index"].isna().all()
