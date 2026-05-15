"""cohort_loader.py — Load and normalize the cohort Excel/CSV file for VIEB."""
from __future__ import annotations

import os
import warnings

import pandas as pd


def load_cohort_excel(path: str) -> pd.DataFrame:
    """
    Load cohort Excel (or CSV) file and normalize it.

    Expects columns: Animal, Treatment, Sex, Age, Genotype
    (case-insensitive; "Animal" and "animal_id" both accepted).

    Returns DataFrame with columns:
        animal_id (int), treatment, sex, age_group, genotype, cohort_label
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = df.columns.str.strip()

    # Map incoming column names (case-insensitive) to normalized names
    col_map: dict[str, str] = {}
    for col in df.columns:
        low = col.lower().strip()
        if low in ("animal", "animal_id"):
            col_map[col] = "animal_id"
        elif low == "treatment":
            col_map[col] = "treatment"
        elif low == "sex":
            col_map[col] = "sex"
        elif low in ("age", "age_group"):
            col_map[col] = "age_group"
        elif low == "genotype":
            col_map[col] = "genotype"
    df = df.rename(columns=col_map)

    # Ensure all expected columns exist
    for col in ("animal_id", "treatment", "sex", "age_group", "genotype"):
        if col not in df.columns:
            df[col] = "" if col != "animal_id" else float("nan")

    # Cast animal_id to int (drop rows where it's missing or non-numeric)
    df["animal_id"] = pd.to_numeric(df["animal_id"], errors="coerce")
    df = df.dropna(subset=["animal_id"]).copy()
    df["animal_id"] = df["animal_id"].astype(int)

    # Normalize treatment: "Veh" (case-insensitive) → "Vehicle"; NaN/empty → "Untreated"
    df["treatment"] = df["treatment"].fillna("Untreated")
    df["treatment"] = df["treatment"].apply(
        lambda v: "Vehicle" if str(v).strip().lower() == "veh" else str(v).strip()
    )
    df["treatment"] = df["treatment"].apply(lambda v: "Untreated" if v == "" else v)

    # Ensure string types for grouping columns
    for col in ("sex", "age_group", "genotype"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Generate cohort_label: "{genotype}_{age_group}_{sex}_{treatment}"
    # Spaces and hyphens within each component are replaced with underscores
    def _clean(s: str) -> str:
        return s.replace(" ", "_").replace("-", "_")

    df["cohort_label"] = (
        df["genotype"].apply(_clean) + "_"
        + df["age_group"].apply(_clean) + "_"
        + df["sex"].apply(_clean) + "_"
        + df["treatment"].apply(_clean)
    )

    return df[["animal_id", "treatment", "sex", "age_group", "genotype", "cohort_label"]].reset_index(drop=True)


def get_cohort_summary(df: pd.DataFrame) -> dict:
    """Returns counts per cohort, treatment, genotype, sex, and age group."""
    return {
        "n_animals":    len(df),
        "n_genotypes":  df["genotype"].nunique()   if "genotype"    in df.columns else 0,
        "n_age_groups": df["age_group"].nunique()  if "age_group"   in df.columns else 0,
        "n_treatments": df["treatment"].nunique()  if "treatment"   in df.columns else 0,
        "n_sexes":      df["sex"].nunique()        if "sex"         in df.columns else 0,
        "n_cohorts":    df["cohort_label"].nunique() if "cohort_label" in df.columns else 0,
        "by_cohort":    df["cohort_label"].value_counts().to_dict() if "cohort_label" in df.columns else {},
        "by_treatment": df["treatment"].value_counts().to_dict()    if "treatment"    in df.columns else {},
        "by_genotype":  df["genotype"].value_counts().to_dict()     if "genotype"     in df.columns else {},
        "by_sex":       df["sex"].value_counts().to_dict()          if "sex"          in df.columns else {},
        "by_age_group": df["age_group"].value_counts().to_dict()    if "age_group"    in df.columns else {},
    }


def match_to_vieb(cohort_df: pd.DataFrame, metadata_csv: str) -> pd.DataFrame:
    """
    Left-join cohort data onto VIEB metadata.csv on animal_id.

    Prints a warning for every animal_id in metadata.csv that has no match
    in the cohort file.  Returns the merged DataFrame.
    """
    meta = pd.read_csv(metadata_csv)

    if "animal_id" not in meta.columns:
        warnings.warn("metadata.csv has no 'animal_id' column — returning metadata unchanged.")
        return meta

    cohort = cohort_df.copy()
    cohort["animal_id"] = cohort["animal_id"].astype(int)

    meta["animal_id"] = pd.to_numeric(meta["animal_id"], errors="coerce")
    meta = meta.dropna(subset=["animal_id"]).copy()
    meta["animal_id"] = meta["animal_id"].astype(int)

    merged = meta.merge(cohort, on="animal_id", how="left")

    meta_ids   = set(meta["animal_id"].unique())
    cohort_ids = set(cohort["animal_id"].unique())
    for aid in sorted(meta_ids - cohort_ids):
        warnings.warn(f"animal_id {aid} in metadata.csv has no match in cohort file")

    return merged
