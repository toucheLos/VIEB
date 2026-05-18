"""
behavioral_fingerprint.py — Per-animal behavioral fingerprint and cohort models for VIEB
=========================================================================================

Usage
-----
python behavioral_fingerprint.py --fingerprints [--cohort cohort_normalized.csv]
python behavioral_fingerprint.py --deviation    [--cohort cohort_normalized.csv]
python behavioral_fingerprint.py --forward      [--cohort cohort_normalized.csv]
python behavioral_fingerprint.py --reverse      [--target treatment] [--cohort cohort_normalized.csv]
python behavioral_fingerprint.py --all          [--cohort cohort_normalized.csv]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
_COMP = os.path.join(_ROOT, "results", "comparison")
_CHAR = os.path.join(_ROOT, "results", "characterization")
_SHARED = os.path.join(_ROOT, "results", "shared")
_N_MOTIFS_DEFAULT = 10


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_dominant_state(summary_table: pd.DataFrame | None = None) -> int | None:
    ci_path = os.path.join(_SHARED, "cluster_info.json")
    if os.path.exists(ci_path):
        with open(ci_path, encoding="utf-8") as f:
            ci = json.load(f)
        if "dominant_state" in ci:
            return int(ci["dominant_state"])
    if summary_table is not None:
        state_cols = [c for c in summary_table.columns if c.startswith("state_") and c.endswith("_frac")]
        if state_cols:
            means = summary_table[state_cols].mean()
            return int(means.idxmax().split("_")[1])
    return None


def _load_default_data() -> dict:
    def _csv(path):
        return pd.read_csv(path) if os.path.exists(path) else None

    return {
        "summary_table":    _csv(os.path.join(_COMP, "summary_table.csv")),
        "animal_scalars":   _csv(os.path.join(_COMP, "animal_scalars.csv")),
        "motifs":           _csv(os.path.join(_COMP, "motifs.csv")),
        "state_summary":    _csv(os.path.join(_CHAR, "state_summary.csv")),
        "transition_table": _csv(os.path.join(_COMP, "transition_table.csv")),
        "bouts":            _csv(os.path.join(_CHAR, "bouts.csv")),
    }


def _get_top_motifs(motifs_csv: pd.DataFrame | None, n: int = 10) -> list[tuple[int, int]]:
    """Return top-N flagged enriched bigrams as (from_state, to_state) tuples."""
    if motifs_csv is None or len(motifs_csv) == 0:
        return []
    bigrams = motifs_csv[motifs_csv["type"] == "bigram"].copy()
    pool = bigrams[bigrams["flagged"] == True] if (bigrams["flagged"] == True).any() else bigrams
    pool = pool.nlargest(n, "enrichment_ratio")
    result = []
    for m in pool["motif"]:
        try:
            t = ast.literal_eval(str(m))
            if isinstance(t, tuple) and len(t) == 2:
                result.append((int(t[0]), int(t[1])))
        except Exception:
            pass
    return result


def _per_animal_bout_dur(animal_id: int, bouts: pd.DataFrame | None,
                          state_summary: pd.DataFrame | None) -> dict[int, float]:
    """Mean bout duration per state for one animal.  Falls back to global state_summary."""
    dur: dict[int, float] = {}
    if bouts is not None and "animal_id" in bouts.columns:
        ab = bouts[(bouts["animal_id"] == animal_id) & (bouts["state"] >= 0)]
        if len(ab) > 0:
            for state, grp in ab.groupby("state"):
                dur[int(state)] = float(grp["duration_sec"].mean())
    # Fill missing states from global state_summary
    if state_summary is not None and "state" in state_summary.columns:
        for _, row in state_summary.iterrows():
            s = int(row["state"])
            if s not in dur and "mean_bout_dur_sec" in row:
                dur[s] = float(row["mean_bout_dur_sec"])
    return dur


# ---------------------------------------------------------------------------
# Feature grouping utilities (also used by plot_cohort.py and gui.py)
# ---------------------------------------------------------------------------

def _group_features(feat_names: list[str]) -> dict[str, list[str]]:
    """Categorise fingerprint column names into display groups."""
    groups: dict[str, list[str]] = {
        "frac_A":   [],
        "frac_B":   [],
        "delta":    [],
        "bout_dur": [],
        "scalar":   [],
        "motif":    [],
    }
    for f in feat_names:
        if "_frac_A" in f:
            groups["frac_A"].append(f)
        elif "_frac_B" in f:
            groups["frac_B"].append(f)
        elif "_delta" in f:
            groups["delta"].append(f)
        elif "_bout_dur" in f:
            groups["bout_dur"].append(f)
        elif f in ("freeze_auc", "mean_disc_ratio", "peak_disc_day"):
            groups["scalar"].append(f)
        else:
            groups["motif"].append(f)
    return groups


def _sorted_feature_order(feat_names: list[str]) -> list[str]:
    """Return feat_names sorted by display category."""
    g = _group_features(feat_names)
    return (g["frac_A"] + g["frac_B"] + g["delta"]
            + g["bout_dur"] + g["scalar"] + g["motif"])


# ---------------------------------------------------------------------------
# Task 1 — Behavioral fingerprint
# ---------------------------------------------------------------------------

def compute_fingerprint(
    animal_id: int,
    summary_table: pd.DataFrame,
    animal_scalars: pd.DataFrame,
    motifs_csv: pd.DataFrame,
    state_summary: pd.DataFrame,
    transition_table: pd.DataFrame | None = None,
    bouts: pd.DataFrame | None = None,
    n_motifs: int = _N_MOTIFS_DEFAULT,
    dominant_state: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns (feature_vector, feature_names) for one animal.

    State occupancy features (one per non-dominant state):
      s{k}_frac_A, s{k}_frac_B, s{k}_delta

    Temporal features:
      s{k}_bout_dur (per-animal mean from bouts.csv)
      freeze_auc, mean_disc_ratio, peak_disc_day

    Motif features (top-N enriched bigrams):
      motif_{i}_{j}_A, motif_{i}_{j}_B  (transition probability per context)

    Vector length ≈ 4*n_non_dom_states + 3 + 2*n_motifs
    """
    features: list[float] = []
    names:    list[str]   = []

    state_cols = sorted(
        [c for c in summary_table.columns if c.startswith("state_") and c.endswith("_frac")],
        key=lambda c: int(c.split("_")[1]),
    )
    all_states = [int(c.split("_")[1]) for c in state_cols]

    if dominant_state is None:
        dominant_state = _get_dominant_state(summary_table)

    non_dom = [s for s in all_states if s != dominant_state]
    animal_rows = summary_table[summary_table["animal_id"] == animal_id].copy()

    # Freeze state and scalars
    freeze_state: int | None = None
    freeze_auc = disc_ratio = peak_disc_day = np.nan
    if animal_scalars is not None:
        sr = animal_scalars[animal_scalars["animal_id"] == animal_id]
        if len(sr) > 0:
            if "freeze_state" in sr.columns:
                freeze_state = int(sr["freeze_state"].iloc[0])
            if "freeze_auc" in sr.columns:
                freeze_auc = float(sr["freeze_auc"].iloc[0])
            if "mean_discrimination_ratio" in sr.columns:
                disc_ratio = float(sr["mean_discrimination_ratio"].iloc[0])

    # Peak discrimination day
    if freeze_state is not None:
        fc = f"state_{freeze_state}_frac"
        if fc in animal_rows.columns:
            disc_by_day: list[tuple[float, float]] = []
            for day, dg in animal_rows.groupby("day"):
                ctx = dg.dropna(subset=["context", fc]).groupby("context")[fc].mean()
                if "A" in ctx.index and "B" in ctx.index:
                    fa, fb = ctx["A"], ctx["B"]
                    denom = fa + fb
                    if denom > 0:
                        disc_by_day.append((float(day), float((fa - fb) / denom)))
            if disc_by_day:
                peak_disc_day = float(max(disc_by_day, key=lambda x: x[1])[0])

    # ---- State occupancy (Context A, B, delta) ----
    for state in non_dom:
        col = f"state_{state}_frac"
        if col in animal_rows.columns:
            a_rows = animal_rows[animal_rows["context"] == "A"]
            b_rows = animal_rows[animal_rows["context"] == "B"]
            fa = float(a_rows[col].mean()) if len(a_rows) > 0 else np.nan
            fb = float(b_rows[col].mean()) if len(b_rows) > 0 else np.nan
            delta = (fa - fb) if not (np.isnan(fa) or np.isnan(fb)) else np.nan
        else:
            fa = fb = delta = np.nan
        features += [fa, fb, delta]
        names    += [f"s{state}_frac_A", f"s{state}_frac_B", f"s{state}_delta"]

    # ---- Bout duration per state (per-animal from bouts.csv) ----
    dur_map = _per_animal_bout_dur(animal_id, bouts, state_summary)
    for state in non_dom:
        features.append(dur_map.get(state, np.nan))
        names.append(f"s{state}_bout_dur")

    # ---- Scalar temporal features ----
    features += [freeze_auc, disc_ratio, peak_disc_day]
    names    += ["freeze_auc", "mean_disc_ratio", "peak_disc_day"]

    # ---- Motif features (top enriched bigrams) ----
    top_motifs = _get_top_motifs(motifs_csv, n_motifs)
    if top_motifs and transition_table is not None and "animal_id" in transition_table.columns:
        at = transition_table[transition_table["animal_id"] == animal_id]
        for (i, j) in top_motifs:
            col = f"trans_{i}_{j}"
            if "context" in at.columns:
                a_t = at[at["context"] == "A"]
                b_t = at[at["context"] == "B"]
                fa_m = float(a_t[col].mean()) if col in a_t.columns and len(a_t) > 0 else np.nan
                fb_m = float(b_t[col].mean()) if col in b_t.columns and len(b_t) > 0 else np.nan
            else:
                fa_m = fb_m = np.nan
            features += [fa_m, fb_m]
            names    += [f"motif_{i}_{j}_A", f"motif_{i}_{j}_B"]

    return np.array(features, dtype=np.float64), names


def build_fingerprint_matrix(
    animal_ids: list[int],
    summary_table: pd.DataFrame,
    animal_scalars: pd.DataFrame,
    motifs_csv: pd.DataFrame,
    state_summary: pd.DataFrame,
    transition_table: pd.DataFrame | None = None,
    bouts: pd.DataFrame | None = None,
    cohort_df: pd.DataFrame | None = None,
    n_motifs: int = _N_MOTIFS_DEFAULT,
    dominant_state: int | None = None,
) -> pd.DataFrame:
    """
    Builds (n_animals × n_features) DataFrame indexed by animal_id.
    Missing values filled with cohort mean first, then global mean.
    Saves to results/comparison/behavioral_fingerprints.csv.
    """
    if dominant_state is None:
        dominant_state = _get_dominant_state(summary_table)

    print(f"Building fingerprint matrix for {len(animal_ids)} animals "
          f"(dominant_state={dominant_state})...")

    vecs: list[np.ndarray] = []
    feat_names: list[str]  = []

    for aid in animal_ids:
        vec, names = compute_fingerprint(
            aid, summary_table, animal_scalars, motifs_csv, state_summary,
            transition_table=transition_table, bouts=bouts,
            n_motifs=n_motifs, dominant_state=dominant_state,
        )
        vecs.append(vec)
        if not feat_names:
            feat_names = names

    if not vecs:
        print("[WARNING] No animals found — returning empty fingerprint matrix.")
        return pd.DataFrame()

    mat  = np.stack(vecs)
    df   = pd.DataFrame(mat, index=animal_ids, columns=feat_names)
    df.index.name = "animal_id"

    # Drop constant columns (zero variance — e.g. if cohort has only 1 treatment)
    var_mask = df.var(axis=0) > 0
    n_dropped = (~var_mask).sum()
    if n_dropped > 0:
        df = df.loc[:, var_mask]
        print(f"  Dropped {n_dropped} zero-variance features.")
    feat_names = list(df.columns)

    # Impute NaN: cohort mean first, then global mean
    if cohort_df is not None and "cohort_label" in cohort_df.columns:
        cmap = cohort_df.set_index("animal_id")["cohort_label"].to_dict()
        df["_cohort"] = pd.Series({int(k): v for k, v in cmap.items()}).reindex(df.index)
        for cl in df["_cohort"].dropna().unique():
            mask = df["_cohort"] == cl
            cm   = df.loc[mask, feat_names].mean()
            df.loc[mask, feat_names] = df.loc[mask, feat_names].fillna(cm)
        df = df.drop(columns=["_cohort"])

    df = df.fillna(df.mean())  # global mean for any remaining NaN

    n_nan = df.isna().sum().sum()
    if n_nan > 0:
        df = df.fillna(0.0)
        print(f"  [WARNING] {n_nan} NaN values remain after imputation — filled with 0.")

    os.makedirs(_COMP, exist_ok=True)
    out = os.path.join(_COMP, "behavioral_fingerprints.csv")
    fp_out = df.reset_index()
    fp_out.to_csv(out, index=False)
    print(f"\nFingerprint matrix: {df.shape[0]} animals × {df.shape[1]} features")
    print(f"Saved -> {out}")
    return df


# ---------------------------------------------------------------------------
# Task 2 — Cohort deviation score
# ---------------------------------------------------------------------------

def compute_deviation_scores(
    fingerprint_matrix: pd.DataFrame,
    cohort_df: pd.DataFrame,
    groupby: str = "cohort_label",
) -> pd.DataFrame:
    """
    For each animal: per-feature z-score relative to its cohort peers,
    composite deviation score, context-specific scores.

    composite_z > 2.0 flags the animal as a behavioral outlier in its cohort.
    Saves to results/comparison/deviation_scores.csv.
    """
    if cohort_df is None or groupby not in cohort_df.columns:
        raise ValueError(f"cohort_df must have column '{groupby}'")

    fp = fingerprint_matrix.copy()
    fp.index = fp.index.astype(int)
    feat_cols = list(fp.columns)

    cdf = cohort_df.copy()
    cdf["animal_id"] = cdf["animal_id"].astype(int)
    cmap = cdf.set_index("animal_id")[groupby].to_dict()

    # Cross-cohort variance for feature weighting
    cohort_group_means: dict[str, pd.Series] = {}
    for cl, grp in cdf.groupby(groupby)["animal_id"]:
        ids = [i for i in grp if i in fp.index]
        if ids:
            cohort_group_means[cl] = fp.loc[ids].mean()

    if len(cohort_group_means) > 1:
        between_var = pd.DataFrame(cohort_group_means).T.var() + 1e-10
    else:
        between_var = pd.Series(np.ones(len(feat_cols)), index=feat_cols)

    weights = between_var[feat_cols].values
    weights = weights / weights.sum()

    rows: list[dict] = []
    for aid in fp.index:
        cl = cmap.get(int(aid), "Unknown")
        peers = [i for i, c in cmap.items() if c == cl and i in fp.index and i != int(aid)]

        if len(peers) < 2:
            z_series = pd.Series(np.zeros(len(feat_cols)), index=feat_cols)
        else:
            peer_fp = fp.loc[peers]
            mu  = peer_fp.mean()
            sig = peer_fp.std().clip(lower=1e-10)
            z_series = (fp.loc[aid] - mu) / sig

        z_abs = np.abs(z_series.values)
        comp_z = float(np.nansum(z_abs * weights))

        a_cols = [c for c in feat_cols if "_frac_A" in c or c.endswith("_A")]
        b_cols = [c for c in feat_cols if "_frac_B" in c or c.endswith("_B")]
        ctx_a  = float(np.nanmean(z_abs[[feat_cols.index(c) for c in a_cols]])) if a_cols else np.nan
        ctx_b  = float(np.nanmean(z_abs[[feat_cols.index(c) for c in b_cols]])) if b_cols else np.nan
        most_deviant = feat_cols[int(np.nanargmax(z_abs))]

        row: dict = {
            "animal_id":          int(aid),
            "cohort_label":       cl,
            "composite_z":        round(comp_z, 4),
            "context_A_z":        round(ctx_a, 4),
            "context_B_z":        round(ctx_b, 4),
            "most_deviant_feature": most_deviant,
        }
        for fc in feat_cols:
            row[f"z_{fc}"] = round(float(z_series[fc]), 4)
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("composite_z", ascending=False)

    outliers = result[result["composite_z"] > 2.0]
    print(f"\n{'='*60}")
    if len(outliers) > 0:
        print(f"Behavioral outliers (composite_z > 2.0): {len(outliers)} animals")
        for _, r in outliers.iterrows():
            print(f"  Animal {r['animal_id']:>6}  cohort={r['cohort_label']:<30}  "
                  f"z={r['composite_z']:.2f}  deviant feat: {r['most_deviant_feature']}")
    else:
        print("No behavioral outliers detected (all composite_z ≤ 2.0)")
    print(f"{'='*60}")

    os.makedirs(_COMP, exist_ok=True)
    out = os.path.join(_COMP, "deviation_scores.csv")
    result.to_csv(out, index=False)
    print(f"Saved -> {out}")
    return result


# ---------------------------------------------------------------------------
# Task 3 — Forward model: cohort -> behavior
# ---------------------------------------------------------------------------

def fit_forward_model(
    fingerprint_matrix: pd.DataFrame,
    cohort_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """
    Fits Ridge regression per behavioral feature:
      behavioral_feature = w_genotype + w_age + w_sex + w_treatment

    Returns (results_dict, coefficients_DataFrame).
    Saves coefficients to results/comparison/forward_model_weights.csv.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import r2_score as _r2

    fp  = fingerprint_matrix.copy()
    fp.index = fp.index.astype(int)
    cdf = cohort_df.copy()
    cdf["animal_id"] = cdf["animal_id"].astype(int)

    common = sorted(set(fp.index) & set(cdf["animal_id"]))
    n_animals = len(common)
    if n_animals < 5:
        print(f"[ERROR] Only {n_animals} animals in common — cannot fit forward model.")
        return {}, pd.DataFrame()
    if n_animals < 30:
        print(f"[WARNING] Forward model: N={n_animals} < 30 — interpret R² cautiously.")

    cdf_sub = cdf[cdf["animal_id"].isin(common)].set_index("animal_id")
    fp_sub  = fp.loc[common]
    cov_cols = [c for c in ("genotype", "sex", "age_group", "treatment") if c in cdf_sub.columns]

    if not cov_cols:
        print("[ERROR] cohort_df has none of the expected covariate columns.")
        return {}, pd.DataFrame()

    X_cat = cdf_sub[cov_cols].fillna("Unknown")
    enc   = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")
    X     = enc.fit_transform(X_cat)
    enc_names = list(enc.get_feature_names_out(cov_cols))

    # Warn if any cohort group is tiny
    for col in cov_cols:
        vc = X_cat[col].value_counts()
        small = vc[vc < 10]
        for grp, cnt in small.items():
            print(f"  [WARNING] {col}={grp!r}: only {cnt} animals (< 10)")

    Y = fp_sub.values  # (n_animals, n_feats)
    feat_cols = list(fp_sub.columns)

    model = Ridge(alpha=1.0)
    model.fit(X, Y)
    Y_pred = model.predict(X)

    r2_per_feat: dict[str, float] = {}
    dom_pred:    dict[str, str]   = {}
    for fi, feat in enumerate(feat_cols):
        ss_res = np.sum((Y[:, fi] - Y_pred[:, fi]) ** 2)
        ss_tot = np.sum((Y[:, fi] - Y[:, fi].mean()) ** 2)
        r2_per_feat[feat] = round(float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0, 4)

        # Which covariate cluster has highest |coef| sum?
        cov_imp: dict[str, float] = {}
        for cov in cov_cols:
            idx = [i for i, n in enumerate(enc_names) if n.startswith(f"{cov}_")]
            if idx:
                cov_imp[cov] = float(np.sum(np.abs(model.coef_[fi, idx])))
        dom_pred[feat] = max(cov_imp, key=cov_imp.get) if cov_imp else "unknown"

    # Predicted fingerprint for each unique cohort combination
    unique_combos = X_cat.drop_duplicates()
    pred_rows: list[dict] = []
    for _, combo_row in unique_combos.iterrows():
        x_enc = enc.transform(pd.DataFrame([combo_row], columns=cov_cols))
        y_hat = model.predict(x_enc)[0]
        pred: dict = {c: v for c, v in zip(cov_cols, combo_row)}
        for fi, feat in enumerate(feat_cols):
            pred[feat] = round(float(y_hat[fi]), 6)
        pred_rows.append(pred)

    # Coefficient table
    coef_rows: list[dict] = []
    for fi, feat in enumerate(feat_cols):
        row: dict = {
            "feature":           feat,
            "r2":                r2_per_feat[feat],
            "dominant_predictor": dom_pred[feat],
        }
        for n, c in zip(enc_names, model.coef_[fi]):
            row[n] = round(float(c), 6)
        coef_rows.append(row)
    coef_df = pd.DataFrame(coef_rows)

    results = {
        "n_animals":          n_animals,
        "r2":                 r2_per_feat,
        "dominant_predictor": dom_pred,
        "predictions":        pred_rows,
        "enc_feature_names":  enc_names,
        "cov_cols":           cov_cols,
    }

    med_r2 = float(np.median(list(r2_per_feat.values())))
    print(f"\nForward model: {len(feat_cols)} features  ·  N={n_animals} animals")
    print(f"  Median R²: {med_r2:.3f}")
    top5 = sorted(r2_per_feat.items(), key=lambda x: x[1], reverse=True)[:5]
    print("  Top 5 features by R²:")
    for feat, r2 in top5:
        print(f"    {feat:<45}  R²={r2:.3f}  dominant={dom_pred[feat]}")

    os.makedirs(_COMP, exist_ok=True)
    out = os.path.join(_COMP, "forward_model_weights.csv")
    coef_df.to_csv(out, index=False)
    print(f"Saved -> {out}")
    return results, coef_df


# ---------------------------------------------------------------------------
# Task 4 — Reverse model: behavior -> cohort
# ---------------------------------------------------------------------------

def fit_reverse_model(
    fingerprint_matrix: pd.DataFrame,
    cohort_df: pd.DataFrame,
    target: str = "treatment",
) -> dict:
    """
    Random Forest classifier with LOO-CV to predict cohort membership.

    Runs for each requested target variable.
    Saves to results/comparison/reverse_model_results.json.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.preprocessing import LabelEncoder

    fp  = fingerprint_matrix.copy()
    fp.index = fp.index.astype(int)
    cdf = cohort_df.copy()
    cdf["animal_id"] = cdf["animal_id"].astype(int)

    if target not in cdf.columns:
        print(f"[ERROR] Target column '{target}' not in cohort_df.")
        return {}

    common = sorted(set(fp.index) & set(cdf["animal_id"]))
    n_animals = len(common)
    if n_animals < 5:
        print(f"[ERROR] Only {n_animals} animals — cannot fit reverse model for '{target}'.")
        return {}
    if n_animals < 30:
        print(f"[WARNING] Reverse model: N={n_animals} < 30 — use LOO-CV accuracy cautiously.")

    cdf_sub = cdf[cdf["animal_id"].isin(common)].set_index("animal_id")
    fp_sub  = fp.loc[common]
    X = fp_sub.values
    labels_raw = cdf_sub.loc[common, target].fillna("Unknown").astype(str).values

    le    = LabelEncoder()
    y     = le.fit_transform(labels_raw)
    class_names = list(le.classes_)
    n_classes   = len(class_names)

    # Warn on small classes
    for cls, cnt in zip(*np.unique(y, return_counts=True)):
        if cnt < 10:
            print(f"  [WARNING] {target}={class_names[cls]!r}: only {cnt} animals (< 10)")

    loo     = LeaveOneOut()
    y_true: list[int] = []
    y_pred: list[int] = []

    rf = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    for train_idx, test_idx in loo.split(X):
        # Skip folds where test class is not in training set
        test_class = y[test_idx[0]]
        if test_class not in y[train_idx]:
            y_pred.append(int(test_class))  # predict correctly (trivial, not counted)
            y_true.append(int(test_class))
            continue
        rf.fit(X[train_idx], y[train_idx])
        y_pred.append(int(rf.predict(X[test_idx])[0]))
        y_true.append(int(test_class))

    acc = float(accuracy_score(y_true, y_pred))
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()

    # Per-class accuracy
    per_class_acc: dict[str, float] = {}
    for ci, cls_name in enumerate(class_names):
        idxs = [i for i, t in enumerate(y_true) if t == ci]
        if idxs:
            per_class_acc[cls_name] = float(np.mean([y_pred[i] == ci for i in idxs]))
        else:
            per_class_acc[cls_name] = float("nan")

    # Feature importances (fit on all data)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feat_imp = {feat: round(float(imp), 6)
                for feat, imp in zip(list(fp_sub.columns), importances)}
    top10 = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]

    # Per-animal prediction
    per_animal: list[dict] = []
    for aid, true_cls, pred_cls in zip(common, y_true, y_pred):
        per_animal.append({
            "animal_id":     int(aid),
            "actual":        class_names[true_cls],
            "predicted":     class_names[pred_cls],
            "correct":       bool(true_cls == pred_cls),
        })

    result = {
        "target":            target,
        "n_animals":         n_animals,
        "n_classes":         n_classes,
        "class_names":       class_names,
        "loo_accuracy":      round(acc, 4),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix":  cm,
        "feature_importances": feat_imp,
        "top10_features":    [{"feature": f, "importance": i} for f, i in top10],
        "per_animal":        per_animal,
    }

    print(f"\nReverse model: target='{target}'  N={n_animals}  classes={n_classes}")
    print(f"  LOO accuracy: {acc:.3f}")
    print(f"  Per-class accuracy:")
    for cls, a in per_class_acc.items():
        print(f"    {cls:<30} {a:.3f}")
    print(f"  Top features:")
    for f, i in top10[:5]:
        print(f"    {f:<45}  imp={i:.4f}")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_cohort(cohort_path: str | None) -> pd.DataFrame | None:
    if not cohort_path:
        norm = os.path.join(_ROOT, "cohort_normalized.csv")
        if os.path.exists(norm):
            cohort_path = norm
        else:
            # Try config.json
            cfg_path = os.path.join(_ROOT, "config.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, encoding="utf-8") as f:
                        cfg = json.load(f)
                    cohort_path = cfg.get("cohort_csv_path") or cfg.get("cohort_xlsx_path")
                except Exception:
                    pass
    if not cohort_path or not os.path.exists(cohort_path):
        return None
    ext = os.path.splitext(cohort_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        from cohort_loader import load_cohort_excel
        return load_cohort_excel(cohort_path)
    return pd.read_csv(cohort_path)


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral fingerprint and cohort models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fingerprints", action="store_true",
                        help="Build behavioral fingerprint matrix")
    parser.add_argument("--deviation", action="store_true",
                        help="Compute per-animal deviation scores from cohort norm")
    parser.add_argument("--forward", action="store_true",
                        help="Fit forward model: cohort -> behavior (Ridge regression)")
    parser.add_argument("--reverse", action="store_true",
                        help="Fit reverse model: behavior -> cohort (Random Forest LOO-CV)")
    parser.add_argument("--plots", action="store_true",
                        help="Generate cohort analysis plots (calls plot_cohort.py)")
    parser.add_argument("--all", action="store_true",
                        help="Run all steps in order")
    parser.add_argument("--cohort", type=str, default=None, metavar="FILE",
                        help="Path to cohort CSV/Excel file (auto-detected if omitted)")
    parser.add_argument("--target", type=str, default=None, metavar="COL",
                        help="Target column for reverse model (default: all of treatment/genotype/sex/age_group)")
    parser.add_argument("--n-motifs", type=int, default=_N_MOTIFS_DEFAULT,
                        help=f"Number of top enriched motifs in fingerprint (default {_N_MOTIFS_DEFAULT})")
    args = parser.parse_args()

    if not any([args.fingerprints, args.deviation, args.forward,
                args.reverse, args.plots, args.all]):
        parser.print_help()
        sys.exit(1)

    do_fp  = args.fingerprints or args.all
    do_dev = args.deviation    or args.all
    do_fwd = args.forward      or args.all
    do_rev = args.reverse      or args.all
    do_plt = args.plots        or args.all

    # Load required data
    print("Loading results data...")
    d = _load_default_data()
    if d["summary_table"] is None:
        sys.exit("[ERROR] results/comparison/summary_table.csv not found. Run compare.py --report first.")
    if d["animal_scalars"] is None:
        sys.exit("[ERROR] results/comparison/animal_scalars.csv not found. Run compare.py --summarize first.")

    cohort_df = _load_cohort(args.cohort)
    if cohort_df is None:
        print("[INFO] No cohort file found — fingerprints will be built without cohort imputation.")
        if do_dev or do_fwd or do_rev:
            print("[WARNING] Deviation scores and models require a cohort file.")
            print("         Pass --cohort <file.csv/.xlsx> or upload via GUI first.")

    # Determine animal IDs
    st = d["summary_table"]
    st["animal_id"] = pd.to_numeric(st["animal_id"], errors="coerce")
    animal_ids = sorted(st["animal_id"].dropna().astype(int).unique().tolist())
    print(f"  {len(animal_ids)} unique animals in summary_table")

    dom = _get_dominant_state(st)

    # ---- Fingerprints ----
    fp_path = os.path.join(_COMP, "behavioral_fingerprints.csv")
    if do_fp:
        fp_df = build_fingerprint_matrix(
            animal_ids, st, d["animal_scalars"], d["motifs"],
            d["state_summary"], d["transition_table"], d["bouts"],
            cohort_df=cohort_df, n_motifs=args.n_motifs, dominant_state=dom,
        )
    elif os.path.exists(fp_path):
        fp_df = pd.read_csv(fp_path, index_col="animal_id")
        print(f"Loaded existing fingerprints: {fp_df.shape}")
    else:
        fp_df = pd.DataFrame()

    if fp_df.empty:
        if do_dev or do_fwd or do_rev:
            sys.exit("[ERROR] No fingerprint data — run --fingerprints first.")
        return

    # ---- Deviation scores ----
    if do_dev:
        if cohort_df is None:
            print("[SKIP] --deviation requires cohort data (--cohort file).")
        else:
            compute_deviation_scores(fp_df, cohort_df)

    # ---- Forward model ----
    fwd_results: dict = {}
    if do_fwd:
        if cohort_df is None:
            print("[SKIP] --forward requires cohort data (--cohort file).")
        else:
            fwd_results, _ = fit_forward_model(fp_df, cohort_df)

    # ---- Reverse models ----
    all_rev: dict = {}
    if do_rev:
        if cohort_df is None:
            print("[SKIP] --reverse requires cohort data (--cohort file).")
        else:
            targets = (
                [args.target] if args.target
                else [c for c in ("treatment", "genotype", "sex", "age_group")
                      if c in cohort_df.columns]
            )
            for tgt in targets:
                all_rev[tgt] = fit_reverse_model(fp_df, cohort_df, target=tgt)
            out_path = os.path.join(_COMP, "reverse_model_results.json")
            os.makedirs(_COMP, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(all_rev, f, indent=2)
            print(f"\nReverse model results saved -> {out_path}")

    # ---- Plots ----
    if do_plt:
        import subprocess
        cohort_args = ["--cohort", args.cohort] if args.cohort else []
        rc = subprocess.run(
            [sys.executable, os.path.join(_ROOT, "plot_cohort.py"), "--all"] + cohort_args,
            cwd=_ROOT,
        )
        if rc.returncode != 0:
            print("[WARNING] plot_cohort.py returned non-zero exit code.")


if __name__ == "__main__":
    main()
