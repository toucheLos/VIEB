"""Validation statistics for comparing clustering/feature-representation runs.

This is the primary quantitative bar for judging alternative pose feature
representations (see ``ml/representations/``) against the current
hand-crafted default: a representation is "better" if it produces states
with higher repeatability and less community-bridging, not if it produces a
prettier UMAP plot.

Two independent metrics:

1. ``compute_repeatability_R`` — Nakagawa & Schielzeth (2010) adjusted
   repeatability (an ANOVA-based intraclass correlation): the proportion of
   total variance in an animal's state-occupancy that is between-individual
   rather than within-individual (session-to-session) variance. Computed
   per state from a ``summary_table.csv``-shaped DataFrame (one row per
   video/session, already joined with ``animal_id``/session metadata by
   ``compare.py cmd_report``).

2. ``compute_transition_modularity`` — community detection (Louvain) on the
   state-transition graph, flagging states whose transitions are split
   roughly evenly across two communities as possibly conflating two
   distinct behaviors. Populates the ``possible_split_states`` field that
   ``views/video_stories.py``'s ``load_possible_split_states()`` hook has
   been waiting for since docs/DECISIONS.md #50.

Both functions degrade gracefully (return a `"skipped"` result with a
`"reason"`) when the data doesn't support the computation, rather than
raising — consistent with the rest of the codebase's philosophy for
optional/conditional analyses (docs/DECISIONS.md #2).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Part A.1 — Nakagawa & Schielzeth adjusted repeatability (R)
# ---------------------------------------------------------------------------

def _one_way_anova_components(groups: list[np.ndarray]) -> tuple[float, float, float]:
    """Return (MS_among, MS_within, n0) for a one-way ANOVA over groups.

    ``n0`` is the unequal-sample-size correction factor from Nakagawa &
    Schielzeth (2010), eq. 6:
        n0 = (1 / (a - 1)) * (sum(n_i) - sum(n_i**2) / sum(n_i))
    where ``a`` is the number of groups (individuals).
    """
    a = len(groups)
    ns = np.array([len(g) for g in groups], dtype=np.float64)
    grand_mean = np.concatenate(groups).mean()

    ss_among = float(np.sum(ns * (np.array([g.mean() for g in groups]) - grand_mean) ** 2))
    ss_within = float(np.sum([np.sum((g - g.mean()) ** 2) for g in groups]))

    df_among = a - 1
    df_within = float(np.sum(ns - 1))

    ms_among = ss_among / df_among if df_among > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    n_total = ns.sum()
    n0 = (1.0 / df_among) * (n_total - np.sum(ns ** 2) / n_total) if df_among > 0 else 0.0
    return ms_among, ms_within, n0


def _repeatability_for_state(values: pd.Series, animals: pd.Series) -> dict:
    df = pd.DataFrame({"value": values, "animal": animals}).dropna()
    groups_by_animal = [g["value"].to_numpy() for _, g in df.groupby("animal") if len(g) >= 2]

    if len(groups_by_animal) < 2:
        return {"skipped": True, "reason": "fewer than 2 animals with >=2 sessions each"}

    ms_among, ms_within, n0 = _one_way_anova_components(groups_by_animal)
    if n0 <= 0:
        return {"skipped": True, "reason": "degenerate design (n0 <= 0)"}

    v_among = max(0.0, (ms_among - ms_within) / n0)
    v_within = ms_within
    denom = v_among + v_within
    r = float(v_among / denom) if denom > 0 else 0.0
    return {
        "skipped": False,
        "R": r,
        "n_animals": len(groups_by_animal),
        "n_sessions_total": int(sum(len(g) for g in groups_by_animal)),
    }


def compute_repeatability_R(
    df: pd.DataFrame,
    state_cols: list[str],
    animal_col: str = "animal_id",
    session_col: Optional[str] = "day",
) -> dict:
    """Compute per-state adjusted repeatability R across repeated sessions.

    Parameters
    ----------
    df : DataFrame with one row per video/session (e.g. summary_table.csv),
        containing ``animal_col`` and each column in ``state_cols``.
    state_cols : list of state-occupancy-fraction column names.
    animal_col : column identifying the individual (repeated-measures unit).
    session_col : column identifying the repeated session/timepoint; only
        used to validate that repeats actually exist (>=2 distinct values
        per animal) — the ANOVA itself just needs >=2 rows per animal.

    Returns
    -------
    dict: {"per_state": {state_col: {...}}, "mean_R": float | None,
           "n_states_scored": int, "skipped": bool, "reason": str | None}
    """
    if animal_col not in df.columns:
        return {"skipped": True, "reason": f"no '{animal_col}' column in metadata", "per_state": {}, "mean_R": None}

    if session_col is not None and session_col in df.columns:
        sessions_per_animal = df.groupby(animal_col)[session_col].nunique()
        if not (sessions_per_animal >= 2).any():
            return {
                "skipped": True,
                "reason": f"no animal has >=2 distinct '{session_col}' values",
                "per_state": {},
                "mean_R": None,
            }

    per_state = {}
    r_values = []
    for col in state_cols:
        if col not in df.columns:
            per_state[col] = {"skipped": True, "reason": "column not present"}
            continue
        result = _repeatability_for_state(df[col], df[animal_col])
        per_state[col] = result
        if not result.get("skipped"):
            r_values.append(result["R"])

    mean_r = float(np.mean(r_values)) if r_values else None
    return {
        "skipped": mean_r is None,
        "reason": None if mean_r is not None else "no state had sufficient repeated-measures data",
        "per_state": per_state,
        "mean_R": mean_r,
        "n_states_scored": len(r_values),
    }


# ---------------------------------------------------------------------------
# Part A.2 — Transition-graph modularity / bridge-state check
# ---------------------------------------------------------------------------

def compute_transition_modularity(
    transition_counts: np.ndarray,
    state_ids: Optional[list[int]] = None,
    bridge_threshold: float = 0.4,
    seed: int = 42,
) -> dict:
    """Community-detect the state-transition graph and flag "bridge" states.

    Parameters
    ----------
    transition_counts : (n_states, n_states) array of raw transition counts
        (directed, need not be symmetric or row-normalized — this function
        symmetrizes internally).
    state_ids : labels for each row/col of ``transition_counts``; defaults
        to ``range(n_states)``.
    bridge_threshold : a state is flagged as a possible split when the
        fraction of its total transition weight crossing community
        boundaries exceeds this value.

    Returns
    -------
    dict: {"skipped", "reason", "modularity_Q", "communities": {state_id: community_id},
           "bridge_scores": {state_id: float}, "possible_split_states": [int, ...]}
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities, modularity as nx_modularity

    n = transition_counts.shape[0]
    if state_ids is None:
        state_ids = list(range(n))

    if n < 3:
        return {
            "skipped": True,
            "reason": "fewer than 3 states — community detection not meaningful",
            "modularity_Q": None,
            "communities": {},
            "bridge_scores": {},
            "possible_split_states": [],
        }

    sym = transition_counts + transition_counts.T
    np.fill_diagonal(sym, 0.0)

    if sym.sum() <= 0:
        return {
            "skipped": True,
            "reason": "no transitions between distinct states",
            "modularity_Q": None,
            "communities": {},
            "bridge_scores": {},
            "possible_split_states": [],
        }

    graph = nx.Graph()
    graph.add_nodes_from(state_ids)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(sym[i, j])
            if w > 0:
                graph.add_edge(state_ids[i], state_ids[j], weight=w)

    communities = louvain_communities(graph, weight="weight", seed=seed)
    q = float(nx_modularity(graph, communities, weight="weight"))

    community_of = {}
    for c_idx, members in enumerate(communities):
        for node in members:
            community_of[node] = c_idx

    bridge_scores = {}
    possible_split = []
    for idx, sid in enumerate(state_ids):
        own_community = community_of.get(sid)
        total_w = float(sym[idx, :].sum())
        if total_w <= 0:
            bridge_scores[sid] = 0.0
            continue
        cross_w = sum(
            float(sym[idx, j]) for j, other_sid in enumerate(state_ids)
            if community_of.get(other_sid) != own_community
        )
        score = cross_w / total_w
        bridge_scores[sid] = score
        if score > bridge_threshold:
            possible_split.append(int(sid))

    return {
        "skipped": False,
        "reason": None,
        "modularity_Q": q,
        "communities": {int(k): int(v) for k, v in community_of.items()},
        "bridge_scores": {int(k): float(v) for k, v in bridge_scores.items()},
        "possible_split_states": possible_split,
    }
