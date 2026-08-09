"""Cross-method comparison: the index join, the occupancy, and the ranking.

The cases here are the ones that would silently produce a wrong answer rather
than an error.

`verify_index` is the load-bearing one. It rebuilds `recording_idx -> path` by
re-globbing, which `recordings.py:1` warns is unsound in general because
`load_sessions` drops unreadable files and never persists the skip list. The
check that makes it sound for a given run is the frame-count match, so the test
that matters is that a *mismatch raises* -- a silent off-by-one would attribute
every recording's behavior to a neighbouring animal and still produce plausible
p-values.

`state_occupancy` has three traps: noise must stay out of the denominator (the
two families' `-1` mean different things), truncation must cut on the
checkpoint's own frame index rather than on row position, and a recording with
no assigned frames must not divide by zero.

`rank.normalize` must not score a missing axis as zero -- MoSeq has no partition
geometry, and "not measured" reading as "measured badly" would push the
reference arm down the table it defines.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from results_analysis.discriminate import (  # noqa: E402
    dedupe_rows, filter_states, state_occupancy, verify_index,
)
from results_analysis.rank import normalize  # noqa: E402
from results_analysis.collect import _local_exponent  # noqa: E402

DLC = "DLC_Resnet50_VIEBFeb11shuffle2_snapshot_best-30"


def _write_labels(path, labels, index, meta=None):
    np.savez_compressed(path, meta=json.dumps(meta or {}),
                        labels=np.asarray(labels, dtype=np.int64),
                        index=np.asarray(index, dtype=np.int64))


def _make_pose_csv(path, n_frames):
    """A DLC csv `recordings.frame_count` reports as exactly `n_frames`.

    `frame_count` counts lines minus DLC's 3-row MultiIndex header, so the
    header here has to be three rows or every count is off by a constant and
    the test passes for the wrong reason.
    """
    with open(path, "w") as fh:
        fh.write("scorer,m,m,m\nbodyparts,nose,nose,nose\n"
                 "coords,x,y,likelihood\n")
        for i in range(n_frames):
            fh.write(f"{i},1.0,2.0,0.9\n")


# ── verify_index ──────────────────────────────────────────────────────────────

def test_verify_index_accepts_an_exact_frame_count_match(tmp_path):
    for i, n in enumerate((5, 7, 3)):
        _make_pose_csv(tmp_path / f"rec_{i}.csv", n)
    paths = verify_index(str(tmp_path), [5, 7, 3])
    assert [os.path.basename(p) for p in paths] == \
        ["rec_0.csv", "rec_1.csv", "rec_2.csv"]


def test_verify_index_raises_when_a_frame_count_disagrees(tmp_path):
    """A skipped file shifts every later index; this is the only thing that
    catches it, so it must raise rather than warn."""
    for i, n in enumerate((5, 7, 3)):
        _make_pose_csv(tmp_path / f"rec_{i}.csv", n)
    with pytest.raises(ValueError, match="frame counts disagree"):
        verify_index(str(tmp_path), [5, 99, 3])


def test_verify_index_raises_when_the_file_count_changed(tmp_path):
    _make_pose_csv(tmp_path / "rec_0.csv", 5)
    with pytest.raises(ValueError, match="pose files"):
        verify_index(str(tmp_path), [5, 7])


# ── state_occupancy ───────────────────────────────────────────────────────────

def test_occupancy_excludes_noise_from_the_denominator(tmp_path):
    """Rows sum to 1 over *assigned* frames. Folding `-1` in would compare
    HDBSCAN's 'unclustered' against Koopman's 'near a separatrix'."""
    p = str(tmp_path / "labels.npz")
    _write_labels(p, [0, 0, 1, -1], [[0, 0], [0, 1], [0, 2], [0, 3]])
    occ, noise, _ = state_occupancy(p, 1)
    assert occ.shape == (1, 2)
    np.testing.assert_allclose(occ[0], [2 / 3, 1 / 3])
    np.testing.assert_allclose(noise[0], 0.25)


def test_occupancy_handles_a_recording_with_no_assigned_frames(tmp_path):
    p = str(tmp_path / "labels.npz")
    _write_labels(p, [0, -1, -1], [[0, 0], [1, 0], [1, 1]])
    occ, noise, _ = state_occupancy(p, 2)
    np.testing.assert_allclose(occ[1], [0.0])
    np.testing.assert_allclose(noise[1], 1.0)


def test_truncation_cuts_on_the_frame_index_not_row_position(tmp_path):
    """Recordings are interleaved in the checkpoint only by construction; the
    cut has to read `index[:,1]`, or one recording gets truncated and the next
    gets dropped entirely."""
    p = str(tmp_path / "labels.npz")
    _write_labels(p, [0, 1, 1, 0, 1, 1],
                  [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
    occ, _, _ = state_occupancy(p, 2, max_frames=2)
    np.testing.assert_allclose(occ[0], [0.5, 0.5])
    np.testing.assert_allclose(occ[1], [0.5, 0.5])


def test_truncation_is_a_no_op_when_the_limit_exceeds_every_recording(tmp_path):
    p = str(tmp_path / "labels.npz")
    _write_labels(p, [0, 1, 1], [[0, 0], [0, 1], [0, 2]])
    full, _, _ = state_occupancy(p, 1)
    trunc, _, _ = state_occupancy(p, 1, max_frames=10_000)
    np.testing.assert_allclose(full, trunc)


# ── dedupe ────────────────────────────────────────────────────────────────────

def test_dedupe_prefers_h5_and_collapses_the_duplicate_row():
    """1,079 of the 4,925 files are the same recordings in the other export
    format; keeping both weights those animals' sessions twice."""
    rid = "20241016_Box_1_CFC_Day_0_(Context_A)_308"
    occ = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    noise = np.array([0.1, 0.2, 0.3])
    paths = [f"/a/{rid}{DLC}.csv", f"/a/{rid}{DLC}.h5", f"/a/other{DLC}.h5"]
    occ2, noise2, rids2, fields2, dropped = dedupe_rows(
        occ, noise, [rid, rid, "other"], [{"a": 1}, {"a": 2}, {"a": 3}], paths)
    assert dropped == 1
    assert rids2 == [rid, "other"]
    # The h5 row survives, so its occupancy is the one kept.
    np.testing.assert_allclose(occ2[0], [0.0, 1.0])
    np.testing.assert_allclose(noise2[0], 0.2)


# ── filter_states ─────────────────────────────────────────────────────────────

def test_filter_states_drops_states_too_sparse_to_test():
    """Matches moseq_control's floor, so both are testing comparably-supported
    states rather than spending FDR budget on all-but-empty columns."""
    n = 80
    occ = np.zeros((n, 3))
    occ[:, 0] = 0.5          # present everywhere
    occ[:, 1] = 0.5          # present everywhere
    occ[:5, 2] = 0.5         # present in 5 recordings, below the floor of 50
    kept, ids = filter_states(occ)
    assert list(ids) == [0, 1]
    assert kept.shape == (n, 2)


# ── ranking ───────────────────────────────────────────────────────────────────

def _row(arm, **kw):
    base = {"arm": arm, "effect": None, "specificity": None, "coverage": None,
            "resolution": None, "parsimony": None, "cleanliness": None}
    base.update(kw)
    return base


def test_a_missing_axis_is_renormalized_not_scored_zero():
    """MoSeq has no partition geometry. If `resolution` counted as 0 for it,
    the reference arm would sink below arms it outperforms on every axis it
    actually has."""
    rows = normalize([
        _row("has-all", effect=1.0, coverage=1.0, resolution=1.0),
        _row("missing-resolution", effect=1.0, coverage=1.0),
    ])
    by = {r["arm"]: r for r in rows}
    assert by["missing-resolution"]["axes_missing"] == \
        ["specificity", "resolution", "parsimony", "cleanliness"]
    # Both are top of every axis they have, so both must score 1.0.
    assert by["has-all"]["composite"] == pytest.approx(1.0)
    assert by["missing-resolution"]["composite"] == pytest.approx(1.0)


def test_ranking_is_sorted_by_composite_and_numbered_from_one():
    rows = normalize([
        _row("low", effect=0.0, coverage=0.0),
        _row("high", effect=1.0, coverage=1.0),
        _row("mid", effect=0.5, coverage=0.5),
    ])
    assert [r["arm"] for r in rows] == ["high", "mid", "low"]
    assert [r["rank"] for r in rows] == [1, 2, 3]


def test_effect_outweighs_coverage():
    """The single judgement the ranking rests on. An arm with a large effect and
    mediocre coverage must beat one with a tiny effect and perfect coverage --
    otherwise diffusion-HDBSCAN's 35/37 on a two-point shift ranks first."""
    rows = normalize([
        _row("big-effect", effect=1.0, coverage=0.0),
        _row("all-significant", effect=0.0, coverage=1.0),
    ])
    assert rows[0]["arm"] == "big-effect"


# ── local exponent ────────────────────────────────────────────────────────────

def test_local_exponent_is_zero_on_a_plateau():
    """0 is what the §3 gate needed and did not get."""
    taus = [0.1, 0.2, 0.4, 0.8, 1.6]
    ex = _local_exponent(taus, [5.0] * 5)
    assert ex and all(e["exponent"] == pytest.approx(0.0) for e in ex)


def test_local_exponent_is_one_on_the_trivial_artifact():
    taus = [0.1, 0.2, 0.4, 0.8, 1.6]
    ex = _local_exponent(taus, list(taus))
    assert ex and all(e["exponent"] == pytest.approx(1.0) for e in ex)


def test_local_exponent_windows_span_at_least_four_times_the_lag():
    taus = [0.1, 0.2, 0.4, 0.8, 1.6]
    for e in _local_exponent(taus, [1.0, 2.0, 3.0, 4.0, 5.0]):
        assert e["tau_hi"] >= 4 * e["tau_lo"]
