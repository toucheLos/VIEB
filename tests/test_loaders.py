"""Tests for the one pose loader (§6a) and recording-id normalization (§6b).

The failures these guard against are all silent: a duplicate recording counted
twice inflates a stationary measure, a frame-count mismatch shifts every later
index onto a neighbouring animal, and a fill convention chosen by accident
changes every downstream feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vieb.data.loaders import (
    DEFAULT_PREFERENCE,
    FrameCountMismatch,
    assert_id_overlap,
    dedupe,
    find_pose_files,
    frame_count,
    load_dataset,
    load_pose_file,
)

BODYPARTS = ["nose", "center", "tail_base"]
SUFFIX = "DLC_Resnet50_VIEBFeb11shuffle2_snapshot_best-30"


def write_dlc_csv(path, n_frames, bodyparts=BODYPARTS, seed=0, missing=None):
    """Write a DLC three-header-row csv."""
    rng = np.random.default_rng(seed)
    cols = {}
    for bp in bodyparts:
        for coord in ("x", "y", "likelihood"):
            cols[(("scorer", bp, coord))] = rng.random(n_frames)
    df = pd.DataFrame(cols)
    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=["scorer", "bodyparts", "coords"]
    )
    if missing is not None:
        df.iloc[missing, 0] = np.nan
    df.to_csv(path)
    return df


def write_flat_csv(path, n_frames, bodyparts=BODYPARTS, seed=0):
    """Write a hand-exported flat-column csv: one header row."""
    rng = np.random.default_rng(seed)
    data = {"frame": np.arange(n_frames)}
    for bp in bodyparts:
        data[f"{bp}_x"] = rng.random(n_frames)
        data[f"{bp}_y"] = rng.random(n_frames)
        data[f"{bp}_likelihood"] = rng.random(n_frames)
    pd.DataFrame(data).to_csv(path, index=False)


class TestFindPoseFiles:
    def test_finds_h5_and_csv(self, tmp_path):
        (tmp_path / "a.h5").touch()
        (tmp_path / "b.csv").touch()
        assert len(find_pose_files(tmp_path)) == 2

    @pytest.mark.parametrize(
        "name",
        ["CollectedData_x.csv", "a_meta.csv", "machinelabels_1.h5",
         "vid_full.h5", "metadata.csv"],
    )
    def test_skips_bookkeeping(self, tmp_path, name):
        # The union of v1's and v2's skip lists; neither covered the other's cases.
        (tmp_path / name).touch()
        (tmp_path / "real.h5").touch()
        found = find_pose_files(tmp_path)
        assert [f.rsplit("/", 1)[-1] for f in found] == ["real.h5"]

    def test_sorted_order(self, tmp_path):
        for n in ["c.h5", "a.h5", "b.h5"]:
            (tmp_path / n).touch()
        found = find_pose_files(tmp_path)
        assert found == sorted(found), "positional checkpoint indices assume sorted()"

    def test_missing_root_is_not_an_error(self, tmp_path):
        assert find_pose_files(tmp_path / "nope") == []


class TestDedupe:
    def test_h5_beats_csv_for_the_same_recording(self, tmp_path):
        # The 1,079-duplicate case: the same recording exported twice.
        stem = f"20241016_Box_1_CFC_Day_0_(Context_A)_308{SUFFIX}"
        (tmp_path / f"{stem}.h5").touch()
        (tmp_path / f"{stem}.csv").touch()
        kept, dropped, ambiguous = dedupe(find_pose_files(tmp_path))
        assert len(kept) == 1
        assert kept[0].endswith(".h5")
        assert len(dropped) == 1
        assert ambiguous == []

    def test_preference_order_is_h5_first(self):
        assert DEFAULT_PREFERENCE[0] == ".h5"

    def test_distinct_recordings_are_both_kept(self, tmp_path):
        (tmp_path / f"rec_308{SUFFIX}.h5").touch()
        (tmp_path / f"rec_309{SUFFIX}.h5").touch()
        kept, dropped, _ = dedupe(find_pose_files(tmp_path))
        assert len(kept) == 2 and dropped == {}

    def test_same_extension_collision_is_flagged(self, tmp_path):
        # A genuine collision, not a format duplicate — surfaced, not resolved
        # quietly, because it means two different files claim one recording.
        (tmp_path / "sub_a").mkdir()
        (tmp_path / "sub_b").mkdir()
        (tmp_path / "sub_a" / f"rec_308{SUFFIX}.h5").touch()
        (tmp_path / "sub_b" / f"rec_308{SUFFIX}.h5").touch()
        kept, dropped, ambiguous = dedupe(find_pose_files(tmp_path))
        assert len(kept) == 1
        assert ambiguous == ["rec_308"]

    def test_kept_order_is_by_recording_id_not_glob_order(self, tmp_path):
        for stem in ["zzz_308", "aaa_309"]:
            (tmp_path / f"{stem}{SUFFIX}.h5").touch()
        kept, _, _ = dedupe(find_pose_files(tmp_path))
        assert [k.rsplit("/", 1)[-1].split("DLC_")[0] for k in kept] == ["aaa_309", "zzz_308"]


class TestFrameCount:
    def test_dlc_csv_three_header_rows(self, tmp_path):
        p = tmp_path / "a.csv"
        write_dlc_csv(p, 40)
        assert frame_count(p) == 40

    def test_flat_csv_one_header_row(self, tmp_path):
        # v2's implementation hardcoded `lines - 3`, which is wrong by 2 here.
        p = tmp_path / "a.csv"
        write_flat_csv(p, 40)
        assert frame_count(p) == 40

    def test_raises_on_unsupported_extension(self, tmp_path):
        p = tmp_path / "a.txt"
        p.touch()
        with pytest.raises(FrameCountMismatch, match="unsupported extension"):
            frame_count(p)

    def test_matches_parsed_length(self, tmp_path):
        p = tmp_path / "a.csv"
        write_dlc_csv(p, 57)
        pose, _, _ = load_pose_file(p)
        assert frame_count(p) == pose.shape[0]


class TestLoadPoseFile:
    def test_dlc_csv_shape_and_bodyparts(self, tmp_path):
        p = tmp_path / "a.csv"
        write_dlc_csv(p, 20)
        pose, conf, bps = load_pose_file(p)
        assert pose.shape == (20, 3, 2)
        assert conf.shape == (20, 3)
        assert bps == BODYPARTS

    def test_flat_csv_dispatches_on_columns_not_extension(self, tmp_path):
        p = tmp_path / "a.csv"
        write_flat_csv(p, 20)
        pose, conf, bps = load_pose_file(p)
        assert pose.shape == (20, 3, 2)
        assert bps == BODYPARTS

    def test_fill_nan_leaves_gaps_unknown(self, tmp_path):
        p = tmp_path / "a.csv"
        write_dlc_csv(p, 20, missing=[5])
        pose, _, _ = load_pose_file(p, fill="nan")
        assert np.isnan(pose[5, 0, 0])

    def test_fill_zero_reproduces_v1_convention(self, tmp_path):
        # v1 filled a missing keypoint with "at the origin, fully confident".
        # It changes every downstream feature, so it is explicit, never default.
        p = tmp_path / "a.csv"
        write_dlc_csv(p, 20, missing=[5])
        pose, conf, _ = load_pose_file(p, fill="zero")
        assert pose[5, 0, 0] == 0.0
        assert not np.isnan(pose).any()
        assert not np.isnan(conf).any()

    def test_rejects_unknown_fill(self, tmp_path):
        p = tmp_path / "a.csv"
        write_dlc_csv(p, 5)
        with pytest.raises(ValueError, match="fill must be"):
            load_pose_file(p, fill="whatever")


class TestLoadDataset:
    def _make_project(self, tmp_path, n=3, frames=30, duplicate=False):
        for i in range(n):
            stem = f"20241016_Box_{i + 1}_CFC_Day_0_(Context_A)_30{i}{SUFFIX}"
            write_dlc_csv(tmp_path / f"{stem}.csv", frames + i, seed=i)
            if duplicate:
                write_dlc_csv(tmp_path / f"{stem}_copy.csv", frames + i, seed=i)
        return tmp_path

    def test_builds_a_boundary_aware_dataset(self, tmp_path):
        self._make_project(tmp_path, n=3, frames=30)
        data, report = load_dataset(tmp_path, fps=30.0)
        assert data.n_recordings == 3
        assert data.n_frames == 30 + 31 + 32
        assert data.boundaries().tolist() == [0, 30, 61, 93]
        assert report["n_loaded"] == 3

    def test_recording_ids_are_normalized(self, tmp_path):
        self._make_project(tmp_path, n=1)
        data, _ = load_dataset(tmp_path, fps=30.0)
        assert data.recording_ids == ["20241016_Box_1_CFC_Day_0_(Context_A)_300"]
        assert "DLC_" not in data.recording_ids[0]

    def test_deduplicate_false_keeps_every_file(self, tmp_path):
        # The legacy selector: the koopman_* runs were fit on the pre-dedup set.
        self._make_project(tmp_path, n=2, duplicate=True)
        deduped, _ = load_dataset(tmp_path, fps=30.0, deduplicate=True)
        legacy, rep = load_dataset(tmp_path, fps=30.0, deduplicate=False)
        assert deduped.n_recordings == 2
        assert legacy.n_recordings == 4
        assert rep["deduplicated"] is False

    def test_frame_count_mismatch_raises_not_warns(self, tmp_path):
        self._make_project(tmp_path, n=2, frames=30)
        # Truncate one file's body so its declared count no longer matches.
        victim = sorted(tmp_path.glob("*.csv"))[0]
        lines = victim.read_text().splitlines()
        victim.write_text("\n".join(lines[:-5]) + "\n")

        # The parse and the count move together for a truncated csv, so force the
        # disagreement the check exists for by appending a bodiless line.
        victim.write_text(victim.read_text() + "\n")
        with pytest.raises(FrameCountMismatch, match="disagree with their own frame"):
            load_dataset(tmp_path, fps=30.0)

    def test_verify_can_be_disabled(self, tmp_path):
        self._make_project(tmp_path, n=1)
        data, _ = load_dataset(tmp_path, fps=30.0, verify_frame_counts=False)
        assert data.n_recordings == 1

    def test_limit_takes_a_subset(self, tmp_path):
        self._make_project(tmp_path, n=4)
        data, _ = load_dataset(tmp_path, fps=30.0, limit=2)
        assert data.n_recordings == 2

    def test_mixed_bodyparts_raises(self, tmp_path):
        write_dlc_csv(tmp_path / f"a_301{SUFFIX}.csv", 20, bodyparts=BODYPARTS)
        write_dlc_csv(tmp_path / f"b_302{SUFFIX}.csv", 20, bodyparts=["nose", "center"])
        with pytest.raises(ValueError, match="uniform in its keypoints"):
            load_dataset(tmp_path, fps=30.0)

    def test_empty_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no pose files"):
            load_dataset(tmp_path, fps=30.0)

    def test_fps_reaches_the_dataset(self, tmp_path):
        self._make_project(tmp_path, n=1)
        data, _ = load_dataset(tmp_path, fps=250.0)
        assert data.fps == 250.0
        assert data.seconds_to_frames(0.5) == 125


class TestAssertIdOverlap:
    def test_passes_when_ids_agree(self):
        ids = [f"rec{i}" for i in range(100)]
        assert assert_id_overlap(ids, list(ids)) == 1.0

    def test_raises_on_normalization_drift(self):
        # The exact failure: one arm stripped DLC_, the other did not. A join
        # would return zero rows and the comparison would still print a table.
        a = [f"rec{i}" for i in range(100)]
        b = [f"rec{i}{SUFFIX}" for i in range(100)]
        with pytest.raises(ValueError, match="normalization drift"):
            assert_id_overlap(a, b, names=("v1", "v2"))

    def test_tolerates_small_differences(self):
        a = [f"rec{i}" for i in range(100)]
        b = [f"rec{i}" for i in range(95)]
        assert assert_id_overlap(a, b) == pytest.approx(0.95)

    def test_empty_side_raises(self):
        with pytest.raises(ValueError, match="one side is empty"):
            assert_id_overlap(["a"], [])
