"""Recording identity, deduplication, and filename metadata.

The cases here are the ones that actually bit. `normalize_id` has to agree with
ExBias's rule character-for-character or a cross-method join silently returns an
empty intersection rather than an error. `RECORDING_RE`'s optional `,_No_Shock`
group has to be present or the parser drops exactly the 298 retrieval-test
recordings -- the post-shock condition the whole analysis is about -- while
still reporting a plausible-looking 3,548 successes. And `dedupe` has to prefer
h5 deterministically, because the 1,079 csv files in the pose directory are the
same recordings as 1,079 of the h5 files, and counting them twice doubles their
weight in the stationary measure.

Luna filenames contain parentheses and commas, so every case here uses a
realistic name rather than a sanitised one.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation.recordings import (  # noqa: E402
    attach_lengths, build_manifest, dedupe, load_manifest, normalize_id,
    parse_id, save_manifest, validate_against_metadata,
)

DLC = "DLC_Resnet50_VIEBFeb11shuffle2_snapshot_best-30"


def test_normalize_id_strips_the_extension_and_the_dlc_suffix():
    """The exact rule from exbias.py:295 -- anything else joins nothing."""
    assert normalize_id(f"20241016_Box_1_CFC_Day_0_(Context_A)_308{DLC}.h5") == \
        "20241016_Box_1_CFC_Day_0_(Context_A)_308"
    assert normalize_id(f"/a/b/20241016_Box_1_CFC_Day_0_(Context_A)_308{DLC}.csv") == \
        "20241016_Box_1_CFC_Day_0_(Context_A)_308"


def test_normalize_id_is_idempotent_and_format_blind():
    """An already-normalized id must survive, and h5/csv/mp4 must agree.

    Idempotence is what lets a caller normalize defensively without having to
    track whether a given string has been through the function already.
    """
    stem = "20241017_Box_2_CFC_Day_1_(Context_A,_No_Shock)_311"
    assert normalize_id(stem) == stem
    ids = {normalize_id(f"{stem}{DLC}{ext}") for ext in (".h5", ".csv", ".mp4", ".npy")}
    assert ids == {stem}


def test_normalize_id_keeps_names_with_no_dlc_suffix_intact():
    """`find` returning -1 and 0 both mean 'no suffix' and must not truncate."""
    assert normalize_id("plain_name.h5") == "plain_name"
    assert normalize_id("DLC_leading.h5") == "DLC_leading"


def test_parse_id_reads_the_experimental_design_off_the_filename():
    fields = parse_id("20241016_Box_1_CFC_Day_0_(Context_A)_308")
    assert fields == {"date": "20241016", "box": 1, "experiment": "CFC",
                      "day": 0, "context": "A", "no_shock": False,
                      "animal": "308"}


def test_parse_id_handles_the_no_shock_recordings():
    """The 298 retrieval tests. Without the optional group the parser reports
    3,548/3,846 successes and drops precisely the post-shock condition."""
    fields = parse_id("20241017_Box_2_CFC_Day_1_(Context_A,_No_Shock)_311")
    assert fields is not None
    assert fields["no_shock"] is True
    assert fields["context"] == "A"
    assert fields["day"] == 1


def test_parse_id_returns_none_rather_than_guessing():
    assert parse_id("not_a_luna_filename") is None
    assert parse_id("") is None


def test_parse_id_types_day_and_box_as_ints_and_no_shock_as_bool():
    """Callers filter on `day >= 1`; a string would compare lexicographically
    and put day 10 before day 2."""
    fields = parse_id("20241016_Box_3_CFD_Day_10_(Context_B)_9001")
    assert fields["day"] == 10 and isinstance(fields["day"], int)
    assert fields["box"] == 3 and isinstance(fields["box"], int)
    assert fields["no_shock"] is False


def test_dedupe_prefers_h5_and_reports_the_losers():
    stem = "20241016_Box_1_CFC_Day_0_(Context_A)_308"
    paths = [f"/p/{stem}{DLC}.csv", f"/p/{stem}{DLC}.h5"]
    kept, dropped, ambiguous = dedupe(paths)
    assert kept == [f"/p/{stem}{DLC}.h5"]
    assert dropped == {stem: [f"/p/{stem}{DLC}.csv"]}
    assert ambiguous == []


def test_dedupe_is_independent_of_input_order():
    """Glob order varies between filesystems; the kept set must not."""
    stem = "20241016_Box_1_CFC_Day_0_(Context_A)_308"
    a = dedupe([f"/p/{stem}{DLC}.csv", f"/p/{stem}{DLC}.h5"])[0]
    b = dedupe([f"/p/{stem}{DLC}.h5", f"/p/{stem}{DLC}.csv"])[0]
    assert a == b


def test_dedupe_flags_a_same_extension_collision_instead_of_hiding_it():
    """Two h5 files for one recording is a real problem, not a format duplicate,
    so it must be surfaced rather than resolved by sort order."""
    stem = "20241016_Box_1_CFC_Day_0_(Context_A)_308"
    kept, _, ambiguous = dedupe([f"/x/{stem}{DLC}.h5", f"/y/{stem}{DLC}.h5"])
    assert len(kept) == 1
    assert ambiguous == [stem]


def test_dedupe_keeps_distinct_recordings_apart():
    paths = [f"/p/20241016_Box_1_CFC_Day_0_(Context_A)_308{DLC}.h5",
             f"/p/20241016_Box_2_CFC_Day_0_(Context_A)_311{DLC}.h5"]
    kept, dropped, _ = dedupe(paths)
    assert len(kept) == 2 and dropped == {}


def test_build_manifest_counts_duplicates_and_parses_every_row():
    stems = ["20241016_Box_1_CFC_Day_0_(Context_A)_308",
             "20241017_Box_2_CFC_Day_1_(Context_A,_No_Shock)_311",
             "20241120_Box_3_CFD_Day_5_(Context_B)_9001"]
    paths = [f"/p/{s}{DLC}.h5" for s in stems]
    paths += [f"/p/{stems[0]}{DLC}.csv"]          # one duplicate
    rows, report = build_manifest(paths, count_frames=False)

    assert report["n_input_files"] == 4
    assert report["n_recordings"] == 3
    assert report["n_duplicates_dropped"] == 1
    assert report["n_unparsed"] == 0
    assert all(r["parse_ok"] for r in rows)
    assert [r["idx"] for r in rows] == [0, 1, 2]
    assert {r["context"] for r in rows} == {"A", "B"}
    assert sum(r["no_shock"] for r in rows) == 1


def test_build_manifest_records_unparsed_rows_without_dropping_them():
    """An unrecognised name is still a recording. Dropping it would silently
    shift every later index -- the exact failure `skipped` already causes."""
    rows, report = build_manifest([f"/p/mystery{DLC}.h5"], count_frames=False)
    assert len(rows) == 1
    assert rows[0]["parse_ok"] is False
    assert rows[0]["context"] is None
    assert report["n_unparsed"] == 1


def test_attach_lengths_refuses_a_checkpoint_from_a_different_file_set():
    """The assertion that replaces the abandoned mask-out-duplicates path."""
    rows, _ = build_manifest(
        [f"/p/20241016_Box_1_CFC_Day_0_(Context_A)_308{DLC}.h5"],
        count_frames=False)
    with pytest.raises(ValueError, match="different file sets"):
        attach_lengths(rows, [5402, 5388])


def test_attach_lengths_records_the_post_load_count():
    rows, _ = build_manifest(
        [f"/p/20241016_Box_1_CFC_Day_0_(Context_A)_308{DLC}.h5"],
        count_frames=False)
    attach_lengths(rows, [5402])
    assert rows[0]["n_frames"] == 5402


def test_manifest_round_trips_through_disk_with_types_intact(tmp_path):
    stems = ["20241016_Box_1_CFC_Day_0_(Context_A)_308",
             "20241017_Box_2_CFC_Day_1_(Context_A,_No_Shock)_311"]
    rows, report = build_manifest([f"/p/{s}{DLC}.h5" for s in stems],
                                  count_frames=False)
    attach_lengths(rows, [5402, 5388])
    save_manifest(str(tmp_path), rows, report)
    back = load_manifest(str(tmp_path))

    assert [r["recording_id"] for r in back] == [r["recording_id"] for r in rows]
    assert back[0]["day"] == 0 and isinstance(back[0]["day"], int)
    assert back[1]["no_shock"] is True and back[0]["no_shock"] is False
    assert back[0]["n_frames"] == 5402


def test_validate_against_metadata_catches_a_disagreement(tmp_path):
    """metadata.csv is independent of the filename parser where they overlap,
    so a mismatch means one of the two is wrong."""
    stem = "20241016_Box_1_CFC_Day_0_(Context_A)_308"
    rows, _ = build_manifest([f"/p/{stem}{DLC}.h5"], count_frames=False)

    meta = tmp_path / "metadata.csv"
    meta.write_text(
        "filename,date,box,experiment,day,context,no_shock,animal_id,fear\n"
        f"{stem}.mp4,20241016,1,CFC,0,A,no,308,\n", encoding="utf-8")
    assert validate_against_metadata(rows, str(meta))["n_mismatches"] == 0

    meta.write_text(
        "filename,date,box,experiment,day,context,no_shock,animal_id,fear\n"
        f"{stem}.mp4,20241016,1,CFC,4,B,no,308,\n", encoding="utf-8")
    report = validate_against_metadata(rows, str(meta))
    assert report["n_checked"] == 1
    assert report["n_mismatches"] == 2          # day and context
    assert {m["field"] for m in report["mismatches"]} == {"day", "context"}
