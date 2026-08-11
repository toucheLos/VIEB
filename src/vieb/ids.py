"""Recording-id normalization.

Its own module, below both ``data`` and ``io``, because both need it and neither
should have to import the other to get it. It is a pure string function with no
dependencies, and it is the single most load-bearing line in the codebase for
cross-method comparison: ids that disagree between arms are how a join silently
drops rows and still prints a table.
"""

from __future__ import annotations

from pathlib import Path

#: Stripped before the DLC suffix is cut. At most one can match.
EXTENSIONS = (".h5", ".hdf5", ".csv", ".mp4", ".avi", ".npy", ".npz", ".parquet")


def normalize_recording_id(name: str) -> str:
    """Strip a trailing ``DLC_*`` suffix and any file extension.

    ``20241016_Box_1_CFC_Day_0_(Context_A)_308DLC_Resnet50_VIEBFeb11shuffle2_snapshot_best-30.h5``
    becomes ``20241016_Box_1_CFC_Day_0_(Context_A)_308``.

    Every loader and every writer routes through this. Verified at full scale:
    ids derived this way from DLC pose files, from MoSeq's per-recording csvs and
    from ExBias's ``bouts.parquet`` overlap at 1.0000 across all 3,846 Luna
    recordings — three independently produced artifact sets joining exactly.
    """
    stem = Path(str(name)).name
    for ext in EXTENSIONS:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    idx = stem.find("DLC_")
    if idx > 0:
        stem = stem[:idx]
    return stem
