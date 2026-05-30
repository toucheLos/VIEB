#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities extracted from gui.py for VIEB."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_MPL = False
try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    import matplotlib.cm as mpl_cm
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    _MPL = True
except Exception:
    pass

_CV2 = False
try:
    import cv2

    _CV2 = True
except Exception:
    pass

from PyQt5.QtGui import QImage, QPixmap

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
APP_CONFIG_PATH = ROOT / "app_config.json"
CONFIG_PATH = ROOT / "config.json"


def _get_results_dir_vc() -> Path:
    try:
        import vieb_config as _vc
        return Path(_vc.get_results_dir())
    except Exception:
        return ROOT / "results"


def _get_clips_dir_vc() -> Path:
    try:
        import vieb_config as _vc
        return Path(_vc.get_clips_dir())
    except Exception:
        return ROOT / "clips"


RESULTS = _get_results_dir_vc()
CLIPS = _get_clips_dir_vc()
VALIDATION_DIR = RESULTS / "validation"

# ---------------------------------------------------------------------------
# WSL2 GPU detection — cached at first use (detection is slow)
# ---------------------------------------------------------------------------

_WSL_CUML: bool | None = None   # None = not yet checked


def _wsl_path(win_path: str) -> str:
    """Convert a Windows absolute path to its WSL2 /mnt/... equivalent."""
    try:
        r = subprocess.run(
            ["wsl", "wslpath", "-u", win_path],
            capture_output=True, text=True, timeout=6,
        )
        p = r.stdout.strip()
        if p:
            return p
    except Exception:
        pass
    # Fallback: manual conversion
    p = win_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = "/mnt/" + p[0].lower() + p[2:]
    return p


def _wsl_python() -> str:
    """Return the WSL path to the venv_wsl Python interpreter."""
    return _wsl_path(str(ROOT / "venv_wsl" / "bin" / "python"))


def _probe_wsl_cuml() -> bool:
    """Return True if WSL2 is reachable and venv_wsl has cuML + a CUDA device."""
    if sys.platform != "win32":
        return False
    venv_py = ROOT / "venv_wsl" / "bin" / "python"
    try:
        exists = venv_py.exists()
    except OSError:
        # WinError 1920: WSL2 symlink exists but Windows can't stat it — treat as present
        exists = True
    if not exists:
        return False
    try:
        wsl_py = _wsl_python()
        result = subprocess.run(
            ["wsl", "bash", "-lc",
             f"{shlex.quote(wsl_py)} -c "
             f"'import cuml; import cupy; cupy.cuda.runtime.getDeviceCount(); print(\"cuml_ok\")'"],
            capture_output=True, text=True, timeout=20,
        )
        return "cuml_ok" in result.stdout
    except Exception:
        return False


def wsl_cuml_available() -> bool:
    """Cached check — only runs the probe once per process."""
    global _WSL_CUML
    if _WSL_CUML is None:
        _WSL_CUML = _probe_wsl_cuml()
    return _WSL_CUML


def wsl_cuml_reset_cache() -> None:
    """Force re-probe on next call (e.g. after user installs cuML in WSL2)."""
    global _WSL_CUML
    _WSL_CUML = None


# ---------------------------------------------------------------------------
# WSL2 step-detection helpers (used by the setup wizard)
# ---------------------------------------------------------------------------

def _wsl_check_installed() -> bool:
    """Return True if the wsl.exe command exists and WSL2 kernel is present."""
    try:
        r = subprocess.run(["wsl", "--version"], capture_output=True, timeout=8)
        return r.returncode == 0
    except Exception:
        return False


def _wsl_check_distro() -> bool:
    """Return True if at least one Linux distro is registered."""
    try:
        # wsl -l output is UTF-16-LE on Windows
        r = subprocess.run(["wsl", "-l", "-q"], capture_output=True, timeout=8)
        text = r.stdout.decode("utf-16-le", errors="ignore").strip()
        return bool(text)
    except Exception:
        return False


def _wsl_check_venv() -> bool:
    """Return True if venv_wsl/bin/python exists (via wsl ls to bypass WinError 1920)."""
    try:
        wsl_py = _wsl_python()
        r = subprocess.run(
            ["wsl", "bash", "-lc", f"test -f {shlex.quote(wsl_py)} && echo ok"],
            capture_output=True, text=True, timeout=8,
        )
        return "ok" in r.stdout
    except Exception:
        return False


def _wsl_elevate_install(extra_args: str = "") -> None:
    """Ask Windows to run `wsl <extra_args>` in an elevated prompt."""
    import ctypes
    cmd = f"wsl {extra_args}".strip()
    # ShellExecuteW with "runas" triggers a UAC prompt
    ctypes.windll.shell32.ShellExecuteW(
        0, "runas", "cmd.exe", f"/k {cmd}", None, 1
    )


STAGES = [
    {
        "id": 1,
        "name": "Pose Estimation (DLC)",
        "desc": "Run DeepLabCut analysis to generate pose CSV files for videos.",
        "cmd": "python setup_dlc_training.py --analyze",
    },
    {
        "id": 2,
        "name": "Feature Extraction",
        "desc": "Extract frame-level behavioral features from tracked keypoints.",
        "cmd": "python compare.py --extract [--no-wavelets]",
    },
    {
        "id": 3,
        "name": "Preprocessing",
        "desc": "Standardize pooled features across all videos.",
        "cmd": "python compare.py --cluster",
    },
    {
        "id": 4,
        "name": "UMAP Reduction",
        "desc": "Reduce feature dimensionality into a compact latent space.",
        "cmd": "python compare.py --cluster",
    },
    {
        "id": 5,
        "name": "HDBSCAN Clustering",
        "desc": "Discover behavioral states and identify noise frames.",
        "cmd": "python compare.py --cluster --min-cluster-size N",
    },
    {
        "id": 6,
        "name": "HMM Smoothing",
        "desc": "Smooth state assignments temporally while preserving noise labels.",
        "cmd": "python compare.py --cluster",
    },
    {
        "id": 7,
        "name": "State Collapsing (optional)",
        "desc": "Merge similar states using centroid distance and kinematic criteria.",
        "cmd": "python compare.py --collapse --collapse-threshold 0.5",
    },
    {
        "id": 8,
        "name": "Report Generation",
        "desc": "Build summary tables, transition outputs, and group comparison plots.",
        "cmd": "python compare.py --report",
    },
    {
        "id": 9,
        "name": "Per-Animal Scalars",
        "desc": "Compute freeze AUC and discrimination metrics for each animal.",
        "cmd": "python compare.py --summarize",
    },
    {
        "id": 10,
        "name": "Motif Discovery",
        "desc": "Find enriched bigram/trigram motifs between contexts.",
        "cmd": "python compare.py --motifs",
    },
    {
        "id": 11,
        "name": "Characterization + Clip Export",
        "desc": "Generate behavior profiles and optionally export exemplar clips.",
        "cmd": "python characterize.py [--clips]",
    },
]

_DEFAULT_CFG = {
    "arena_bounds": {"x_min": 0, "y_min": 0, "x_max": 1280, "y_max": 960},
    "results_dir": str(RESULTS),
    "raw_videos_dir": str(ROOT / "raw_videos"),
    "fps": 30,
    "window_size": [1280, 800],
    "last_view": "Overview",
    "min_cluster_size": 2000,
    "collapse_threshold": 0.5,
    "use_wavelets": True,
    "enable_state_collapse": False,
    "export_clips": False,
    "onboarding_complete": False,
    "project_name": "VIEB Project",
    "last_completed_stage": "",
    "stage_status": {},
    "stage_last_run": {},
    "context_groups": "A,B,C",
    "cohort_csv_path": "",
    "current_run_saved": False,
    "current_run_id": "",
}

_SPINNER = ["|", "/", "-", "\\"]
_NAV_VIEWS = [
    "Overview",
    "Pipeline",
    "Cluster Runs",
    "Browse States",
    "Analysis",
    "Validation",
    "Settings",
]

_NAV_ICONS = {
    "Overview":       "⊞",
    "Pipeline":       "▶",
    "Cluster Runs":   "⊙",
    "Browse States":  "▣",
    "Analysis":       "◈",
    "Validation":     "✓",
    "Settings":       "≡",
}

PROJECTS_JSON = ROOT / "projects.json"


def _load_projects() -> list:
    if PROJECTS_JSON.exists():
        try:
            return json.loads(PROJECTS_JSON.read_text(encoding="utf-8")).get("projects", [])
        except Exception:
            pass
    return []


def _save_projects(projects: list) -> None:
    PROJECTS_JSON.write_text(json.dumps({"projects": projects}, indent=2), encoding="utf-8")


def _register_project(path: str) -> None:
    """Add or move a project to the top of the recent-projects list."""
    projects = _load_projects()
    path = os.path.abspath(path)
    projects = [p for p in projects if os.path.abspath(p.get("path", "")) != path]
    projects.insert(0, {
        "name": os.path.basename(path),
        "path": path,
        "config": os.path.join(path, "config.yaml"),
        "added": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })
    _save_projects(projects[:20])  # keep at most 20 recent entries


def _get_project_config_path() -> Path:
    """Return the config.json path for the currently active project."""
    if APP_CONFIG_PATH.exists():
        try:
            app_cfg = json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
            active = app_cfg.get("active_project", "")
            if active:
                p = Path(active)
                if p.exists():
                    return p / "config.json"
        except Exception:
            pass
    return CONFIG_PATH


def _load_cfg():
    cfg = json.loads(json.dumps(_DEFAULT_CFG))
    path = _get_project_config_path()
    if path.exists():
        try:
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    if "arena_bounds" not in cfg:
        cfg["arena_bounds"] = dict(_DEFAULT_CFG["arena_bounds"])
    for k, v in _DEFAULT_CFG.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def _save_cfg(cfg):
    _get_project_config_path().write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _fmt_ts(ts):
    if not ts:
        return "-"
    if isinstance(ts, str):
        return ts
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _state_key(stage_id):
    return str(stage_id)


def _results_exist():
    return RESULTS.exists() and any(RESULTS.iterdir())


def _find_dlc_project():
    for p in ROOT.glob("VIEB-*/config.yaml"):
        return p.parent
    return None


def _derive_stage_statuses(data: dict) -> dict:
    """Return {str_stage_id: 'done'|'pending'} derived from result files on disk."""
    def _s(cond): return "done" if cond else "pending"
    fi = data.get("feature_index")
    ci = data.get("cluster_info")
    return {
        "1":  _s(bool(fi) and len(fi) > 0),
        "2":  _s(bool(fi)),
        "3":  _s(bool(ci)), "4": _s(bool(ci)), "5": _s(bool(ci)), "6": _s(bool(ci)),
        "7":  _s((RESULTS / "shared" / "collapse_mapping.json").exists()),
        "8":  _s(data.get("summary") is not None),
        "9":  _s(data.get("animal_scalars") is not None),
        "10": _s(data.get("motifs") is not None),
        "11": _s(data.get("state_summary") is not None),
    }


def _has_pose_csvs(raw_videos_dir: Path):
    if not raw_videos_dir.exists():
        return False
    return any(raw_videos_dir.glob("*DLC*.csv")) or any(raw_videos_dir.glob("*.csv"))


def _open_folder(path: Path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Human-readable log translation
# ---------------------------------------------------------------------------

_LOG_PATTERNS = [
    # (substring_to_match,  human_message)
    ("GPU detected",          "✓ GPU found — training and inference will be faster."),
    ("No GPU detected",       "ℹ No GPU found — running on CPU. Training will be slow."),
    ("Analyzing",             "⏳ Running pose estimation on videos…"),
    ("Analysis complete",     "✓ Pose estimation complete. All videos processed."),
    ("Extracting features",   "⏳ Extracting behavioral features from pose CSVs…"),
    ("Done. Extracted",       "✓ Feature extraction complete. Ready to cluster."),
    ("Fitting UMAP",          "⏳ Fitting UMAP dimensionality reduction (may take several minutes)…"),
    ("Fitting HDBSCAN",       "⏳ Fitting HDBSCAN clustering…"),
    ("Behavioral states discovered", None),  # pass-through (already readable)
    ("HMM smoother",          "⏳ Smoothing state assignments with HMM…"),
    ("Per-video labels",      "⏳ Saving per-video state labels…"),
    ("Summary table saved",   "✓ Report generated. Comparison plots written to results/comparison/"),
    ("Extracting motifs",     "⏳ Computing bigram/trigram enrichment between contexts…"),
    ("Motifs →",              "✓ Motif discovery complete."),
    ("Collapse mapping",      "✓ States collapsed. Labels rewritten."),
    ("Training complete",     "✓ Model training complete. Run Evaluate to check accuracy."),
    ("Evaluation complete",   "✓ Evaluation complete. Check evaluation-results/ folder."),
    ("[VIEB] Error:",         None),  # pass-through errors verbatim
]


def _translate_log(raw: str) -> str | None:
    """Return a human-readable message for raw CLI output, or None to use raw."""
    stripped = raw.strip()
    if not stripped:
        return None
    for pattern, msg in _LOG_PATTERNS:
        if pattern in stripped:
            return msg  # None means pass raw through
    return None  # default: pass raw through


def _state_colors(n: int):
    """Return tab20 RGBA array; index i always maps to the same colour regardless of n."""
    return mpl_cm.tab20(np.linspace(0, 1, max(1, n)))


def _thumb_from_video(path: Path | None, size=(180, 110)):
    if not _CV2:
        return None
    if path is None:
        return None
    cap = cv2.VideoCapture(str(path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.resize(frame, size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buf = frame.copy()
    img = QImage(buf.data, size[0], size[1], 3 * size[0], QImage.Format_RGB888)
    return QPixmap.fromImage(img)
