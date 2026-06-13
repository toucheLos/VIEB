#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VIEB GUI - Video Interpreter for Experimental Behavior."""

from __future__ import annotations

import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from PyQt5.QtCore import QFileSystemWatcher, QObject, QThread, QTimer, Qt, pyqtSignal
    from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QKeySequence, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QButtonGroup,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QRadioButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QStackedWidget,
        QStatusBar,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("PyQt5 not found. Install with: pip install PyQt5")
    sys.exit(1)

_MPL = False
# All matplotlib globals — populated lazily on first call to _init_mpl()
mpl_cm = None
mpimg = None
plt = None
PdfPages = None
FigureCanvas = None
Figure = None


def _init_mpl() -> bool:
    """Import matplotlib on first call; subsequent calls return the cached result."""
    global _MPL, mpl_cm, mpimg, plt, PdfPages, FigureCanvas, Figure
    if _MPL:
        return True
    try:
        import matplotlib as _mpl
        _mpl.use("Qt5Agg")
        import matplotlib.cm as _cm
        import matplotlib.image as _img
        import matplotlib.pyplot as _plt
        from matplotlib.backends.backend_pdf import PdfPages as _PdfPages
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FC
        from matplotlib.figure import Figure as _Fig
        mpl_cm = _cm
        mpimg = _img
        plt = _plt
        PdfPages = _PdfPages
        FigureCanvas = _FC
        Figure = _Fig
        _MPL = True
    except Exception:
        pass
    return _MPL

_CV2 = False
try:
    import cv2
    _CV2 = True
except Exception:
    pass

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"
APP_CONFIG_PATH = ROOT / "app_config.json"


def _results_path() -> Path:
    import vieb_config
    return Path(vieb_config.get_results_dir())


def _clips_path() -> Path:
    import vieb_config
    return Path(vieb_config.get_clips_dir())


def _validation_path() -> Path:
    return _results_path() / "validation"


RESULTS = _results_path()
CLIPS = _clips_path()
VALIDATION_DIR = _validation_path()

try:
    from views.analysis import AnalysisView as _AnalysisView
    _HAS_ANALYSIS_VIEW = True
except Exception:
    _AnalysisView = None
    _HAS_ANALYSIS_VIEW = False

try:
    from views.cluster_runs import ClusterRunsView as _ClusterRunsView
    _HAS_CLUSTER_RUNS_VIEW = True
except Exception:
    _ClusterRunsView = None
    _HAS_CLUSTER_RUNS_VIEW = False

from views.help import HelpView

# ---------------------------------------------------------------------------
# WSL2 / Linux GPU detection - cached at first use
# ---------------------------------------------------------------------------

_WSL_CUML: bool | str | None = None  # True=working, "installed"=cuml installed but CUDA init failed, False=not installed
_linux_gpu_name: str | None = None  # populated by _probe_gpu_async on Linux


def _probe_nvidia_smi() -> str | None:
    """Return the first GPU name from nvidia-smi, or None if unavailable."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        name = r.stdout.strip().splitlines()[0].strip()
        return name if name else None
    except Exception:
        return None


def _probe_linux_cuml():
    """Return True if cuML is usable, 'installed' if installed but CUDA init fails, False if not installed."""
    try:
        import cuml  # noqa: F401
    except ImportError:
        return False
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        cp.array([1.0])
        return True
    except Exception:
        return "installed"


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
    p = win_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        p = "/mnt/" + p[0].lower() + p[2:]
    return p


def _wsl_python() -> str:
    return _wsl_path(str(ROOT / "venv_wsl" / "bin" / "python"))


def _probe_wsl_cuml() -> bool:
    """Return True if WSL2 is reachable and venv_wsl has cuML + CUDA."""
    if sys.platform != "win32":
        return False
    venv_py = ROOT / "venv_wsl" / "bin" / "python"
    try:
        exists = venv_py.exists()
    except OSError:
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
    global _WSL_CUML
    if _WSL_CUML is None:
        _WSL_CUML = _probe_wsl_cuml()
    return _WSL_CUML is True


def wsl_cuml_reset_cache() -> None:
    global _WSL_CUML
    _WSL_CUML = None


def _wsl_check_installed() -> bool:
    try:
        r = subprocess.run(["wsl", "--version"], capture_output=True, timeout=8)
        return r.returncode == 0
    except Exception:
        return False


def _probe_torch_cuda() -> bool:
    """Return True if torch is installed and reports a usable CUDA GPU.

    Used on Linux/macOS instead of the WSL2 + cuML probe.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _wsl_check_distro() -> bool:
    try:
        r = subprocess.run(["wsl", "-l", "-q"], capture_output=True, timeout=8)
        text = r.stdout.decode("utf-16-le", errors="ignore").strip()
        return bool(text)
    except Exception:
        return False


def _wsl_check_venv() -> bool:
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
    import ctypes
    cmd = f"wsl {extra_args}".strip()
    ctypes.windll.shell32.ShellExecuteW(0, "runas", "cmd.exe", f"/k {cmd}", None, 1)


def _transition_matrix_cols(df: pd.DataFrame, requested_n: int = 0) -> tuple[list[str], int]:
    """Return transition columns and a matrix size that match the saved table."""
    if df is None or df.empty:
        return [], 0

    pairs = []
    for col in df.columns:
        parts = col.split("_")
        if len(parts) == 3 and parts[0] == "trans" and parts[1].isdigit() and parts[2].isdigit():
            pairs.append((int(parts[1]), int(parts[2])))

    if not pairs:
        return [], 0

    available = {f"trans_{i}_{j}" for i, j in pairs}
    if requested_n > 0:
        requested_cols = [f"trans_{i}_{j}" for i in range(requested_n) for j in range(requested_n)]
        if all(col in available for col in requested_cols):
            return requested_cols, requested_n

    inferred_n = max(max(i, j) for i, j in pairs) + 1
    inferred_cols = [f"trans_{i}_{j}" for i in range(inferred_n) for j in range(inferred_n)]
    if all(col in available for col in inferred_cols):
        return inferred_cols, inferred_n

    square_n = int(math.sqrt(len(pairs)))
    if square_n * square_n == len(pairs):
        sorted_cols = [f"trans_{i}_{j}" for i, j in sorted(pairs)]
        return sorted_cols, square_n

    return [], 0


def _state_fraction_cols(df: pd.DataFrame, n: int | None = None) -> list[str]:
    if df is None or df.empty:
        return []
    if n is not None and n > 0:
        return [f"state_{i}_frac" for i in range(n) if f"state_{i}_frac" in df.columns]
    return [c for c in df.columns if c.startswith("state_") and c.endswith("_frac")]


def _dominant_state(summary: pd.DataFrame, n: int | None = None) -> int | None:
    cols = _state_fraction_cols(summary, n)
    if not cols:
        return None
    means = summary[cols].mean()
    if means.empty:
        return None
    return int(means.idxmax().split("_")[1])


def _state_means(summary: pd.DataFrame, n: int, hide_leading: bool = False) -> tuple[list[int], list[float], int | None]:
    cols = _state_fraction_cols(summary, n)
    if not cols:
        return [], [], None
    lead = _dominant_state(summary, n)
    state_ids = [int(c.split("_")[1]) for c in cols]
    vals = [float(summary[c].mean()) for c in cols]
    if hide_leading and lead is not None:
        kept = [(sid, val) for sid, val in zip(state_ids, vals) if sid != lead]
        state_ids = [sid for sid, _ in kept]
        vals = [val for _, val in kept]
        total = sum(vals)
        if total > 0:
            vals = [v / total for v in vals]
    return state_ids, vals, lead


STAGES = [
    {
        "id": 0,
        "name": "Environment Setup",
        "desc": "Create the Python virtual environment and configure GPU acceleration (optional).",
        "cmd": "python setup.py",
    },
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
        "name": "Preprocessing · UMAP · Clustering · Smoothing",
        "desc": "Standardize features, reduce with UMAP, cluster with HDBSCAN, then smooth labels with HMM.",
        "cmd": "python compare.py --cluster --min-cluster-size N --umap-dims N [--hdbscan-min-samples N] [--validate]",
    },
    {
        "id": 7,
        "name": "State Collapsing (optional)",
        "desc": "Merge states whose centroids exceed a cosine similarity threshold in full feature space.",
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
        "name": "Generate Clips",
        "desc": "Export exemplar video clips for each behavioral state.",
        "cmd": "python generate_clips.py",
    },
]

_STAGE_BY_ID = {s["id"]: s for s in STAGES}

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
    "context_descriptions": {},
    "cohort_csv_path": "",
    "metadata_csv_path": "",
    "hdbscan_min_samples": 0,
    "umap_dims": 10,
    "validate": False,
    "min_confidence": 0.7,
    "diagnose_mcs": "",
    "umap_sweep": False,
    "hdbscan_jobs": 1,
    "current_run_saved": False,
    "current_run_id": "",
    "column_map": {
        "animal_id":  "animal_id",
        "day":        "day",
        "context":    "context",
        "experiment": "experiment",
        "cohort":     "",
        "event":      "",
    },
    "object_keypoints": [],
    "condition_a_label": "",
    "condition_b_label": "",
    "primary_metric_label": "",
    "reviewer_categories": [],
    "reviewer_seed": 0,
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
    "Help",
]

_NAV_ICONS = {
    "Overview":       "⊞",
    "Pipeline":       "▶",
    "Cluster Runs":   "⊙",
    "Browse States":  "▣",
    "Analysis":       "◈",
    "Validation":     "✓",
    "Settings":       "≡",
    "Help":           "?",
}


def _load_app_config() -> dict:
    if APP_CONFIG_PATH.exists():
        try:
            return json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"projects": [], "active_project": "", "default_project": ""}


def _save_app_config(app_cfg: dict) -> None:
    APP_CONFIG_PATH.write_text(json.dumps(app_cfg, indent=2), encoding="utf-8")


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


def _migrate_to_project() -> Path:
    """Create projects/luna_fear_conditioning/ from the existing ROOT/config.json.

    Called silently on first launch when no app_config.json exists.
    Returns the new project directory path.
    """
    projects_dir = ROOT / "projects"
    projects_dir.mkdir(exist_ok=True)

    # Read existing config (or use defaults)
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            cfg = json.loads(json.dumps(_DEFAULT_CFG))
    else:
        cfg = json.loads(json.dumps(_DEFAULT_CFG))

    # Ensure paths are absolute
    for key in ("results_dir", "raw_videos_dir"):
        val = cfg.get(key, "")
        if val and not Path(val).is_absolute():
            cfg[key] = str((ROOT / val).resolve())

    # Derive project folder name from project_name or default
    project_name = cfg.get("project_name", "VIEB Project")
    import re as _re
    slug = project_name.lower().strip()
    slug = _re.sub(r"[^\w\s-]", "", slug)
    slug = _re.sub(r"[\s_-]+", "_", slug)
    slug = slug.strip("_") or "vieb_project"

    project_dir = projects_dir / slug
    # If directory already exists (e.g. a previous partial migration), use it
    project_dir.mkdir(exist_ok=True)

    project_cfg_path = project_dir / "config.json"
    project_cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    return project_dir


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


class _Capture(QObject):
    text = pyqtSignal(str)
    encoding = "utf-8"
    errors = "replace"

    def write(self, s):
        if s:
            self.text.emit(s)

    def flush(self):
        pass

    def isatty(self):
        return False


class _MplCanvasStub(QWidget):
    """Fallback canvas used when matplotlib is not available."""
    def __init__(self, parent=None, figsize=(6, 4)):  # noqa: ARG002
        super().__init__(parent)
        self.fig = None
        self.ax = None

    def draw(self):
        pass


_MplCanvasReal = None  # cached real class, created after first successful _init_mpl()


def MplCanvas(parent=None, figsize=(6, 4)):
    """Factory that returns a real FigureCanvas when matplotlib is available."""
    global _MplCanvasReal
    if _init_mpl():
        if _MplCanvasReal is None:
            class _Impl(FigureCanvas):
                def __init__(self, _parent=None, _figsize=(6, 4)):
                    self.fig = Figure(figsize=_figsize, tight_layout=True)
                    super().__init__(self.fig)
                    self.setParent(_parent)
                    self.ax = self.fig.add_subplot(111)
            _MplCanvasReal = _Impl
        return _MplCanvasReal(_parent=parent, _figsize=figsize)
    return _MplCanvasStub(parent=parent, figsize=figsize)


class DataLoader(QThread):
    loaded = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, cohort_csv_path: str = ""):
        super().__init__()
        self._cohort_path = cohort_csv_path

    def run(self):
        data = {}
        try:
            def _csv(rel):
                p = RESULTS / rel
                return pd.read_csv(p) if p.exists() else None

            def _json(rel):
                p = RESULTS / rel
                return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

            data["summary"] = _csv("comparison/summary_table.csv")
            data["state_summary"] = _csv("characterization/state_summary.csv")
            data["context_report"] = _csv("characterization/context_report.csv")
            data["transition_table"] = _csv("comparison/transition_table.csv")
            data["bouts"] = _csv("characterization/bouts.csv")
            data["motifs"] = _csv("comparison/motifs.csv")
            data["cluster_info"] = _json("shared/cluster_info.json")
            data["feature_index"] = _json("features/index.json")
            data["animal_scalars"] = _csv("comparison/animal_scalars.csv")
            data["fingerprints"] = _csv("comparison/behavioral_fingerprints.csv")
            data["deviation_scores"] = _csv("comparison/deviation_scores.csv")
            data["reverse_results"] = (
                json.loads((RESULTS / "comparison" / "reverse_model_results.json")
                           .read_text(encoding="utf-8"))
                if (RESULTS / "comparison" / "reverse_model_results.json").exists()
                else None
            )
            data["labels_per_frame"] = _csv("characterization/labels_per_frame.csv")
            data["validation_labels"] = _csv("validation/frame_labels.csv")
            data["validation_sample"] = _csv("validation/current_sample.csv")
            import vieb_config as _vc_dl
            meta_p = Path(_vc_dl.get_metadata_path())
            data["metadata"] = pd.read_csv(meta_p) if meta_p.exists() else None
            data["cohort"] = None
            if self._cohort_path:
                cp = Path(self._cohort_path)
                if cp.exists():
                    try:
                        if cp.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
                            from cohort_loader import load_cohort_excel
                            data["cohort"] = load_cohort_excel(str(cp))
                        else:
                            data["cohort"] = pd.read_csv(cp)
                    except Exception:
                        pass
        except Exception as e:
            self.error.emit(str(e))
            return
        self.loaded.emit(data)


class PipelineRunner(QThread):
    log = pyqtSignal(str)
    stage_started = pyqtSignal(int)
    stage_done = pyqtSignal(int, bool)
    all_done = pyqtSignal(bool)

    def __init__(self, stage_ids: list[int], cfg: dict):
        super().__init__()
        self.stage_ids = stage_ids
        self.cfg = cfg
        self._proc: subprocess.Popen | None = None
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _run_subprocess(self, args):
        if self._stop_flag:
            return False
        p = subprocess.Popen(
            [sys.executable, *args],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._proc = p
        assert p.stdout is not None
        for line in p.stdout:
            if self._stop_flag:
                p.terminate()
                break
            self.log.emit(line)
        rc = p.wait()
        self._proc = None
        return rc == 0 and not self._stop_flag

    def _run_cluster_wsl(self, fps: float, mcs: int) -> bool:
        if self._stop_flag:
            return False
        wsl_py = _wsl_python()
        wsl_cwd = _wsl_path(str(ROOT))
        cmd = (
            f"cd {shlex.quote(wsl_cwd)} && "
            f"{shlex.quote(wsl_py)} compare.py --cluster "
            f"--fps {fps} --min-cluster-size {mcs}"
        )
        self.log.emit("[GPU] Delegating clustering to WSL2 (cuML UMAP + HDBSCAN)…\n")
        p = subprocess.Popen(
            ["wsl", "bash", "-lc", cmd],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._proc = p
        assert p.stdout is not None
        for line in p.stdout:
            if self._stop_flag:
                p.terminate()
                break
            self.log.emit(line)
        rc = p.wait()
        self._proc = None
        return rc == 0 and not self._stop_flag

    def run(self):
        ok_all = True
        try:
            fps = float(self.cfg.get("fps", 30))
            mcs = int(self.cfg.get("min_cluster_size", 50))
            collapse_threshold = float(self.cfg.get("collapse_threshold", 0.5))
            use_wavelets = bool(self.cfg.get("use_wavelets", True))
            enable_collapse = bool(self.cfg.get("enable_state_collapse", False))
            export_clips = bool(self.cfg.get("export_clips", False))
            hdbscan_min_samples = int(self.cfg.get("hdbscan_min_samples", 0)) or None
            umap_dims = int(self.cfg.get("umap_dims", 10))
            validate = bool(self.cfg.get("validate", False))
            min_confidence = float(self.cfg.get("min_confidence", 0.7))

            for sid in self.stage_ids:
                if self._stop_flag:
                    self.log.emit("[stopped] Pipeline stopped by user.\n")
                    break
                if sid == 7 and not enable_collapse:
                    self.stage_done.emit(7, True)
                    continue

                if sid == 3:
                    self.stage_started.emit(3)
                    try:
                        if wsl_cuml_available() and sys.platform == "win32":
                            ok = self._run_cluster_wsl(fps, mcs)
                        else:
                            cluster_args = [
                                "compare.py", "--cluster",
                                "--fps", str(fps),
                                "--min-cluster-size", str(mcs),
                                "--umap-dims", str(umap_dims),
                            ]
                            if hdbscan_min_samples:
                                cluster_args += ["--hdbscan-min-samples", str(hdbscan_min_samples)]
                            if validate:
                                cluster_args.append("--validate")
                            ok = self._run_subprocess(cluster_args)
                        if ok:
                            self.stage_done.emit(3, True)
                        else:
                            raise RuntimeError("Clustering subprocess returned non-zero exit code.")
                    except Exception:
                        self.log.emit(traceback.format_exc())
                        self.stage_done.emit(3, False)
                        ok_all = False
                        break
                    continue

                self.stage_started.emit(sid)
                try:
                    if sid == 1:
                        ok = self._run_subprocess(["setup_dlc_training.py", "--analyze"])
                        if not ok:
                            raise RuntimeError("Pose estimation failed.")
                    elif sid == 2:
                        extract_args = ["compare.py", "--extract", "--fps", str(fps)]
                        if not use_wavelets:
                            extract_args.append("--no-wavelets")
                        ok = self._run_subprocess(extract_args)
                        if not ok:
                            raise RuntimeError("Feature extraction failed.")
                    elif sid == 7:
                        ok = self._run_subprocess([
                            "compare.py", "--collapse",
                            "--collapse-threshold", str(collapse_threshold),
                        ])
                        if not ok:
                            raise RuntimeError("State collapsing failed.")
                    elif sid == 8:
                        ok = self._run_subprocess([
                            "compare.py", "--report", "--fps", str(fps),
                            "--min-confidence", str(min_confidence),
                        ])
                        if not ok:
                            raise RuntimeError("Report generation failed.")
                    elif sid == 9:
                        ok = self._run_subprocess(["compare.py", "--summarize"])
                        if not ok:
                            raise RuntimeError("Per-animal scalar computation failed.")
                    elif sid == 10:
                        self.log.emit("[info] Stage 10 (Motif Discovery) is not yet available — skipping.\n")
                    elif sid == 11:
                        char_args = ["characterize.py", "--fps", str(fps)]
                        if export_clips:
                            char_args.append("--clips")
                        ok = self._run_subprocess(char_args)
                        if not ok:
                            raise RuntimeError("Characterization failed.")
                    self.stage_done.emit(sid, True)
                except (Exception, SystemExit) as _exc:
                    msg = (
                        f"Stage {sid} exited: {_exc}\n"
                        if isinstance(_exc, SystemExit)
                        else traceback.format_exc()
                    )
                    self.log.emit(msg)
                    self.stage_done.emit(sid, False)
                    ok_all = False
                    break
        except Exception:
            self.log.emit(traceback.format_exc())
            ok_all = False
        self.all_done.emit(ok_all)


class SubprocessWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(bool)

    def __init__(self, args: list[str]):
        super().__init__()
        self.args = args

    def run(self):
        try:
            p = subprocess.Popen(
                [sys.executable, *self.args],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert p.stdout is not None
            for line in p.stdout:
                self.log.emit(line)
            self.done.emit(p.wait() == 0)
        except Exception:
            self.log.emit(traceback.format_exc())
            self.done.emit(False)


class ClipGenerationWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(bool)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def run(self):
        cap = _Capture()
        cap.text.connect(self.log)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = cap
        ok = False
        try:
            sys.path.insert(0, str(ROOT))
            from characterize import cmd_clips

            cmd_clips(fps=float(self.cfg.get("fps", 30)))
            ok = True
        except Exception:
            print(traceback.format_exc())
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        self.done.emit(ok)


class ExportWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(bool)

    def __init__(self, options: dict, destination: Path):
        super().__init__()
        self.options = options
        self.destination = destination

    def _copy_file(self, rel):
        src = RESULTS / rel
        if src.exists():
            dst = self.destination / src.name
            shutil.copy2(src, dst)
            self.log.emit(f"Copied: {src} -> {dst}\n")

    def _copy_plot_pngs(self):
        comp = RESULTS / "comparison"
        if not comp.exists():
            return
        for p in comp.glob("*.png"):
            shutil.copy2(p, self.destination / p.name)
            self.log.emit(f"Copied plot: {p.name}\n")

    def _copy_clips(self):
        if CLIPS.exists():
            dst = self.destination / "clips"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(CLIPS, dst)
            self.log.emit("Copied clips directory.\n")

    def run(self):
        try:
            self.destination.mkdir(parents=True, exist_ok=True)
            if self.options.get("summary"):
                self._copy_file(Path("comparison") / "summary_table.csv")
            if self.options.get("animal"):
                self._copy_file(Path("comparison") / "animal_scalars.csv")
            if self.options.get("states"):
                self._copy_file(Path("characterization") / "state_summary.csv")
            if self.options.get("transitions"):
                self._copy_file(Path("comparison") / "transition_table.csv")
            if self.options.get("motifs"):
                self._copy_file(Path("comparison") / "motifs.csv")
            if self.options.get("plots"):
                self._copy_plot_pngs()
            if self.options.get("clips"):
                self._copy_clips()
            if self.options.get("pdf"):
                pdf = export_pdf_report()
                shutil.copy2(pdf, self.destination / pdf.name)
                self.log.emit(f"Copied report: {pdf.name}\n")
            self.done.emit(True)
        except Exception:
            self.log.emit(traceback.format_exc())
            self.done.emit(False)


class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._fps = 30.0
        self._speed = 1.0
        self._total = 0
        self._cur = 0
        self._loop = False
        self._playing = False
        self._frame_buf = None
        self.setFocusPolicy(Qt.StrongFocus)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._display = QLabel("No video loaded", alignment=Qt.AlignCenter)
        self._display.setMinimumSize(320, 220)
        self._display.setStyleSheet("background:#111;color:#999;")
        lay.addWidget(self._display)

        ctrl = QHBoxLayout()
        self._btn_play = QPushButton("Play")
        self._btn_play.clicked.connect(self.toggle_play)
        ctrl.addWidget(self._btn_play)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.sliderMoved.connect(self.seek)
        ctrl.addWidget(self._slider)

        self._loop_btn = QCheckBox("Loop")
        self._loop_btn.toggled.connect(lambda v: setattr(self, "_loop", v))
        ctrl.addWidget(self._loop_btn)

        ctrl.addWidget(QLabel("Speed"))
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.25x", "0.5x", "1x"])
        self._speed_combo.setCurrentText("1x")
        self._speed_combo.currentTextChanged.connect(self._set_speed)
        ctrl.addWidget(self._speed_combo)
        lay.addLayout(ctrl)

    def _set_speed(self, text):
        val = text.replace("x", "")
        self._speed = float(val)
        if self._playing:
            self.play()

    def load(self, path: str):
        if not _CV2:
            self._display.setText("OpenCV unavailable")
            return
        self.pause()
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            self._display.setText("Cannot open clip")
            return
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._slider.setMaximum(max(0, self._total - 1))
        self._show(0)

    def _show(self, idx):
        if not self._cap:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            return
        h, w = frame.shape[:2]
        max_w, max_h = self._display.width(), self._display.height()
        scale = min(max_w / w, max_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        frame = cv2.resize(frame, (nw, nh))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame_buf = frame.copy()
        img = QImage(self._frame_buf.data, nw, nh, 3 * nw, QImage.Format_RGB888)
        self._display.setPixmap(QPixmap.fromImage(img))
        self._cur = idx
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)

    def _next_frame(self):
        nxt = self._cur + 1
        if nxt >= self._total:
            if self._loop:
                nxt = 0
            else:
                self.pause()
                return
        self._show(nxt)

    def play(self):
        if not self._cap:
            return
        self._playing = True
        self._btn_play.setText("Pause")
        interval = max(1, int(1000 / max(0.01, self._fps * self._speed)))
        self._timer.start(interval)

    def pause(self):
        self._playing = False
        self._btn_play.setText("Play")
        self._timer.stop()

    def toggle_play(self):
        self.pause() if self._playing else self.play()

    def seek(self, idx):
        self.pause()
        self._show(idx)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space:
            self.toggle_play()
            e.accept()
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        self.pause()
        if self._cap:
            self._cap.release()
        super().closeEvent(e)


class _Card(QFrame):
    def __init__(self, title, value="-"):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame{background:#FFFFFF;border:1px solid #E5E5E5;border-radius:6px;}"
        )
        self.setFixedHeight(90)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        t = QLabel(title)
        t.setStyleSheet(
            "color:#9B9B9B;font-size:10px;font-weight:600;"
            "text-transform:uppercase;letter-spacing:1px;"
            "border:none;background:transparent;"
        )
        lay.addWidget(t)
        self._value = QLabel(value)
        self._value.setFont(QFont("Consolas", 22, QFont.Bold))
        self._value.setStyleSheet("color:#1A1A1A;border:none;background:transparent;")
        lay.addWidget(self._value)

    def set(self, value):
        self._value.setText(str(value))


class WslSetupDialog(QDialog):
    """Small GPU setup wizard for WSL2 + cuML."""

    _checks_done = pyqtSignal(list)

    _STEPS = [
        ("WSL2 installed", "Windows Subsystem for Linux"),
        ("Linux distro", "Ubuntu or another registered distro"),
        ("Python environment", "venv_wsl/bin/python"),
        ("cuML GPU library", "RAPIDS cuML with CUDA"),
    ]

    _SETUP_SCRIPT = """\
set -e
cd {wsl_root}
sudo apt-get update -q
sudo apt-get install -y -q python3 python3-venv python3-pip
python3 -m venv venv_wsl
venv_wsl/bin/pip install --upgrade pip -q
venv_wsl/bin/pip install numpy pandas scikit-learn umap-learn hdbscan joblib -q
venv_wsl/bin/pip install --extra-index-url https://pypi.nvidia.com cuml-cu12
venv_wsl/bin/python -c "import cuml; import cupy; cupy.cuda.runtime.getDeviceCount(); print('cuml_ok')"
"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GPU Acceleration Setup")
        self.resize(660, 540)
        self._active_thread = None
        self._pending_action = None
        self._checks_done.connect(self._on_checks_done)
        self._build()
        QTimer.singleShot(200, self._run_checks)

    def _build(self):
        lay = QVBoxLayout(self)
        title = QLabel("GPU Acceleration Setup")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        lay.addWidget(title)
        intro = QLabel("UMAP and HDBSCAN can use your RTX GPU through WSL2 + cuML.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#555;")
        lay.addWidget(intro)
        self._icons = []
        for name, detail in self._STEPS:
            row = QHBoxLayout()
            icon = QLabel("...")
            icon.setFixedWidth(28)
            row.addWidget(icon)
            row.addWidget(QLabel(f"<b>{name}</b> - {detail}"), stretch=1)
            lay.addLayout(row)
            self._icons.append(icon)
        self._action_lbl = QLabel("Checking...")
        self._action_lbl.setStyleSheet("color:#1a73e8;font-weight:bold;")
        lay.addWidget(self._action_lbl)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(190)
        self._log.setStyleSheet("background:#111;color:#d4d4d4;font-family:Consolas;font-size:11px;")
        lay.addWidget(self._log)
        btn_row = QHBoxLayout()
        self._action_btn = QPushButton("Checking...")
        self._action_btn.setEnabled(False)
        self._action_btn.clicked.connect(self._on_action)
        self._recheck_btn = QPushButton("Re-check")
        self._recheck_btn.setEnabled(False)
        self._recheck_btn.clicked.connect(self._run_checks)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._action_btn, stretch=1)
        btn_row.addWidget(self._recheck_btn)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

    def _run_checks(self):
        self._recheck_btn.setEnabled(False)
        self._action_btn.setEnabled(False)
        self._action_btn.setText("Checking...")
        self._action_lbl.setText("Checking your system...")
        for icon in self._icons:
            icon.setText("...")

        class _CheckThread(QThread):
            done = pyqtSignal(list)
            def run(self):
                self.done.emit([
                    _wsl_check_installed(),
                    _wsl_check_distro(),
                    _wsl_check_venv(),
                    _probe_wsl_cuml(),
                ])

        t = _CheckThread(self)
        t.done.connect(self._checks_done)
        self._active_thread = t
        t.start()

    def _on_checks_done(self, results: list):
        global _WSL_CUML
        _WSL_CUML = bool(results[-1])
        for icon, ok in zip(self._icons, results):
            icon.setText("OK" if ok else "NO")
            icon.setStyleSheet(f"color:{'#1b5e20' if ok else '#b71c1c'};font-weight:bold;")
        self._recheck_btn.setEnabled(True)
        wsl_ok, distro_ok, venv_ok, cuml_ok = results
        if cuml_ok:
            self._action_lbl.setText("GPU ready. Clustering can use WSL2 + cuML.")
            self._action_btn.setEnabled(False)
        elif not wsl_ok:
            self._action_lbl.setText("Install WSL2, then restart if Windows asks.")
            self._action_btn.setText("Install WSL2 (Admin)")
            self._action_btn.setEnabled(True)
            self._pending_action = lambda: _wsl_elevate_install("--install")
        elif not distro_ok:
            self._action_lbl.setText("Install Ubuntu in WSL2.")
            self._action_btn.setText("Install Ubuntu")
            self._action_btn.setEnabled(True)
            self._pending_action = lambda: _wsl_elevate_install("--install -d Ubuntu")
        else:
            self._action_lbl.setText("Create venv_wsl and install cuML.")
            self._action_btn.setText("Set up GPU environment")
            self._action_btn.setEnabled(True)
            self._pending_action = self._setup_env

    def _on_action(self):
        if self._pending_action:
            self._pending_action()

    def _setup_env(self):
        self._action_btn.setEnabled(False)
        self._recheck_btn.setEnabled(False)
        script = self._SETUP_SCRIPT.format(wsl_root=shlex.quote(_wsl_path(str(ROOT))))

        class _SetupThread(QThread):
            log = pyqtSignal(str)
            done = pyqtSignal(bool)
            def __init__(self, script):
                super().__init__()
                self._script = script
            def run(self):
                try:
                    proc = subprocess.Popen(
                        ["wsl", "bash", "-lc", self._script],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                    )
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        self.log.emit(line)
                    self.done.emit(proc.wait() == 0)
                except Exception as exc:
                    self.log.emit(f"[error] {exc}\n")
                    self.done.emit(False)

        t = _SetupThread(script)
        t.log.connect(lambda s: self._log.insertPlainText(s))
        t.done.connect(lambda ok: (wsl_cuml_reset_cache(), self._run_checks()) if ok else self._recheck_btn.setEnabled(True))
        self._active_thread = t
        t.start()


class DiagnoseDialog(QDialog):
    """Run compare.py --diagnose in the background with configurable options."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Diagnostic")
        self.resize(760, 580)
        self._thread = None
        lay = QVBoxLayout(self)
        hdr = QLabel("HDBSCAN min_cluster_size sweep")
        hdr.setFont(QFont("Arial", 12, QFont.Bold))
        lay.addWidget(hdr)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        self._mcs_edit = QLineEdit()
        self._mcs_edit.setPlaceholderText("50,100,200,300,500,750,1000,1500,2000,3000")
        self._mcs_edit.setToolTip("Comma-separated min_cluster_size values to sweep. Leave blank to use defaults.")
        form.addRow("MCS values (blank=defaults):", self._mcs_edit)
        self._umap_sweep = QCheckBox("Sweep UMAP n_neighbors (slower)")
        form.addRow("", self._umap_sweep)
        self._jobs = QSpinBox()
        self._jobs.setRange(1, 16)
        self._jobs.setValue(1)
        self._jobs.setToolTip("Parallel jobs for HDBSCAN core-distance computation.")
        form.addRow("HDBSCAN parallel jobs:", self._jobs)
        lay.addLayout(form)

        self._run_btn = QPushButton("Run Diagnostic")
        self._run_btn.clicked.connect(self.start)
        lay.addWidget(self._run_btn)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("background:#111;color:#d4d4d4;font-family:Consolas;font-size:12px;")
        lay.addWidget(self._log, stretch=1)
        self._status = QLabel("Configure options above and click Run Diagnostic.")
        lay.addWidget(self._status)

    def start(self):
        if self._thread and self._thread.isRunning():
            return
        args = ["compare.py", "--diagnose"]
        mcs_text = self._mcs_edit.text().strip()
        if mcs_text:
            args += ["--diagnose-mcs", mcs_text]
        if self._umap_sweep.isChecked():
            args.append("--umap-sweep")
        jobs = self._jobs.value()
        if jobs > 1:
            args += ["--hdbscan-jobs", str(jobs)]

        class _DiagThread(QThread):
            log = pyqtSignal(str)
            done = pyqtSignal(bool)
            def __init__(self, args):
                super().__init__()
                self._args = args
            def run(self):
                try:
                    proc = subprocess.Popen(
                        [sys.executable, *self._args],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
                    )
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        self.log.emit(line)
                    self.done.emit(proc.wait() == 0)
                except Exception as exc:
                    self.log.emit(f"[error] {exc}\n")
                    self.done.emit(False)

        self._log.clear()
        self._thread = _DiagThread(args)
        self._thread.log.connect(lambda s: (self._log.insertPlainText(s), self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())))
        self._thread.done.connect(lambda ok: (
            self._status.setText("Done." if ok else "Failed."),
            self._run_btn.setEnabled(True),
        ))
        self._run_btn.setEnabled(False)
        self._status.setText("Running...")
        self._thread.start()


class OverviewView(QWidget):
    export_requested = pyqtSignal()
    load_previous_requested = pyqtSignal()
    cohort_path_changed = pyqtSignal(str)
    navigate_help = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._cohort_df = None
        self._data = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(16)

        self._prev_banner = QFrame()
        self._prev_banner.setStyleSheet("QFrame{background:#fff3cd;border:1px solid #ffc107;border-radius:8px;}")
        pb = QHBoxLayout(self._prev_banner)
        self._prev_lbl = QLabel("Previous analysis results available.")
        pb.addWidget(self._prev_lbl, stretch=1)
        self._prev_load_btn = QPushButton("Load Previous Session")
        self._prev_load_btn.clicked.connect(self.load_previous_requested.emit)
        pb.addWidget(self._prev_load_btn)
        self._prev_banner.hide()
        lay.addWidget(self._prev_banner)

        top = QHBoxLayout()
        title = QLabel("Overview")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        top.addWidget(title)
        _ov_hbtn = QPushButton("?")
        _ov_hbtn.setFixedSize(20, 20)
        _ov_hbtn.setFlat(True)
        _ov_hbtn.setToolTip("Open Help: What is VIEB?")
        _ov_hbtn.setCursor(Qt.PointingHandCursor)
        _ov_hbtn.setStyleSheet(
            "QPushButton{border:1px solid #aaa;border-radius:10px;color:#555;"
            "background:#f5f5f5;font-size:10px;font-weight:bold;}"
            "QPushButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
        )
        _ov_hbtn.clicked.connect(lambda: self.navigate_help.emit("what_is_vieb"))
        top.addWidget(_ov_hbtn)
        top.addStretch()
        self._export_btn = QPushButton("Export Results")
        self._export_btn.clicked.connect(self.export_requested.emit)
        top.addWidget(self._export_btn)
        lay.addLayout(top)

        row = QHBoxLayout()
        self._c_videos = _Card("Total Videos")
        self._c_frames = _Card("Total Frames")
        self._c_states = _Card("States")
        self._c_noise = _Card("Noise %")
        for c in (self._c_videos, self._c_frames, self._c_states, self._c_noise):
            row.addWidget(c)
        lay.addLayout(row)

        cohort_row = QHBoxLayout()
        self._cohort_btn = QPushButton("Upload Cohort File")
        self._cohort_btn.clicked.connect(self._upload_cohort_csv)
        cohort_row.addWidget(self._cohort_btn)
        self._cohort_status = QLabel("No cohort file loaded")
        self._cohort_status.setStyleSheet("color:#777;")
        cohort_row.addWidget(self._cohort_status, stretch=1)
        lay.addLayout(cohort_row)

        box = QGroupBox("Mean State Occupancy")
        bl = QVBoxLayout(box)
        chk_row = QHBoxLayout()
        chk_row.addStretch()
        self._hide_leading = QCheckBox("Hide leading state and rescale")
        self._hide_leading.toggled.connect(self._render_state_occupancy)
        chk_row.addWidget(self._hide_leading)
        bl.addLayout(chk_row)
        if _init_mpl():
            self._canvas = MplCanvas(figsize=(8, 4))
            bl.addWidget(self._canvas)
        else:
            self._canvas = None
            bl.addWidget(QLabel("Install matplotlib to view charts."))
        lay.addWidget(box)

        self._run_lbl = QLabel("Last run: -")
        self._run_lbl.setStyleSheet("color:#777;")
        lay.addWidget(self._run_lbl)
        lay.addStretch()

    def update_data(self, data):
        self._data = data
        cohort = data.get("cohort")
        if cohort is not None:
            self._cohort_df = cohort
            self._cohort_status.setText(self._cohort_status_text())
        summary = data.get("summary")
        ci = data.get("cluster_info")
        fi = data.get("feature_index")
        if summary is None:
            self._c_videos.set("-")
            return
        self._c_videos.set(len(summary))
        self._c_states.set(ci.get("n_clusters", 0) if ci else "-")

        total = 0
        if isinstance(fi, dict):
            for v in fi.values():
                if isinstance(v, dict):
                    total += int(v.get("n_frames", 0))
        self._c_frames.set(f"{total:,}" if total else "-")

        state_cols = [c for c in summary.columns if c.startswith("state_") and c.endswith("_frac")]
        if state_cols:
            mean_sum = summary[state_cols].sum(axis=1).mean()
            noise = (1 - float(mean_sum)) * 100
            self._c_noise.set(f"{noise:.1f}%")
            self._render_state_occupancy()
        p = RESULTS / "comparison" / "summary_table.csv"
        if p.exists():
            self._run_lbl.setText(f"Last run: {_fmt_ts(p.stat().st_mtime)}")

    def _render_state_occupancy(self):
        if not self._canvas:
            return
        summary = self._data.get("summary")
        ci = self._data.get("cluster_info")
        self._canvas.ax.clear()
        if summary is None or ci is None:
            self._canvas.ax.text(0.5, 0.5, "No summary data", ha="center", va="center")
            self._canvas.draw()
            return
        n = int(ci.get("n_clusters", 0))
        state_ids, vals, lead = _state_means(summary, n, self._hide_leading.isChecked())
        colors = mpl_cm.tab20(np.linspace(0, 1, max(1, len(state_ids))))
        self._canvas.ax.bar(state_ids, vals, color=colors)
        self._canvas.ax.set_xlabel("State ID")
        self._canvas.ax.set_ylabel("Visible fraction" if self._hide_leading.isChecked() else "Fraction")
        if self._hide_leading.isChecked() and lead is not None:
            self._canvas.ax.set_title(f"Mean State Occupancy (state {lead} hidden)")
        else:
            self._canvas.ax.set_title("Mean State Occupancy")
        self._canvas.fig.tight_layout()
        self._canvas.draw()

    def show_load_banner(self, has_results: bool):
        p = RESULTS / "comparison" / "summary_table.csv"
        if has_results and p.exists():
            self._prev_lbl.setText(f"Results from {_fmt_ts(p.stat().st_mtime)} available on disk.")
            self._prev_banner.show()
        else:
            self._prev_banner.hide()

    def _upload_cohort_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Cohort File", "",
            "Cohort files (*.csv *.xlsx *.xls);;CSV files (*.csv);;Excel files (*.xlsx *.xls)"
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".xlsx", ".xls", ".xlsm"):
                from cohort_loader import load_cohort_excel
                df = load_cohort_excel(path)
                norm_path = str(ROOT / "cohort_normalized.csv")
                df.to_csv(norm_path, index=False)
                emit_path = norm_path
            else:
                df = pd.read_csv(path)
                emit_path = path
        except Exception as exc:
            QMessageBox.warning(self, "Load Error", f"Could not load cohort file:\n{exc}")
            return
        if "animal_id" not in df.columns:
            QMessageBox.warning(self, "Invalid File", "Cohort file must include an animal_id column.")
            return
        self._cohort_df = df
        self._cohort_status.setText(self._cohort_status_text())
        self.cohort_path_changed.emit(emit_path)

    def _cohort_status_text(self):
        if self._cohort_df is None:
            return "No cohort file loaded"
        df = self._cohort_df
        bits = [f"{len(df)} animals"]
        for col in ("genotype", "treatment", "sex"):
            if col in df.columns:
                bits.append(f"{df[col].nunique()} {col}")
        return " | ".join(bits)


class _TerminalWidget(QTextEdit):
    """Read-only dark terminal with Copy/Clear buttons overlaid at the bottom-right."""

    _BTN_STYLE = (
        "QPushButton{background:rgba(45,45,45,210);color:#aaa;border:1px solid #555;"
        "border-radius:3px;font-size:10px;padding:1px 7px;}"
        "QPushButton:hover{background:rgba(80,80,80,230);color:#eee;}"
    )

    def __init__(self, extra_clear=None, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self._extra_clear = extra_clear

        self._overlay = QWidget(self)
        self._overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        lay = QHBoxLayout(self._overlay)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        copy_btn = QPushButton("Copy")
        copy_btn.setFixedHeight(20)
        copy_btn.setStyleSheet(self._BTN_STYLE)
        copy_btn.setCursor(Qt.ArrowCursor)
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.toPlainText()))
        lay.addWidget(copy_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(20)
        clear_btn.setStyleSheet(self._BTN_STYLE)
        clear_btn.setCursor(Qt.ArrowCursor)
        clear_btn.clicked.connect(self._do_clear)
        lay.addWidget(clear_btn)

        self._overlay.adjustSize()
        self._overlay.raise_()

    def _do_clear(self):
        self.clear()
        if self._extra_clear:
            self._extra_clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._overlay.adjustSize()
        self._overlay.move(
            self.width() - self._overlay.width() - 2,
            self.height() - self._overlay.height() - 2,
        )


_STAGE_HELP_MAP: dict[int, str] = {
    1:  "stage_1_dlc",
    2:  "stage_2_features",
    3:  "stage_3_clustering",
    7:  "stage_4_collapse",
    8:  "stage_5_comparison",
    9:  "stage_7_quantification",
    10: "stage_5_comparison",
    11: "stage_6_clips",
}


class StageRow(QFrame):
    run_stage = pyqtSignal(int)
    run_from_here = pyqtSignal(int)
    mark_completed = pyqtSignal(int, bool)
    changed = pyqtSignal(str, object)
    run_diagnose = pyqtSignal()
    run_subcluster = pyqtSignal(int)
    navigate_help = pyqtSignal(str)

    def __init__(self, stage: dict, cfg: dict):
        super().__init__()
        self.stage = stage
        self.cfg = cfg
        self.logs = deque(maxlen=20)
        self._build()

    def _build(self):
        self.setStyleSheet("QFrame{border:none;background:#fff;}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)

        top = QHBoxLayout()
        self._icon = QLabel("○")
        self._icon.setFixedWidth(18)
        top.addWidget(self._icon)
        name = QLabel(f"Stage {self.stage['id']}: {self.stage['name']}")
        name.setFont(QFont("Arial", 11, QFont.Bold))
        top.addWidget(name)
        top.addStretch()
        _help_anchor = _STAGE_HELP_MAP.get(self.stage["id"])
        if _help_anchor:
            _hb = QToolButton()
            _hb.setText("?")
            _hb.setFixedSize(20, 20)
            _hb.setToolTip("Open Help for this stage")
            _hb.setCursor(Qt.PointingHandCursor)
            _hb.setStyleSheet(
                "QToolButton{border:1px solid #aaa;border-radius:10px;color:#555;"
                "background:#f5f5f5;font-size:10px;font-weight:bold;}"
                "QToolButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
            )
            _hb.clicked.connect(lambda _, a=_help_anchor: self.navigate_help.emit(a))
            top.addWidget(_hb)
        self._ts = QLabel("Last run: -")
        self._eta = QLabel("ETA: -")
        top.addWidget(self._ts)
        top.addWidget(QLabel("  "))
        top.addWidget(self._eta)
        self._expand = QToolButton()
        self._expand.setText("▾")
        self._expand.setCheckable(True)
        self._expand.clicked.connect(self._toggle)
        top.addWidget(self._expand)
        self._done_cb = QCheckBox()
        self._done_cb.setToolTip("Mark stage as completed")
        self._done_cb.toggled.connect(lambda v: self.mark_completed.emit(self.stage["id"], v))
        top.addWidget(self._done_cb)
        lay.addLayout(top)

        d = QLabel(self.stage["desc"])
        d.setStyleSheet("color:#555;")
        d.setWordWrap(True)
        lay.addWidget(d)
        self._quality_lbl = None
        self._dom_state_id = -1
        if self.stage["id"] == 3:
            self._quality_lbl = QLabel("")
            self._quality_lbl.setStyleSheet("color:#666;")
            lay.addWidget(self._quality_lbl)

        self._details = QWidget()
        dl = QVBoxLayout(self._details)
        cmd = QLabel(f"CLI: {self.stage['cmd']}")
        cmd.setStyleSheet("font-family:Consolas;color:#222;")
        dl.addWidget(cmd)

        params = QHBoxLayout()
        if self.stage["id"] == 2:
            self._wave = QCheckBox("Use Morlet wavelets")
            self._wave.setChecked(bool(self.cfg.get("use_wavelets", True)))
            self._wave.toggled.connect(lambda v: self.changed.emit("use_wavelets", v))
            params.addWidget(self._wave)
            params.addWidget(QLabel("  FPS"))
            self._fps_spin = QDoubleSpinBox()
            self._fps_spin.setRange(1.0, 120.0)
            self._fps_spin.setSingleStep(1.0)
            self._fps_spin.setDecimals(1)
            self._fps_spin.setValue(float(self.cfg.get("fps", 30.0)))
            self._fps_spin.valueChanged.connect(lambda v: self.changed.emit("fps", v))
            params.addWidget(self._fps_spin)
        if self.stage["id"] == 3:
            params.addWidget(QLabel("min_cluster_size"))
            self._mcs = QSpinBox()
            self._mcs.setRange(10, 10000)
            self._mcs.setValue(int(self.cfg.get("min_cluster_size", 50)))
            self._mcs.valueChanged.connect(lambda v: self.changed.emit("min_cluster_size", v))
            params.addWidget(self._mcs)
            params.addWidget(QLabel("  min_samples (0=auto)"))
            self._hms = QSpinBox()
            self._hms.setRange(0, 5000)
            self._hms.setValue(int(self.cfg.get("hdbscan_min_samples", 0)))
            self._hms.valueChanged.connect(lambda v: self.changed.emit("hdbscan_min_samples", v))
            params.addWidget(self._hms)
            params.addWidget(QLabel("  UMAP dims"))
            self._udims = QSpinBox()
            self._udims.setRange(2, 50)
            self._udims.setValue(int(self.cfg.get("umap_dims", 10)))
            self._udims.valueChanged.connect(lambda v: self.changed.emit("umap_dims", v))
            params.addWidget(self._udims)
            self._validate_cb = QCheckBox("80/20 validation split")
            self._validate_cb.setChecked(bool(self.cfg.get("validate", False)))
            self._validate_cb.toggled.connect(lambda v: self.changed.emit("validate", v))
            params.addWidget(self._validate_cb)
        if self.stage["id"] == 7:
            self._collapse = QCheckBox("Enable state collapsing")
            self._collapse.setChecked(bool(self.cfg.get("enable_state_collapse", False)))
            self._collapse.toggled.connect(lambda v: self.changed.emit("enable_state_collapse", v))
            params.addWidget(self._collapse)
            params.addWidget(QLabel("  threshold"))
            self._ct = QDoubleSpinBox()
            self._ct.setRange(0.0, 1.0)
            self._ct.setSingleStep(0.05)
            self._ct.setDecimals(2)
            self._ct.setValue(float(self.cfg.get("collapse_threshold", 0.5)))
            self._ct.valueChanged.connect(lambda v: self.changed.emit("collapse_threshold", v))
            params.addWidget(self._ct)
        if self.stage["id"] == 8:
            params.addWidget(QLabel("min confidence"))
            self._mconf = QDoubleSpinBox()
            self._mconf.setRange(0.0, 1.0)
            self._mconf.setSingleStep(0.05)
            self._mconf.setDecimals(2)
            self._mconf.setToolTip(
                "Exclude frames whose HDBSCAN soft probability is below this threshold.\n"
                "Literature default: 0.7 (Luxem et al. 2022; Gordon et al. 2023)."
            )
            self._mconf.setValue(float(self.cfg.get("min_confidence", 0.7)))
            self._mconf.valueChanged.connect(lambda v: self.changed.emit("min_confidence", v))
            params.addWidget(self._mconf)
        if self.stage["id"] == 11:
            self._clips = QCheckBox("Export video clips")
            self._clips.setChecked(bool(self.cfg.get("export_clips", False)))
            self._clips.toggled.connect(lambda v: self.changed.emit("export_clips", v))
            params.addWidget(self._clips)
        params.addStretch()
        dl.addLayout(params)

        self._log = _TerminalWidget(extra_clear=self._clear_log)
        self._log.setMinimumHeight(120)
        self._log.setStyleSheet("background:#181818;color:#d4d4d4;font-family:Consolas;font-size:11px;")
        dl.addWidget(self._log)

        acts = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(lambda: self.run_stage.emit(self.stage["id"]))
        self._from_btn = QPushButton("Run from here")
        self._from_btn.clicked.connect(lambda: self.run_from_here.emit(self.stage["id"]))
        acts.addWidget(self._run_btn)
        acts.addWidget(self._from_btn)
        if self.stage["id"] == 3:
            diag_btn = QPushButton("Diagnose")
            diag_btn.clicked.connect(self.run_diagnose.emit)
            _diag_hb = QToolButton()
            _diag_hb.setText("?")
            _diag_hb.setFixedSize(20, 20)
            _diag_hb.setToolTip("Open Help: Diagnose Clustering Parameters")
            _diag_hb.setCursor(Qt.PointingHandCursor)
            _diag_hb.setStyleSheet(
                "QToolButton{border:1px solid #aaa;border-radius:10px;color:#555;"
                "background:#f5f5f5;font-size:10px;font-weight:bold;}"
                "QToolButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
            )
            _diag_hb.clicked.connect(lambda: self.navigate_help.emit("diagnose"))
            split_btn = QPushButton("Split Dominant State")
            split_btn.clicked.connect(lambda: self.run_subcluster.emit(self._dom_state_id))
            _split_hb = QToolButton()
            _split_hb.setText("?")
            _split_hb.setFixedSize(20, 20)
            _split_hb.setToolTip("Open Help: Split Dominant State")
            _split_hb.setCursor(Qt.PointingHandCursor)
            _split_hb.setStyleSheet(
                "QToolButton{border:1px solid #aaa;border-radius:10px;color:#555;"
                "background:#f5f5f5;font-size:10px;font-weight:bold;}"
                "QToolButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
            )
            _split_hb.clicked.connect(lambda: self.navigate_help.emit("split_dominant"))
            acts.addWidget(diag_btn)
            acts.addWidget(_diag_hb)
            acts.addWidget(split_btn)
            acts.addWidget(_split_hb)
        acts.addStretch()
        dl.addLayout(acts)

        self._details.hide()
        lay.addWidget(self._details)

    def _toggle(self):
        self._details.setVisible(self._expand.isChecked())
        self._expand.setText("▴" if self._expand.isChecked() else "▾")

    def set_eta(self, text):
        self._eta.setText(f"ETA: {text}")

    def set_status(self, status):
        icon_map = {
            "pending": ("○", "#888"),
            "running": ("◔", "#e0a400"),
            "done": ("✓", "green"),
            "error": ("✕", "red"),
        }
        icon, color = icon_map.get(status, ("○", "#888"))
        self._icon.setText(icon)
        self._icon.setStyleSheet(f"color:{color};font-weight:bold;")
        self._done_cb.blockSignals(True)
        self._done_cb.setChecked(status == "done")
        self._done_cb.blockSignals(False)

    def set_last_run(self, ts):
        self._ts.setText(f"Last run: {_fmt_ts(ts)}")

    def set_cluster_quality(self, dom_frac: float, dom_state_id: int):
        self._dom_state_id = int(dom_state_id)
        if self._quality_lbl is None:
            return
        self._quality_lbl.setText(
            f"Dominant state: {dom_state_id} ({dom_frac * 100:.1f}% mean occupancy)"
        )
        color = "#b71c1c" if dom_frac >= 0.5 else "#555"
        self._quality_lbl.setStyleSheet(f"color:{color};")

    def append_log(self, line):
        self.logs.append(line.rstrip("\n"))
        self._log.setPlainText("\n".join(self.logs))
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _clear_log(self):
        self.logs.clear()
        self._log.clear()

    def set_enabled(self, enabled):
        self._run_btn.setEnabled(enabled)
        self._from_btn.setEnabled(enabled)


class RunPipelineView(QWidget):
    pipeline_done = pyqtSignal()
    worker_running = pyqtSignal(bool)
    navigate_dlc = pyqtSignal()
    cluster_finished = pyqtSignal()
    navigate_help = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._rows: dict[int, StageRow] = {}
        self._worker = None
        self._active_stages = set()
        self._wsl_thread = None
        self._build()
        QTimer.singleShot(800, self._probe_gpu_async)

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)

        top = QHBoxLayout()
        t = QLabel("Pipeline")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        top.addWidget(t)
        top.addStretch()
        self._expand_all_btn = QPushButton("Expand All")
        self._expand_all_btn.setFixedHeight(28)
        self._expand_all_btn.clicked.connect(lambda: self._set_all_expanded(True))
        top.addWidget(self._expand_all_btn)
        self._collapse_all_btn = QPushButton("Collapse All")
        self._collapse_all_btn.setFixedHeight(28)
        self._collapse_all_btn.clicked.connect(lambda: self._set_all_expanded(False))
        top.addWidget(self._collapse_all_btn)
        self._dlc_btn = QPushButton("DLC Setup")
        self._dlc_btn.clicked.connect(self.navigate_dlc.emit)
        top.addWidget(self._dlc_btn)
        self._run_full = QPushButton("Run Full Pipeline")
        self._run_full.clicked.connect(self.run_full_pipeline)
        top.addWidget(self._run_full)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setStyleSheet("background:#c62828;color:white;font-weight:bold;")
        self._stop_btn.setFixedHeight(28)
        self._stop_btn.clicked.connect(self._stop_pipeline)
        self._stop_btn.setVisible(False)
        top.addWidget(self._stop_btn)
        lay.addLayout(top)

        # GPU badge — shown in header while Stage 0 row is hidden
        gpu_row = QHBoxLayout()
        self._gpu_badge = QLabel("Checking GPU...")
        self._gpu_badge.setStyleSheet("background:#f5f5f5;border:1px solid #ddd;border-radius:4px;padding:4px 10px;color:#555;")
        gpu_row.addWidget(self._gpu_badge)
        gpu_row.addStretch()
        self._gpu_setup_btn = QPushButton("Set up GPU acceleration")
        self._gpu_setup_btn.setFixedHeight(26)
        self._gpu_setup_btn.clicked.connect(self._open_wsl_setup)
        gpu_row.addWidget(self._gpu_setup_btn)
        lay.addLayout(gpu_row)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#666;")
        lay.addWidget(self._status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        v = QVBoxLayout(holder)
        for stage in STAGES:
            if stage["id"] == 0:  # hidden — re-enable when Environment Setup is ready
                continue
            row = StageRow(stage, self.cfg)
            if stage["id"] == 0:
                row.run_stage.connect(lambda _: self._open_wsl_setup())
                row._run_btn.setText("Open Setup ▶")
                row._from_btn.hide()
                if self._venv_exists():
                    row.set_status("done")
            elif stage["id"] == 1:
                row.run_stage.connect(lambda _: self.navigate_dlc.emit())
                row._run_btn.setText("Open DLC Setup")
                row._from_btn.hide()
            else:
                row.run_stage.connect(self._run_stage)
                row.run_from_here.connect(self._run_from_here)
            row.mark_completed.connect(self._mark_completed)
            row.changed.connect(self._param_changed)
            row.navigate_help.connect(self.navigate_help.emit)
            if stage["id"] == 3:
                row.run_diagnose.connect(self._run_diagnose)
                row.run_subcluster.connect(self._run_subcluster)
            self._rows[stage["id"]] = row
            v.addWidget(row)
        v.addStretch()
        scroll.setWidget(holder)
        lay.addWidget(scroll)

        log_label = QLabel("Pipeline Output")
        log_label.setStyleSheet("font-weight:bold;color:#444;")
        lay.addWidget(log_label)
        self._global_log = _TerminalWidget()
        self._global_log.setMinimumHeight(120)
        self._global_log.setMaximumHeight(300)
        self._global_log.setStyleSheet("background:#151515;color:#cfd8dc;font-family:Consolas;font-size:11px;")
        lay.addWidget(self._global_log)

    def _probe_gpu_async(self):
        class _ProbeThread(QThread):
            result = pyqtSignal(object)
            def run(self):
                global _linux_gpu_name
                if sys.platform == "win32":
                    self.result.emit(_probe_wsl_cuml())
                else:
                    _linux_gpu_name = _probe_nvidia_smi()
                    self.result.emit(_probe_linux_cuml())

        if self._wsl_thread and self._wsl_thread.isRunning():
            return
        self._wsl_thread = _ProbeThread(self)
        self._wsl_thread.result.connect(self._on_gpu_probe)
        self._wsl_thread.start()

    def _on_gpu_probe(self, ok):
        global _WSL_CUML
        _WSL_CUML = ok
        self.refresh_gpu_badge()

    def refresh_gpu_badge(self):
        on_windows = sys.platform == "win32"
        if _WSL_CUML is True:
            badge_text = "GPU ready: WSL2 + cuML" if on_windows else f"GPU ready: cuML ({_linux_gpu_name or 'NVIDIA'})"
            self._gpu_badge.setText(badge_text)
            self._gpu_badge.setStyleSheet(
                "background:#e8f5e9;border:1px solid #a5d6a7;"
                "border-radius:4px;padding:4px 10px;color:#1b5e20;"
            )
            self._gpu_setup_btn.setVisible(on_windows)
            if on_windows:
                self._gpu_setup_btn.setText("GPU Setup")
        elif _WSL_CUML is False:
            if on_windows:
                badge_text = "CPU mode: WSL2 + cuML not found"
                self._gpu_badge.setText(badge_text)
                self._gpu_badge.setStyleSheet(
                    "background:#fff8e1;border:1px solid #ffe082;"
                    "border-radius:4px;padding:4px 10px;color:#795548;"
                )
                self._gpu_setup_btn.setVisible(True)
                self._gpu_setup_btn.setText("Set up GPU acceleration")
            elif _linux_gpu_name:
                if _WSL_CUML == "installed":
                    # cuML is installed but CUDA driver can't initialize (e.g. nvidia-uvm I/O error)
                    self._gpu_badge.setText(f"GPU detected ({_linux_gpu_name}) — cuML installed, CUDA init failed (try rebooting)")
                    self._gpu_badge.setStyleSheet(
                        "background:#fff8e1;border:1px solid #ffe082;"
                        "border-radius:4px;padding:4px 10px;color:#795548;"
                    )
                    self._gpu_setup_btn.setVisible(False)
                else:
                    # GPU hardware found but cuML not installed
                    self._gpu_badge.setText(f"GPU detected ({_linux_gpu_name}) — cuML not installed")
                    self._gpu_badge.setStyleSheet(
                        "background:#fff3e0;border:1px solid #ffb74d;"
                        "border-radius:4px;padding:4px 10px;color:#e65100;"
                    )
                    self._gpu_setup_btn.setVisible(True)
                    self._gpu_setup_btn.setText("Install cuML (GPU acceleration)")
            else:
                self._gpu_badge.setText("No NVIDIA GPU detected — running on CPU")
                self._gpu_badge.setStyleSheet(
                    "background:#f5f5f5;border:1px solid #ddd;"
                    "border-radius:4px;padding:4px 10px;color:#888;"
                )
                self._gpu_setup_btn.setVisible(False)
        else:
            self._gpu_badge.setText("Checking GPU...")

    @staticmethod
    def _venv_exists() -> bool:
        venv = ROOT / "venv"
        if sys.platform == "win32":
            return (venv / "Scripts" / "python.exe").exists()
        return (venv / "bin" / "python").exists()

    def _open_wsl_setup(self):
        if sys.platform == "win32":
            dlg = WslSetupDialog(self)
            dlg.exec_()
            wsl_cuml_reset_cache()
            self._probe_gpu_async()
        else:
            from _dialogs import LinuxGpuSetupDialog
            dlg = LinuxGpuSetupDialog(_linux_gpu_name, self)
            dlg.exec_()
            self._probe_gpu_async()

    def _param_changed(self, key, value):
        self.cfg[key] = value

    def _mark_completed(self, sid, completed):
        key = _state_key(sid)
        self.cfg.setdefault("stage_status", {})[key] = "done" if completed else "pending"
        if completed:
            self.cfg.setdefault("stage_last_run", {})[key] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cfg["last_completed_stage"] = _STAGE_BY_ID[sid]["name"]
            self._rows[sid].set_status("done")
        else:
            self._rows[sid].set_status("pending")
        self._rows[sid].set_last_run(self.cfg.get("stage_last_run", {}).get(key))
        _save_cfg(self.cfg)

    def estimate_times(self, data):
        idx = data.get("feature_index") or {}
        ci = data.get("cluster_info") or {}
        n_videos = len(idx) if isinstance(idx, dict) else 0
        n_frames = sum(int(v.get("n_frames", 0)) for v in idx.values()) if isinstance(idx, dict) else 0
        for sid, row in self._rows.items():
            if sid == 1:
                mins = max(5, int(n_videos * 0.4))
            elif sid == 2:
                mins = max(2, int(n_frames / 120000))
            elif sid == 3:
                mins = max(5, int(n_frames / 90000))
            elif sid == 7:
                mins = 2
            elif sid == 8:
                mins = 2
            elif sid == 9:
                mins = 1
            elif sid == 10:
                mins = 1
            else:
                mins = 3 + (5 if self.cfg.get("export_clips") else 0)
            row.set_eta(f"~{mins} min")

    def update_from_cfg(self):
        ss = self.cfg.get("stage_status", {})
        ts = self.cfg.get("stage_last_run", {})
        for sid, row in self._rows.items():
            row.set_status(ss.get(_state_key(sid), "pending"))
            row.set_last_run(ts.get(_state_key(sid)))

    def _append_log(self, line):
        self._global_log.insertPlainText(line)
        sb = self._global_log.verticalScrollBar()
        sb.setValue(sb.maximum())
        for sid in self._active_stages:
            self._rows[sid].append_log(line)

    def _set_all_expanded(self, expand: bool):
        for row in self._rows.values():
            row._expand.setChecked(expand)
            row._toggle()

    def _stop_pipeline(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._status.setText("Stopping…")
            self._stop_btn.setEnabled(False)

    def _set_buttons(self, enabled):
        self._run_full.setEnabled(enabled)
        self._stop_btn.setVisible(not enabled)
        self._stop_btn.setEnabled(not enabled)
        for row in self._rows.values():
            row.set_enabled(enabled)

    def _start_worker(self, stage_ids):
        if self._worker and self._worker.isRunning():
            return
        self._worker = PipelineRunner(stage_ids, self.cfg)
        self._active_stages = set(stage_ids)
        self._worker.log.connect(self._append_log)
        self._worker.stage_started.connect(self._on_stage_started)
        self._worker.stage_done.connect(self._on_stage_done)
        self._worker.all_done.connect(self._on_all_done)
        self._set_buttons(False)
        self.worker_running.emit(True)
        self._worker.start()

    def _on_stage_started(self, sid):
        self._rows[sid].set_status("running")
        self._status.setText(f"Running stage {sid}: {STAGES[sid - 1]['name']}")

    def _on_stage_done(self, sid, ok):
        key = _state_key(sid)
        self.cfg.setdefault("stage_status", {})[key] = "done" if ok else "error"
        if ok:
            self.cfg.setdefault("stage_last_run", {})[key] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cfg["last_completed_stage"] = _STAGE_BY_ID[sid]["name"]
        self._rows[sid].set_status("done" if ok else "error")
        self._rows[sid].set_last_run(self.cfg["stage_last_run"].get(key))
        _save_cfg(self.cfg)
        if sid == 3 and ok:
            self.cluster_finished.emit()

    def _on_all_done(self, ok):
        stopped = self._worker is not None and getattr(self._worker, "_stop_flag", False)
        self._set_buttons(True)
        self.worker_running.emit(False)
        if stopped:
            self._status.setText("Pipeline stopped.")
        else:
            self._status.setText("Pipeline completed." if ok else "Pipeline failed.")
        if ok:
            self.pipeline_done.emit()

    def update_cluster_quality(self, data: dict):
        row = self._rows.get(3)
        ci = data.get("cluster_info")
        summary = data.get("summary")
        if row is None or ci is None or summary is None:
            return
        state_cols = [c for c in summary.columns if c.startswith("state_") and c.endswith("_frac")]
        if not state_cols:
            return
        means = summary[state_cols].mean()
        dom_col = means.idxmax()
        row.set_cluster_quality(float(means[dom_col]), int(dom_col.split("_")[1]))

    def _run_diagnose(self):
        dlg = DiagnoseDialog(self)
        dlg.show()

    def _run_subcluster(self, dom_state_id: int):
        if dom_state_id < 0:
            QMessageBox.information(self, "Subcluster", "Run Report Generation first so the dominant state can be identified.")
            return
        reply = QMessageBox.question(
            self, "Split Dominant State",
            f"Run compare.py --subcluster --state {dom_state_id}?\n\n"
            "This rewrites label files and should be followed by Report Generation.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._start_worker_cmd(
            [sys.executable, "compare.py", "--subcluster", "--state", str(dom_state_id)],
            f"Splitting state {dom_state_id}...",
        )

    def _start_worker_cmd(self, cmd: list, label: str):
        class _CmdThread(QThread):
            log = pyqtSignal(str)
            done = pyqtSignal(bool)
            def __init__(self, cmd):
                super().__init__()
                self._cmd = cmd
            def run(self):
                try:
                    proc = subprocess.Popen(
                        self._cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace", cwd=str(ROOT),
                    )
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        self.log.emit(line)
                    self.done.emit(proc.wait() == 0)
                except Exception as exc:
                    self.log.emit(f"[error] {exc}\n")
                    self.done.emit(False)

        self._status.setText(label)
        self._set_buttons(False)
        self.worker_running.emit(True)
        self._cmd_thread = _CmdThread(cmd)
        self._cmd_thread.log.connect(self._append_log)
        self._cmd_thread.done.connect(lambda ok: (
            self._set_buttons(True),
            self.worker_running.emit(False),
            self._status.setText("Done." if ok else "Failed."),
            self.pipeline_done.emit() if ok else None,
        ))
        self._cmd_thread.start()

    def _build_sequence(self, start_sid=1, from_here=False):
        all_ids = [s["id"] for s in STAGES if s["id"] >= start_sid] if from_here else [start_sid]

        if not self.cfg.get("enable_state_collapse", False):
            all_ids = [i for i in all_ids if i != 7]
        if not self.cfg.get("export_clips", False) and from_here:
            pass
        if _has_pose_csvs(Path(self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")))):
            if 1 in all_ids:
                all_ids.remove(1)
                self._rows[1].set_status("done")
        if from_here:
            done_ids = {
                int(k)
                for k, v in self.cfg.get("stage_status", {}).items()
                if str(v) == "done" and str(k).isdigit()
            }
            all_ids = [sid for sid in all_ids if sid not in done_ids]
        return all_ids

    def run_full_pipeline(self):
        self._start_worker(self._build_sequence(1, from_here=True))

    def _run_stage(self, sid):
        self._start_worker(self._build_sequence(sid, from_here=False))

    def _run_from_here(self, sid):
        self._start_worker(self._build_sequence(sid, from_here=True))


_LABELING_GUIDE = """
<h3>Labeling Guide - DeepLabCut + Napari</h3>
<p><b>What you are doing:</b> label the 8 mouse body keypoints in extracted frames.</p>
<ol>
  <li>left_ear</li><li>right_ear</li><li>nose</li><li>center</li>
  <li>left_hip</li><li>right_hip</li><li>tail_base</li><li>tail_tip</li>
</ol>
<p><b>Napari controls:</b> click to place points, use Left/Right to change frames,
Ctrl+Z to undo, and Ctrl+S to save before closing.</p>
"""


class DLCSetupView(QWidget):
    """Compact DeepLabCut setup/actions page."""

    navigate_pipeline = pyqtSignal()

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._worker = None
        self._build()
        self._refresh_pretrained()
        QTimer.singleShot(0, self._detect_and_show_status)

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        title = QLabel("DLC Setup")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        lay.addWidget(title)

        self._done_banner = QFrame()
        self._done_banner.setStyleSheet("QFrame{background:#e8f5e9;border:1px solid #a5d6a7;border-radius:6px;}")
        db = QHBoxLayout(self._done_banner)
        self._done_lbl = QLabel("")
        self._done_lbl.setWordWrap(True)
        db.addWidget(self._done_lbl, stretch=1)
        self._done_banner.hide()
        lay.addWidget(self._done_banner)

        project = QGroupBox("DLC Project")
        pl = QVBoxLayout(project)
        path_row = QHBoxLayout()
        self._path_le = QLineEdit()
        self._path_le.setPlaceholderText("DLC project directory containing config.yaml")
        self._path_le.textChanged.connect(self._refresh_project_status)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_project)
        path_row.addWidget(self._path_le, stretch=1)
        path_row.addWidget(browse)
        pl.addLayout(path_row)
        self._project_status = QLabel("")
        self._project_status.setWordWrap(True)
        pl.addWidget(self._project_status)
        lay.addWidget(project)

        pre = QGroupBox("Pretrained Model")
        pr = QHBoxLayout(pre)
        self._pretrained_combo = QComboBox()
        pr.addWidget(self._pretrained_combo, stretch=1)
        use_pre = QPushButton("Use Pretrained Model")
        use_pre.clicked.connect(self._use_pretrained)
        pr.addWidget(use_pre)
        lay.addWidget(pre)

        actions = QGroupBox("DLC Actions")
        al = QVBoxLayout(actions)
        btn_grid = QGridLayout()
        btn_grid.setSpacing(6)
        for i, (label, slot) in enumerate([
            ("Extract Frames", lambda: self._run_dlc_subprocess(["-c", "import setup_dlc_training as s; s.extract_frames()"])),
            ("Open Labeling GUI", lambda: self._run_dlc_subprocess(["setup_dlc_training.py", "--label"])),
            ("Train", lambda: self._run_dlc_subprocess(["setup_dlc_training.py", "--train"])),
            ("Evaluate", lambda: self._run_dlc_subprocess(["setup_dlc_training.py", "--evaluate"])),
            ("Run Pose Estimation", lambda: self._run_dlc_subprocess(["setup_dlc_training.py", "--analyze"])),
        ]):
            b = QPushButton(label)
            b.setMinimumHeight(32)
            b.clicked.connect(slot)
            btn_grid.addWidget(b, i // 3, i % 3)
        al.addLayout(btn_grid)
        self._log = _TerminalWidget()
        self._log.setMaximumHeight(220)
        self._log.setMinimumHeight(80)
        self._log.setStyleSheet("background:#151515;color:#cfd8dc;font-family:Consolas;font-size:11px;")
        al.addWidget(self._log)
        lay.addWidget(actions)

        bottom = QHBoxLayout()
        guide = QPushButton("Labeling Guide")
        guide.clicked.connect(self._show_labeling_guide)
        bottom.addWidget(guide)
        bottom.addStretch()
        proceed = QPushButton("Proceed to Pipeline")
        proceed.clicked.connect(self.navigate_pipeline.emit)
        bottom.addWidget(proceed)
        lay.addLayout(bottom)
        lay.addStretch()

    def _refresh_project_status(self):
        path = self._path_le.text().strip()
        if not path:
            self._project_status.setText("")
            return
        cfg = os.path.join(path, "config.yaml")
        if os.path.exists(cfg):
            self._project_status.setText(f"Valid DLC project: {cfg}")
            self._project_status.setStyleSheet("color:#2e7d32;")
            try:
                import vieb_config
                vieb_config.set_dlc_project_path(path)
            except Exception:
                pass
        else:
            self._project_status.setText("config.yaml not found in this directory.")
            self._project_status.setStyleSheet("color:#c62828;")

    def _browse_project(self):
        config_file, _ = QFileDialog.getOpenFileName(
            self, "Select DLC config.yaml", str(ROOT), "DLC Config (config.yaml);;All files (*)"
        )
        if config_file:
            self._path_le.setText(os.path.dirname(config_file))

    def _refresh_pretrained(self):
        self._pretrained_combo.clear()
        try:
            from pretrained_manager import list_available_pretrained
            models = list_available_pretrained()
        except Exception:
            models = []
        if models:
            for m in models:
                self._pretrained_combo.addItem(m.get("model_name", "?"))
        else:
            self._pretrained_combo.addItem("(no pretrained models found)")

    def _use_pretrained(self):
        name = self._pretrained_combo.currentText()
        if not name or name.startswith("("):
            QMessageBox.information(self, "No Model", "No pretrained models found in pretrained/.")
            return
        raw_dir = self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")).replace("\\", "\\\\")
        code = (
            "from pretrained_manager import load_pretrained_model, analyze_with_pretrained; "
            f"load_pretrained_model({name!r}, {raw_dir!r}); "
            f"analyze_with_pretrained({name!r}, {raw_dir!r})"
        )
        self._run_dlc_subprocess(["-c", code])

    def _run_dlc_subprocess(self, args: list[str]):
        if self._worker and self._worker.isRunning():
            self._log.insertPlainText("A DLC task is already running.\n")
            return
        self._worker = SubprocessWorker(args)
        self._worker.log.connect(lambda t: self._log.insertPlainText(t))
        self._worker.done.connect(lambda ok: self._log.insertPlainText(f"\n{'OK' if ok else 'FAIL'}: {' '.join(args)}\n"))
        self._worker.start()

    def _show_labeling_guide(self):
        QMessageBox.information(self, "Labeling Guide", _LABELING_GUIDE)

    def _detect_and_show_status(self):
        project_path = None
        try:
            import vieb_config
            project_path = vieb_config.get_dlc_project_path()
        except Exception:
            pass
        if not project_path:
            dlc_dir = _find_dlc_project()
            project_path = str(dlc_dir) if dlc_dir else None
        raw_dir = Path(self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")))
        csv_count = len(list(raw_dir.glob("*DLC*.csv"))) if raw_dir.exists() else 0
        label_count = len(list((RESULTS / "shared").glob("*_labels.npy"))) if (RESULTS / "shared").exists() else 0
        if not any([project_path, csv_count, label_count]):
            return
        parts = []
        if project_path:
            parts.append(f"DLC project: {project_path}")
            self._path_le.setText(project_path)
        if csv_count:
            parts.append(f"{csv_count} pose CSV(s) found.")
        if label_count:
            parts.append(f"{label_count} state label file(s) found.")
        self._done_lbl.setText(" | ".join(parts) + " You do not need to redo DLC unless adding videos.")
        self._done_banner.show()


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


class StateDetailDialog(QDialog):
    clips_generated = pyqtSignal()
    request_clip_generation = pyqtSignal(int)

    def __init__(self, sid, s_row=None, c_row=None, bouts_df=None, cfg=None, parent=None):
        super().__init__(parent)
        self.sid = sid
        self.s_row = s_row
        self.c_row = c_row
        self.bouts_df = bouts_df
        self.cfg = cfg or {}
        self._player = None
        self._clip_sections = []
        self._clip_files = []
        self._clip_page = 0
        self._clips_per_page = 12
        self.setWindowTitle(f"State {sid} Details")
        self.resize(840, 580)
        self.setMinimumSize(520, 360)
        self.setSizeGripEnabled(True)
        self.setModal(False)
        self._build()

    def _clip_groups(self):
        sd = CLIPS / f"state_{self.sid}"
        if not sd.exists():
            return {}, False
        return {
            "Longest Bouts": sorted(sd.glob("longest_*.mp4"))[:5],
            "Most Typical": sorted(sd.glob("typical_*.mp4"))[:5],
            "Context-Enriched": sorted(sd.glob("context_*.mp4"))[:5],
        }, True

    def _meta_for(self, clip: Path):
        if self.bouts_df is None or self.bouts_df.empty:
            return "duration: -, bout: -, context: -, animal: -, day: -"
        sub = self.bouts_df[self.bouts_df["state"] == self.sid]
        if sub.empty:
            return "duration: -, bout: -, context: -, animal: -, day: -"
        if clip.name.startswith("context_"):
            c = clip.name.split("_")[1]
            sub = sub[sub["context"].astype(str) == c]
        sub = sub.sort_values("duration_sec", ascending=False)
        row = sub.iloc[0]
        return (
            f"duration: {float(row.get('duration_sec', 0)):.2f}s, "
            f"bout: {int(row.get('end_frame', 0) - row.get('start_frame', 0) + 1)} fr, "
            f"context: {row.get('context', '-')}, animal: {row.get('animal_id', '-')}, day: {row.get('day', '-')}"
        )

    def _build_kinematics(self, tabs):
        w = QWidget()
        lay = QVBoxLayout(w)
        if _init_mpl() and self.s_row is not None:
            c = MplCanvas(figsize=(6, 3))
            metrics = {
                "Speed": self.s_row.get("mean_centroid_speed", 0),
                "AngVel": self.s_row.get("mean_angular_vel", 0),
                "BodyLen": self.s_row.get("mean_body_length_px", 0),
                "Elongation": self.s_row.get("mean_elongation", 0),
                "Entropy": self.s_row.get("mean_entropy", 0),
                "BoutSec": self.s_row.get("mean_bout_dur_sec", 0),
            }
            c.ax.bar(list(metrics.keys()), [float(v or 0) for v in metrics.values()], color="#4a90d9")
            c.ax.set_title("Kinematic Profile")
            c.fig.tight_layout()
            lay.addWidget(c)
        else:
            lay.addWidget(QLabel("Run Characterization + Clip Export to generate state kinematics."))
        tabs.addTab(w, "Kinematics")

    def _start_generate(self):
        self.request_clip_generation.emit(self.sid)
        QMessageBox.information(
            self,
            "Background Job Started",
            "Clip generation is running in the background. You can keep navigating the app.",
        )
        self.close()

    def _build_clips(self, tabs):
        w = QWidget()
        lay = QVBoxLayout(w)
        clip_dir = CLIPS / f"state_{self.sid}"
        self._clip_files = sorted(clip_dir.glob("*.mp4")) if clip_dir.exists() else []
        if self._clip_files and _CV2:
            self._clips_grid_host = QWidget()
            self._clips_grid = QGridLayout(self._clips_grid_host)
            self._clips_grid.setSpacing(10)
            sc = QScrollArea()
            sc.setWidgetResizable(True)
            sc.setWidget(self._clips_grid_host)
            lay.addWidget(sc, stretch=1)
            page_row = QHBoxLayout()
            self._prev_page_btn = QPushButton("Previous")
            self._prev_page_btn.clicked.connect(self._prev_clip_page)
            self._page_lbl = QLabel("")
            self._page_lbl.setAlignment(Qt.AlignCenter)
            self._next_page_btn = QPushButton("Next")
            self._next_page_btn.clicked.connect(self._next_clip_page)
            page_row.addWidget(self._prev_page_btn)
            page_row.addWidget(self._page_lbl, stretch=1)
            page_row.addWidget(self._next_page_btn)
            lay.addLayout(page_row)
            self._render_clip_page()
        else:
            lay.addWidget(QLabel("Clips are not available for this state yet."))
            self._gen_btn = QPushButton("Generate clips for this state")
            self._gen_btn.clicked.connect(self._start_generate)
            lay.addWidget(self._gen_btn)
            lay.addWidget(QLabel("Generation runs in background so you can continue browsing."))
        tabs.addTab(w, "Video Clips")

    def _render_clip_page(self):
        while self._clips_grid.count():
            item = self._clips_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        total_pages = max(1, math.ceil(len(self._clip_files) / self._clips_per_page))
        self._clip_page = max(0, min(self._clip_page, total_pages - 1))
        start = self._clip_page * self._clips_per_page
        page_files = self._clip_files[start:start + self._clips_per_page]
        for idx, clip in enumerate(page_files):
            card = QWidget()
            cl = QVBoxLayout(card)
            b = QPushButton("")
            pm = _thumb_from_video(clip, size=(220, 135))
            if pm is not None:
                b.setIcon(QIcon(pm))
                b.setIconSize(pm.size())
            b.setFixedSize(230, 145)
            b.clicked.connect(lambda _, p=clip: self._open_clip_dialog(p))
            name_lbl = QLabel(clip.name)
            name_lbl.setAlignment(Qt.AlignCenter)
            name_lbl.setWordWrap(True)
            cl.addWidget(b, alignment=Qt.AlignCenter)
            cl.addWidget(name_lbl)
            self._clips_grid.addWidget(card, idx // 4, idx % 4)
        self._page_lbl.setText(f"Page {self._clip_page + 1} of {total_pages}")
        self._prev_page_btn.setEnabled(self._clip_page > 0)
        self._next_page_btn.setEnabled(self._clip_page < total_pages - 1)

    def _open_clip_dialog(self, path: Path):
        dlg = QDialog(self)
        dlg.setWindowTitle(path.name)
        lay = QVBoxLayout(dlg)
        player = VideoPlayer(parent=dlg)
        lay.addWidget(player)
        player.load(str(path))
        dlg.resize(720, 560)
        dlg.exec_()

    def _prev_clip_page(self):
        self._clip_page = max(0, self._clip_page - 1)
        self._render_clip_page()

    def _next_clip_page(self):
        total_pages = max(1, math.ceil(len(self._clip_files) / self._clips_per_page))
        self._clip_page = min(total_pages - 1, self._clip_page + 1)
        self._render_clip_page()

    def _update_clip_card_sizes(self):
        if not self._clip_sections:
            return
        for sec in self._clip_sections:
            cards = sec["cards"]
            if not cards:
                continue
            viewport_w = sec["scroll"].viewport().width()
            n_cards = len(cards)
            avail = max(220, viewport_w - (n_cards + 1) * 10)
            card_w = max(150, min(300, int(avail / max(1, n_cards))))
            thumb_w = int(card_w * 0.92)
            thumb_h = max(88, int(thumb_w * 0.62))
            for item in cards:
                card = item["card"]
                btn = item["button"]
                pm = item["pixmap"]
                card.setFixedWidth(card_w)
                btn.setFixedSize(thumb_w, thumb_h)
                if pm is not None:
                    scaled = pm.scaled(
                        max(40, thumb_w - 8),
                        max(30, thumb_h - 8),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    btn.setIcon(QIcon(scaled))
                    btn.setIconSize(scaled.size())

    def _build(self):
        lay = QVBoxLayout(self)
        hdr = QLabel(f"State {self.sid}")
        hdr.setFont(QFont("Arial", 14, QFont.Bold))
        lay.addWidget(hdr)
        tabs = QTabWidget()
        self._build_kinematics(tabs)
        self._build_clips(tabs)
        lay.addWidget(tabs)
        btn = QDialogButtonBox(QDialogButtonBox.Close)
        btn.rejected.connect(self.reject)
        lay.addWidget(btn)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_clip_card_sizes()


class StateCard(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, sid, s_row=None, c_row=None, strength=None):
        super().__init__()
        self.sid = sid
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(200, 235)
        self.setStyleSheet(
            "QFrame{background:#fff;border:none;border-radius:8px;}"
            "QFrame:hover{background:#f5f9ff;}"
        )
        lay = QVBoxLayout(self)
        th = QLabel(alignment=Qt.AlignCenter)
        th.setFixedSize(170, 100)
        th.setStyleSheet("background:#eee;border-radius:4px;")
        pm = _thumb_from_video(next(iter((CLIPS / f"state_{sid}").glob("*.mp4")), None)) if (CLIPS / f"state_{sid}").exists() else None
        if pm:
            th.setPixmap(pm)
        else:
            th.setText(f"State {sid}")
        lay.addWidget(th, alignment=Qt.AlignCenter)
        lab = QLabel(f"State {sid}")
        lab.setFont(QFont("Arial", 10, QFont.Bold))
        lay.addWidget(lab)
        if s_row is not None:
            hl = str(s_row.get("heuristic_label", ""))
            lay.addWidget(QLabel((hl[:32] + "...") if len(hl) > 32 else hl))
        if c_row is not None:
            scores = {
                "A": float(c_row.get("A_enrichment", 0) or 0),
                "B": float(c_row.get("B_enrichment", 0) or 0),
                "C": float(c_row.get("C_enrichment", 0) or 0),
            }
            best = max(scores, key=scores.get)
            lay.addWidget(QLabel(f"Context {best} enriched"))
        if strength is not None and not np.isnan(strength):
            lay.addWidget(QLabel(f"Strength: {100 * float(strength):.1f}%"))
        lay.addStretch()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.sid)


class BrowseStatesView(QWidget):
    navigate_to_pipeline = pyqtSignal()
    request_clip_generation = pyqtSignal(int)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._data = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        t = QLabel("Browse States")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        lay.addWidget(t)
        flt = QHBoxLayout()
        flt.addWidget(QLabel("Context"))
        self._context = QComboBox()
        self._context.currentTextChanged.connect(self._rerender)
        flt.addWidget(self._context)
        flt.addWidget(QLabel("Day"))
        self._day = QComboBox()
        self._day.currentTextChanged.connect(self._rerender)
        flt.addWidget(self._day)
        self._hide_leading = QCheckBox("Hide leading state and rescale")
        self._hide_leading.setChecked(True)
        self._hide_leading.toggled.connect(self._rerender)
        flt.addWidget(self._hide_leading)
        flt.addStretch()
        lay.addLayout(flt)
        self._status = QLabel("No data loaded.")
        lay.addWidget(self._status)
        self._placeholder_btn = QPushButton("Run Characterization + Clip Export")
        self._placeholder_btn.clicked.connect(self.navigate_to_pipeline.emit)
        self._placeholder_btn.hide()
        lay.addWidget(self._placeholder_btn)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._gw = QWidget()
        self._grid = QGridLayout(self._gw)
        self._grid.setSpacing(12)
        scroll.setWidget(self._gw)
        lay.addWidget(scroll)
        self._detail_dialogs = []

    def _configure_filters(self, summary):
        self._context.blockSignals(True)
        self._day.blockSignals(True)
        self._context.clear()
        self._day.clear()
        self._context.addItem("All")
        self._day.addItem("All")
        groups = [s.strip() for s in str(self.cfg.get("context_groups", "A,B,C")).split(",") if s.strip()]
        data_ctx = []
        if summary is not None and "context" in summary.columns:
            data_ctx = sorted(summary["context"].dropna().astype(str).unique().tolist())
        existing = set()
        for i in range(self._context.count()):
            existing.add(self._context.itemText(i))
        for g in groups:
            if g not in existing:
                self._context.addItem(g)
                existing.add(g)
        for c in data_ctx:
            if c not in existing:
                self._context.addItem(c)
        if summary is not None and "day" in summary.columns:
            for d in sorted(summary["day"].dropna().astype(str).unique()):
                self._day.addItem(str(d))
        self._context.blockSignals(False)
        self._day.blockSignals(False)

    def _filtered_summary(self):
        summary = self._data.get("summary")
        if summary is None:
            return None
        df = summary.copy()
        ctx = self._context.currentText()
        day = self._day.currentText()
        if ctx and ctx != "All" and "context" in df.columns:
            df = df[df["context"].astype(str) == ctx]
        if day and day != "All" and "day" in df.columns:
            df = df[df["day"].astype(str) == day]
        return df

    def _rerender(self):
        self._render_cards()

    def update_data(self, data):
        self._data = data
        self._configure_filters(data.get("summary"))
        self._render_cards()

    def _render_cards(self):
        while self._grid.count():
            i = self._grid.takeAt(0)
            if i.widget():
                i.widget().deleteLater()
        ci = self._data.get("cluster_info")
        ss = self._data.get("state_summary")
        cr = self._data.get("context_report")
        if not ci:
            self._status.setText("Run HDBSCAN Clustering to generate this data.")
            self._placeholder_btn.show()
            return
        self._placeholder_btn.hide()
        n = int(ci.get("n_clusters", 0))
        if n <= 0:
            self._status.setText("No states discovered.")
            return
        leading = None
        summary = self._filtered_summary()
        strength_by_state = {}
        if summary is not None and not summary.empty:
            state_ids, vals, leading = _state_means(summary, n, self._hide_leading.isChecked())
            strength_by_state = {sid: val for sid, val in zip(state_ids, vals)}
        ctx = self._context.currentText() or "All"
        day = self._day.currentText() or "All"
        if self._hide_leading.isChecked() and leading is not None:
            self._status.setText(
                f"{n} states discovered. Context: {ctx}, Day: {day}. "
                f"Leading state {leading} hidden; remaining states rescaled."
            )
        else:
            self._status.setText(f"{n} states discovered. Context: {ctx}, Day: {day}.")
        row = col = 0
        for sid in range(n):
            if self._hide_leading.isChecked() and sid == leading:
                continue
            s_row = None
            c_row = None
            if ss is not None and "state" in ss.columns:
                r = ss[ss["state"] == sid]
                if not r.empty:
                    s_row = r.iloc[0]
            if cr is not None and "state" in cr.columns:
                r = cr[cr["state"] == sid]
                if not r.empty:
                    c_row = r.iloc[0]
            c = StateCard(sid, s_row, c_row, strength=strength_by_state.get(sid))
            c.clicked.connect(self._open_detail)
            self._grid.addWidget(c, row, col)
            col += 1
            if col >= 5:
                col = 0
                row += 1

    def _open_detail(self, sid):
        ss = self._data.get("state_summary")
        cr = self._data.get("context_report")
        bouts = self._data.get("bouts")
        s_row = c_row = None
        if ss is not None and "state" in ss.columns:
            r = ss[ss["state"] == sid]
            if not r.empty:
                s_row = r.iloc[0]
        if cr is not None and "state" in cr.columns:
            r = cr[cr["state"] == sid]
            if not r.empty:
                c_row = r.iloc[0]
        dlg = StateDetailDialog(sid, s_row, c_row, bouts_df=bouts, cfg=self.cfg, parent=self)
        dlg.request_clip_generation.connect(self.request_clip_generation.emit)
        dlg.clips_generated.connect(lambda: None)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        self._detail_dialogs.append(dlg)
        self._detail_dialogs = [d for d in self._detail_dialogs if d.isVisible()]


class ValidationView(QWidget):
    navigate_to_pipeline = pyqtSignal()
    navigate_help = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._data = {}
        self._sample = None
        self._cursor = 0
        self._pose_cache = {}
        self._cap_cache = {}
        self._feature_cache = {}
        self._label_map = {
            Qt.Key_F: "freeze",
            Qt.Key_W: "walk",
            Qt.Key_G: "groom",
            Qt.Key_R: "rear",
            Qt.Key_O: "other",
            Qt.Key_S: "skip",
        }
        self.setFocusPolicy(Qt.StrongFocus)
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        _val_top = QHBoxLayout()
        title = QLabel("Validation")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        _val_top.addWidget(title)
        _val_hbtn = QPushButton("?")
        _val_hbtn.setFixedSize(20, 20)
        _val_hbtn.setFlat(True)
        _val_hbtn.setToolTip("Open Help for Clip Reviewer")
        _val_hbtn.setCursor(Qt.PointingHandCursor)
        _val_hbtn.setStyleSheet(
            "QPushButton{border:1px solid #aaa;border-radius:10px;color:#555;"
            "background:#f5f5f5;font-size:10px;font-weight:bold;}"
            "QPushButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
        )
        _val_hbtn.clicked.connect(lambda: self.navigate_help.emit("clip_reviewer"))
        _val_top.addWidget(_val_hbtn)
        _val_top.addStretch()
        outer.addLayout(_val_top)

        split = QHBoxLayout()
        outer.addLayout(split, stretch=1)

        left = QGroupBox("Frame Sampler")
        ll = QVBoxLayout(left)
        self._video_combo = QComboBox()
        self._state_combo = QComboBox()
        self._n_slider = QSlider(Qt.Horizontal)
        self._n_slider.setRange(10, 200)
        self._n_slider.setValue(50)
        self._n_lbl = QLabel("Frames: 50")
        self._n_slider.valueChanged.connect(lambda v: self._n_lbl.setText(f"Frames: {v}"))
        self._sample_btn = QPushButton("Sample Frames From Video")
        self._sample_btn.clicked.connect(self._sample_frames)
        ll.addWidget(QLabel("Video to Validate"))
        ll.addWidget(self._video_combo)
        ll.addWidget(QLabel("State"))
        ll.addWidget(self._state_combo)
        ll.addWidget(self._n_lbl)
        ll.addWidget(self._n_slider)
        ll.addWidget(self._sample_btn)
        self._progress_lbl = QLabel("0 of 0 frames labeled")
        ll.addWidget(self._progress_lbl)
        split.addWidget(left, stretch=1)

        center = QGroupBox("Frame Display")
        cl = QVBoxLayout(center)
        self._frame = QLabel("Select a video and sample frames to begin", alignment=Qt.AlignCenter)
        self._frame.setMinimumSize(540, 360)
        self._frame.setStyleSheet("background:#111;color:#999;")
        cl.addWidget(self._frame)
        self._frame_info = QLabel("State: - | kinematics: -")
        cl.addWidget(self._frame_info)
        self._shortcuts = QLabel("Shortcuts: F=freeze, W=walk, G=groom, R=rear, O=other, S=skip")
        self._shortcuts.setStyleSheet("color:#666;")
        cl.addWidget(self._shortcuts)
        split.addWidget(center, stretch=2)

        right = QGroupBox("Label Assignment")
        rl = QVBoxLayout(right)
        for name in ("Freeze", "Walk", "Groom", "Rear", "Other"):
            b = QPushButton(name)
            b.setMinimumHeight(44)
            b.clicked.connect(lambda _, n=name.lower(): self._assign(n))
            rl.addWidget(b)
        skip = QPushButton("Skip")
        skip.setMinimumHeight(44)
        skip.clicked.connect(lambda: self._assign("skip"))
        rl.addWidget(skip)
        rl.addStretch()
        split.addWidget(right, stretch=1)

        bottom = QGroupBox("Results")
        bl = QVBoxLayout(bottom)
        if _init_mpl():
            self._cm_canvas = MplCanvas(figsize=(5, 3))
            bl.addWidget(self._cm_canvas)
        else:
            self._cm_canvas = None
            bl.addWidget(QLabel("Install matplotlib for confusion matrix heatmap."))
        self._agree = QLabel("Agreement per state: -")
        bl.addWidget(self._agree)
        self._export_btn = QPushButton("Export labels CSV")
        self._export_btn.clicked.connect(self._export_validation)
        bl.addWidget(self._export_btn)
        outer.addWidget(bottom)

    def update_data(self, data):
        self._data = data
        ci = data.get("cluster_info")
        lpf = data.get("labels_per_frame")
        if ci is None or lpf is None or lpf.empty:
            self._sample_btn.setEnabled(False)
            self._progress_lbl.setText("Run Characterization + Clip Export to generate this data.")
            return
        self._sample_btn.setEnabled(True)
        n = int(ci.get("n_clusters", 0))
        summary = data.get("summary")
        dominant = -1
        if summary is not None:
            dominant = max(
                [(i, float(summary.get(f"state_{i}_frac", pd.Series([0])).mean())) for i in range(n)],
                key=lambda x: x[1],
            )[0]
        self._state_combo.clear()
        for sid in range(n):
            if sid != dominant:
                self._state_combo.addItem(f"State {sid}", sid)
        self._video_combo.clear()
        if "stem" in lpf.columns:
            stems = sorted(lpf["stem"].dropna().astype(str).unique().tolist())
            for s in stems:
                self._video_combo.addItem(s, s)
        sample = data.get("validation_sample")
        if sample is not None and not sample.empty:
            # Resume previous work only when same selected video/state exists in sample
            current_video = self._video_combo.currentData() if self._video_combo.count() else None
            current_state = self._state_combo.currentData() if self._state_combo.count() else None
            resumed = sample
            if current_video is not None and "stem" in sample.columns:
                resumed = resumed[resumed["stem"].astype(str) == str(current_video)]
            if current_state is not None and "cluster_label" in resumed.columns:
                resumed = resumed[resumed["cluster_label"] == int(current_state)]
            self._sample = resumed.reset_index(drop=True) if not resumed.empty else None
            if self._sample is not None:
                self._cursor = int((self._sample["manual_label"].fillna("") != "").sum())
                self._show_current()
                self._refresh_results()

    def _sample_frames(self):
        lpf = self._data.get("labels_per_frame")
        fi = self._data.get("feature_index") or {}
        if lpf is None or lpf.empty:
            return
        stem = self._video_combo.currentData()
        if not stem:
            QMessageBox.information(self, "Validation", "Select a video to validate.")
            return
        sid = int(self._state_combo.currentData())
        n = int(self._n_slider.value())
        sub = lpf[(lpf["state"] == sid) & (lpf["stem"].astype(str) == str(stem))]
        if sub.empty:
            QMessageBox.information(self, "Validation", "No frames available for this state in selected video.")
            return
        if len(sub) > n:
            sub = sub.sample(n=n, random_state=42)
        sub = sub.copy()
        sub.rename(columns={"frame": "frame_idx"}, inplace=True)
        sub["cluster_label"] = sub["state"]
        sub["manual_label"] = ""
        sub["timestamp"] = ""
        for idx, row in sub.iterrows():
            stem = row["stem"]
            info = fi.get(stem, {}) if isinstance(fi, dict) else {}
            sub.at[idx, "video_path"] = info.get("video_path", str(ROOT / "raw_videos" / f"{stem}.mp4"))
            sub.at[idx, "csv_path"] = info.get("csv_path", "")
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        sub.to_csv(VALIDATION_DIR / "current_sample.csv", index=False)
        self._sample = sub.reset_index(drop=True)
        self._cursor = 0
        self._show_current()
        self._refresh_results()

    def _load_pose(self, csv_path):
        if csv_path in self._pose_cache:
            return self._pose_cache[csv_path]
        try:
            from pose_io import load_pose

            pose, conf, _ = load_pose(csv_path)
            self._pose_cache[csv_path] = (pose, conf)
            return pose, conf
        except Exception:
            return None, None

    def _draw_frame(self, row):
        if not _CV2:
            return
        video = row.get("video_path", "")
        frame_idx = int(row.get("frame_idx", 0))
        csv_path = row.get("csv_path", "")
        cap = self._cap_cache.get(video)
        if cap is None:
            cap = cv2.VideoCapture(str(video))
            self._cap_cache[video] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            self._frame.setText("Could not load frame.")
            return
        pose, _ = self._load_pose(csv_path)
        if pose is not None and frame_idx < len(pose):
            pts = pose[frame_idx]
            colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (255, 128, 0),
                (128, 0, 255),
            ]
            for i, pt in enumerate(pts):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), 4, colors[i], -1)
            edges = [(2, 3), (3, 6), (0, 1)]
            for a, b in edges:
                pa = tuple(np.int32(pts[a]))
                pb = tuple(np.int32(pts[b]))
                cv2.line(frame, pa, pb, (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        mxw, mxh = self._frame.width(), self._frame.height()
        sc = min(mxw / w, mxh / h)
        nw, nh = int(w * sc), int(h * sc)
        frame = cv2.resize(frame, (nw, nh))
        buf = frame.copy()
        img = QImage(buf.data, nw, nh, 3 * nw, QImage.Format_RGB888)
        self._frame.setPixmap(QPixmap.fromImage(img))

    def _kinematic_text(self, row):
        fi = self._data.get("feature_index") or {}
        stem = row.get("stem", "")
        frame_idx = int(row.get("frame_idx", 0))
        info = fi.get(stem, {}) if isinstance(fi, dict) else {}
        fp = info.get("features_path", "")
        if not fp:
            return "-"
        arr = self._feature_cache.get(fp)
        if arr is None and Path(fp).exists():
            arr = np.load(fp)
            self._feature_cache[fp] = arr
        if arr is None or frame_idx >= len(arr):
            return "-"
        feat = arr[frame_idx]
        return f"speed={feat[36]:.3f}, ang_vel={feat[39]:.3f}, entropy={feat[40]:.3f}"

    def _show_current(self):
        if self._sample is None or self._sample.empty:
            self._frame.setText("No sample loaded.")
            return
        unl = self._sample["manual_label"].fillna("") == ""
        if unl.sum() == 0:
            self._frame.setText("All frames labeled.")
            return
        idxs = self._sample.index[unl]
        self._cursor = int(idxs[0])
        row = self._sample.loc[self._cursor]
        self._draw_frame(row)
        self._frame_info.setText(
            f"State {int(row.get('cluster_label', -1))} | "
            f"frame {int(row.get('frame_idx', 0))} | {self._kinematic_text(row)}"
        )
        done = int((self._sample["manual_label"].fillna("") != "").sum())
        self._progress_lbl.setText(f"{done} of {len(self._sample)} frames labeled")

    def _assign(self, manual_label):
        if self._sample is None or self._sample.empty:
            return
        self._sample.at[self._cursor, "manual_label"] = manual_label
        self._sample.at[self._cursor, "timestamp"] = datetime.now().isoformat(timespec="seconds")
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        self._sample.to_csv(VALIDATION_DIR / "current_sample.csv", index=False)
        self._sample.to_csv(VALIDATION_DIR / "frame_labels.csv", index=False)
        self._refresh_results()
        self._show_current()

    def _refresh_results(self):
        if self._sample is None or self._sample.empty:
            return
        done = self._sample[self._sample["manual_label"].fillna("") != ""]
        if done.empty:
            self._agree.setText("Agreement per state: -")
            if self._cm_canvas:
                self._cm_canvas.ax.clear()
                self._cm_canvas.ax.text(0.5, 0.5, "No labels yet", ha="center", va="center")
                self._cm_canvas.draw()
            return
        cros = pd.crosstab(done["cluster_label"], done["manual_label"])
        if self._cm_canvas:
            self._cm_canvas.ax.clear()
            self._cm_canvas.ax.imshow(cros.values, aspect="auto", cmap="Blues")
            self._cm_canvas.ax.set_xticks(range(len(cros.columns)))
            self._cm_canvas.ax.set_xticklabels(cros.columns, rotation=45, ha="right")
            self._cm_canvas.ax.set_yticks(range(len(cros.index)))
            self._cm_canvas.ax.set_yticklabels(cros.index)
            self._cm_canvas.ax.set_xlabel("Manual Label")
            self._cm_canvas.ax.set_ylabel("Cluster")
            self._cm_canvas.fig.tight_layout()
            self._cm_canvas.draw()
        agreements = []
        for sid, grp in done.groupby("cluster_label"):
            top = grp["manual_label"].value_counts().max()
            agreements.append(f"S{sid}: {100 * top / len(grp):.1f}%")
        self._agree.setText("Agreement per state: " + ", ".join(agreements))

    def _export_validation(self):
        p = VALIDATION_DIR / "frame_labels.csv"
        if not p.exists():
            QMessageBox.information(self, "Validation", "No labels to export yet.")
            return
        d = QFileDialog.getExistingDirectory(self, "Select Destination", str(ROOT))
        if not d:
            return
        dst = Path(d) / "frame_labels.csv"
        shutil.copy2(p, dst)
        QMessageBox.information(self, "Validation", f"Exported to {dst}")

    def keyPressEvent(self, e):
        if e.key() in self._label_map:
            self._assign(self._label_map[e.key()])
            e.accept()
            return
        super().keyPressEvent(e)


class TransitionMatrixView(QWidget):
    def __init__(self):
        super().__init__()
        self._data = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        t = QLabel("Transition Matrix")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        lay.addWidget(t)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Filter by"))
        self._f_combo = QComboBox()
        self._f_combo.addItems(["All", "Day", "Experiment"])
        self._f_combo.currentTextChanged.connect(self._update_vals)
        ctrl.addWidget(self._f_combo)
        self._v_combo = QComboBox()
        self._v_combo.currentTextChanged.connect(self._plot)
        ctrl.addWidget(self._v_combo)
        ctrl.addStretch()
        lay.addLayout(ctrl)
        if _init_mpl():
            self._canvas = MplCanvas(figsize=(12, 5))
            lay.addWidget(self._canvas)
        else:
            self._canvas = None
            lay.addWidget(QLabel("Install matplotlib."))

    def update_data(self, data):
        self._data = data
        self._update_vals(self._f_combo.currentText())
        self._plot()

    def _update_vals(self, f):
        self._v_combo.clear()
        summary = self._data.get("summary")
        if summary is None:
            self._v_combo.addItem("All")
            return
        if f == "Day" and "day" in summary.columns:
            vals = ["All"] + [str(v) for v in sorted(summary["day"].dropna().unique())]
        elif f == "Experiment" and "experiment" in summary.columns:
            vals = ["All"] + sorted(summary["experiment"].dropna().astype(str).unique())
        else:
            vals = ["All"]
        self._v_combo.addItems(vals)

    def _plot(self):
        if not self._canvas:
            return
        png = RESULTS / "comparison" / "transition_by_context.png"
        summary = self._data.get("summary")
        ci = self._data.get("cluster_info")
        ftype = self._f_combo.currentText()
        val = self._v_combo.currentText()
        if png.exists() and (ftype == "All" or val == "All"):
            img = mpimg.imread(str(png))
            self._canvas.ax.clear()
            self._canvas.ax.imshow(img)
            self._canvas.ax.axis("off")
            self._canvas.draw()
            return
        if summary is None or ci is None:
            self._canvas.ax.clear()
            self._canvas.ax.text(0.5, 0.5, "Run Report Generation to generate this data.", ha="center", va="center")
            self._canvas.draw()
            return
        n = int(ci.get("n_clusters", 0))
        df = summary.copy()
        if ftype == "Day" and val != "All" and "day" in df.columns:
            df = df[df["day"].astype(str) == val]
        if ftype == "Experiment" and val != "All" and "experiment" in df.columns:
            df = df[df["experiment"].astype(str) == val]
        if "context" not in df.columns:
            return
        state_cols = [f"state_{i}_frac" for i in range(n) if f"state_{i}_frac" in df.columns]
        contexts = sorted(df["context"].dropna().unique())[:2]
        self._canvas.fig.clear()
        for i, ctx in enumerate(contexts):
            vals = df[df["context"] == ctx][state_cols].mean().values
            mat = np.outer(vals, vals)
            if mat.max() > 0:
                mat = mat / mat.max()
            ax = self._canvas.fig.add_subplot(1, len(contexts), i + 1)
            im = ax.imshow(mat, cmap="Blues", aspect="auto")
            ax.set_title(f"Context {ctx}")
            ax.set_xlabel("To")
            ax.set_ylabel("From")
            self._canvas.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self._canvas.fig.tight_layout()
        self._canvas.draw()


class MotifsView(QWidget):
    navigate_to_pipeline = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._data = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        t = QLabel("Motifs")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        lay.addWidget(t)
        top = QHBoxLayout()
        top.addWidget(QLabel("Type"))
        self._type = QComboBox()
        self._type.addItems(["All", "bigram", "trigram"])
        self._type.currentTextChanged.connect(self._refresh)
        top.addWidget(self._type)
        top.addStretch()
        lay.addLayout(top)
        self._summary = QLabel("Motif summary: -")
        lay.addWidget(self._summary)
        if _init_mpl():
            self._canvas = MplCanvas(figsize=(9, 3))
            lay.addWidget(self._canvas)
        else:
            self._canvas = None
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["Motif", "Type", "Context A Freq", "Context B Freq", "Enrichment", "CI"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSortingEnabled(True)
        lay.addWidget(self._table)
        self._empty = QLabel("Run Motif Discovery to generate this data.")
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setStyleSheet("color:#777;")
        lay.addWidget(self._empty)
        self._btn = QPushButton("Go to Run Pipeline")
        self._btn.clicked.connect(self.navigate_to_pipeline.emit)
        lay.addWidget(self._btn)
        self._heat_lbl = QLabel()
        self._heat_lbl.setAlignment(Qt.AlignCenter)
        self._heat_lbl.hide()
        lay.addWidget(self._heat_lbl)

    def update_data(self, data):
        self._data = data
        self._refresh()

    def _refresh(self):
        motifs = self._data.get("motifs") if isinstance(self._data, dict) else None
        if motifs is None or motifs.empty:
            self._table.hide()
            self._empty.show()
            self._btn.show()
            self._summary.setText("Motif summary: no data")
            if self._canvas:
                self._canvas.ax.clear()
                self._canvas.ax.text(0.5, 0.5, "No motif data", ha="center", va="center")
                self._canvas.draw()
            return
        m = motifs.copy()
        t = self._type.currentText()
        if t != "All" and "type" in m.columns:
            m = m[m["type"].astype(str) == t]
        self._table.show()
        self._empty.hide()
        self._btn.hide()
        self._table.setRowCount(len(m))
        cols = [
            ("motif", 0),
            ("type", 1),
            ("context_A_freq", 2),
            ("context_B_freq", 3),
            ("enrichment_ratio", 4),
            ("ci_low", 5),
        ]
        for r, (_, row) in enumerate(m.iterrows()):
            for c, idx in cols:
                if c == "ci_low":
                    txt = f"[{row.get('ci_low', '')}, {row.get('ci_high', '')}]"
                else:
                    txt = str(row.get(c, ""))
                it = QTableWidgetItem(txt)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(r, idx, it)
        if "flagged" in m.columns:
            n_flag = int(m["flagged"].fillna(False).sum())
        else:
            n_flag = int((m.get("enrichment_ratio", pd.Series(dtype=float)) >= 2.0).sum())
        self._summary.setText(f"Motif summary: {len(m)} motifs, {n_flag} enriched candidates")
        if self._canvas:
            self._canvas.ax.clear()
            top = m.sort_values("enrichment_ratio", ascending=False).head(10)
            labels = [str(v) for v in top.get("motif", pd.Series(dtype=str)).tolist()]
            vals = [float(v) for v in top.get("enrichment_ratio", pd.Series(dtype=float)).fillna(0).tolist()]
            if labels:
                self._canvas.ax.barh(range(len(labels)), vals, color="#4a90d9")
                self._canvas.ax.set_yticks(range(len(labels)))
                self._canvas.ax.set_yticklabels([l[:24] + ("..." if len(l) > 24 else "") for l in labels], fontsize=8)
                self._canvas.ax.invert_yaxis()
                self._canvas.ax.set_xlabel("Enrichment Ratio")
                self._canvas.ax.set_title("Top Enriched Motifs")
            else:
                self._canvas.ax.text(0.5, 0.5, "No motifs for selected type", ha="center", va="center")
            self._canvas.fig.tight_layout()
            self._canvas.draw()
        heat = RESULTS / "comparison" / "motif_heatmap.png"
        if heat.exists():
            pm = QPixmap(str(heat))
            if not pm.isNull():
                self._heat_lbl.setPixmap(pm.scaledToWidth(680, Qt.SmoothTransformation))
                self._heat_lbl.show()
            else:
                self._heat_lbl.hide()
        else:
            self._heat_lbl.hide()


class AnimalExplorerView(QWidget):
    def __init__(self):
        super().__init__()
        self._data = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        t = QLabel("Animal Explorer")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        lay.addWidget(t)
        top = QHBoxLayout()
        top.addWidget(QLabel("Animal ID"))
        self._combo = QComboBox()
        self._combo.currentTextChanged.connect(self._update)
        top.addWidget(self._combo)
        self._all_animals = QCheckBox("Plot all animals")
        self._all_animals.toggled.connect(lambda _: self._update(self._combo.currentText()))
        top.addWidget(self._all_animals)
        top.addWidget(QLabel("Focus State"))
        self._focus_state = QComboBox()
        self._focus_state.currentTextChanged.connect(lambda _: self._update(self._combo.currentText()))
        top.addWidget(self._focus_state)
        top.addStretch()
        lay.addLayout(top)

        if _init_mpl():
            self._line = MplCanvas(figsize=(10, 3))
            lay.addWidget(self._line)
        else:
            self._line = None
            lay.addWidget(QLabel("Install matplotlib to view charts."))

        panels = QTabWidget()
        lay.addWidget(panels, stretch=1)

        self._session_table = QTableWidget()
        self._session_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._session_table.cellDoubleClicked.connect(self._open_row_folder)
        panels.addTab(self._session_table, "Session Table")

        self._disc_w = QWidget()
        dl = QVBoxLayout(self._disc_w)
        if _init_mpl():
            self._disc_canvas = MplCanvas(figsize=(10, 3))
            dl.addWidget(self._disc_canvas)
        else:
            self._disc_canvas = None
            dl.addWidget(QLabel("Install matplotlib to view chart."))
        panels.addTab(self._disc_w, "Discrimination Trajectory")

        self._heat_w = QWidget()
        hl = QVBoxLayout(self._heat_w)
        if _init_mpl():
            self._heat_canvas = MplCanvas(figsize=(5, 4))
            hl.addWidget(self._heat_canvas)
        else:
            self._heat_canvas = None
            hl.addWidget(QLabel("Install matplotlib to view chart."))
        panels.addTab(self._heat_w, "Transition Heatmap")

    def update_data(self, data):
        self._data = data
        summary = data.get("summary")
        if summary is None or "animal_id" not in summary.columns:
            return
        self._combo.blockSignals(True)
        self._combo.clear()
        animals = sorted(summary["animal_id"].dropna().astype(str).unique())
        self._combo.addItems(animals)
        self._combo.blockSignals(False)
        ci = data.get("cluster_info") or {}
        n = int(ci.get("n_clusters", 0))
        self._focus_state.blockSignals(True)
        self._focus_state.clear()
        for sid in range(n):
            self._focus_state.addItem(f"State {sid}", sid)
        self._focus_state.blockSignals(False)
        if animals:
            self._update(animals[0])

    def _open_row_folder(self, row, _col):
        stem_item = self._session_table.item(row, 0)
        if not stem_item:
            return
        stem = stem_item.text()
        path = CLIPS / stem
        if not path.exists():
            path = CLIPS
        _open_folder(path)

    def _update(self, animal_id):
        summary = self._data.get("summary")
        ci = self._data.get("cluster_info")
        if summary is None or not animal_id:
            return
        sub = summary[summary["animal_id"].astype(str) == str(animal_id)].copy()
        if sub.empty:
            return
        n = int(ci.get("n_clusters", 0)) if ci else 0
        state_cols = [f"state_{i}_frac" for i in range(n) if f"state_{i}_frac" in sub.columns]
        noise = 1 - sub[state_cols].sum(axis=1) if state_cols else pd.Series([0] * len(sub))

        headers = ["stem", "date", "day", "context", "experiment"] + state_cols + ["noise_frac"]
        self._session_table.setColumnCount(len(headers))
        self._session_table.setHorizontalHeaderLabels(headers)
        self._session_table.setRowCount(len(sub))
        for r, (_, row) in enumerate(sub.iterrows()):
            for c, h in enumerate(headers):
                if h == "noise_frac":
                    txt = f"{float(noise.iloc[r]):.3f}"
                else:
                    txt = str(row.get(h, ""))
                it = QTableWidgetItem(txt)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self._session_table.setItem(r, c, it)
        self._session_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        if self._line:
            self._line.ax.clear()
            if state_cols and "day" in summary.columns:
                if self._all_animals.isChecked():
                    sid = int(self._focus_state.currentData() or 0)
                    col = f"state_{sid}_frac"
                    if col in summary.columns:
                        colors = mpl_cm.tab20(np.linspace(0, 1, max(1, summary["animal_id"].nunique())))
                        for i, (aid, grp) in enumerate(summary.groupby("animal_id")):
                            if grp.empty:
                                continue
                            by_day = grp.groupby("day")[col].mean()
                            self._line.ax.plot(
                                by_day.index,
                                by_day.values,
                                label=str(aid),
                                color=colors[i % len(colors)],
                                linewidth=1.4,
                                alpha=0.85,
                            )
                        self._line.ax.set_title(f"State {sid} Occupancy - All Animals")
                        self._line.ax.legend(fontsize=7, ncol=2)
                else:
                    by_day = sub.groupby("day")[state_cols].mean()
                    colors = mpl_cm.tab20(np.linspace(0, 1, max(1, len(state_cols))))
                    for i, col in enumerate(state_cols[:20]):
                        sid = int(col.split("_")[1])
                        self._line.ax.plot(days := by_day.index, by_day[col], label=f"S{sid}", color=colors[i], linewidth=1.4)
                    self._line.ax.set_title(f"State Occupancy - Animal {animal_id}")
                self._line.ax.set_xlabel("Day")
                self._line.ax.set_ylabel("Fraction")
                self._line.fig.tight_layout()
            self._line.draw()

        if self._disc_canvas:
            self._disc_canvas.ax.clear()
            ratios = []
            days = sorted(sub["day"].dropna().unique()) if "day" in sub.columns else []
            for day in days:
                g = sub[sub["day"] == day]
                if "context" in g.columns:
                    cols = [c for c in state_cols if c in g.columns]
                    if not cols:
                        continue
                    freeze_col = cols[0]
                    a = g[g["context"] == "A"][freeze_col].mean()
                    b = g[g["context"] == "B"][freeze_col].mean()
                    if pd.notna(a) and pd.notna(b) and (a + b) > 0:
                        ratios.append(((a - b) / (a + b), day))
            if ratios:
                xs = [d for _, d in ratios]
                ys = [v for v, _ in ratios]
                self._disc_canvas.ax.plot(xs, ys, marker="o")
                self._disc_canvas.ax.axhline(0, color="gray", linestyle="--")
            scalars = self._data.get("animal_scalars")
            auc = np.nan
            if scalars is not None and "animal_id" in scalars.columns:
                r = scalars[scalars["animal_id"].astype(str) == str(animal_id)]
                if not r.empty:
                    auc = float(r.iloc[0].get("freeze_auc", np.nan))
            self._disc_canvas.ax.set_title(f"Discrimination Trajectory (AUC={auc:.3f})")
            self._disc_canvas.ax.set_xlabel("Day")
            self._disc_canvas.ax.set_ylabel("Discrimination Ratio")
            self._disc_canvas.fig.tight_layout()
            self._disc_canvas.draw()

        if self._heat_canvas:
            self._heat_canvas.ax.clear()
            tt = self._data.get("transition_table")
            if tt is not None and "animal_id" in tt.columns and ci:
                sub_t = tt[tt["animal_id"].astype(str) == str(animal_id)]
                n = int(ci.get("n_clusters", 0))
                cols, matrix_n = _transition_matrix_cols(sub_t, n)
                if cols and matrix_n:
                    mean_vals = sub_t[cols].mean().values
                    mat = mean_vals.reshape(matrix_n, matrix_n)
                    im = self._heat_canvas.ax.imshow(mat, cmap="Blues", aspect="auto")
                    self._heat_canvas.ax.set_title("Mean Transition Matrix")
                    self._heat_canvas.ax.set_xlabel("To")
                    self._heat_canvas.ax.set_ylabel("From")
                    self._heat_canvas.fig.colorbar(im, ax=self._heat_canvas.ax, fraction=0.046, pad=0.04)
            self._heat_canvas.fig.tight_layout()
            self._heat_canvas.draw()


class QuantificationView(QWidget):
    """Compact quantification workspace with core result browsers."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or {}
        self._data = {}
        self._cohort_df = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        lay.addWidget(self._tabs)
        self._tabs.addTab(self._build_master_tab(), "Master Table")
        self._tabs.addTab(self._build_learning_tab(), "Learning Curves")
        self._tv = TransitionMatrixView()
        self._tabs.addTab(self._tv, "Transition Matrix")
        self._mv = MotifsView()
        self._tabs.addTab(self._mv, "Motifs")
        self._av = AnimalExplorerView()
        self._tabs.addTab(self._av, "Animal Explorer")

    def _build_master_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 16, 20, 16)
        hdr = QHBoxLayout()
        title = QLabel("Master Quantification Table")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        hdr.addWidget(title)
        hdr.addStretch()
        self._quant_run_btn = QPushButton("Run Quantification")
        self._quant_run_btn.clicked.connect(self._run_quantification)
        hdr.addWidget(self._quant_run_btn)
        self._quant_export_btn = QPushButton("Export")
        self._quant_export_btn.clicked.connect(self._export_master_table)
        hdr.addWidget(self._quant_export_btn)
        lay.addLayout(hdr)
        self._quant_no_data_lbl = QLabel(
            "No quantification data found.\nClick Run Quantification to generate results/quantification/master_table.csv"
        )
        self._quant_no_data_lbl.setAlignment(Qt.AlignCenter)
        self._quant_no_data_lbl.setStyleSheet("color:#888;font-style:italic;")
        lay.addWidget(self._quant_no_data_lbl)
        self._master_table = QTableWidget(0, 0)
        self._master_table.setSortingEnabled(True)
        self._master_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._master_table.hide()
        lay.addWidget(self._master_table, stretch=1)
        return w

    def _build_learning_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(20, 16, 20, 16)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Group by:"))
        self._lc_group_combo = QComboBox()
        self._lc_group_combo.addItems(["All Animals", "Sex", "Genotype", "Treatment", "Age Group"])
        self._lc_group_combo.currentIndexChanged.connect(self._render_learning_curves)
        ctrl.addWidget(self._lc_group_combo)
        self._lc_indiv_chk = QCheckBox("Show individual animals")
        self._lc_indiv_chk.stateChanged.connect(self._render_learning_curves)
        ctrl.addWidget(self._lc_indiv_chk)
        ctrl.addStretch()
        lay.addLayout(ctrl)
        if _init_mpl():
            self._lc_canvas = MplCanvas(figsize=(10, 4))
            lay.addWidget(self._lc_canvas, stretch=1)
        else:
            self._lc_canvas = None
            lay.addWidget(QLabel("Install matplotlib to view charts."))
        return w

    def update_data(self, data):
        self._data = data
        cohort = data.get("cohort")
        if cohort is not None:
            self._cohort_df = cohort
        self._refresh_master_table()
        self._render_learning_curves()
        self._tv.update_data(data)
        self._mv.update_data(data)
        self._av.update_data(data)

    def _run_quantification(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Run Quantification")
        lay = QVBoxLayout(dlg)
        log = QTextEdit()
        log.setReadOnly(True)
        log.setStyleSheet("background:#111;color:#ddd;font-family:Consolas;")
        lay.addWidget(log)
        worker = SubprocessWorker(["quantify.py"])
        worker.log.connect(lambda t: log.insertPlainText(t))
        worker.done.connect(lambda ok: (log.insertPlainText(f"\n{'OK' if ok else 'FAIL'}\n"), self.update_data(self._data)))
        dlg._worker = worker
        worker.start()
        dlg.resize(720, 420)
        dlg.exec_()

    def _refresh_master_table(self):
        path = RESULTS / "quantification" / "master_table.csv"
        if not path.exists():
            self._master_table.hide()
            self._quant_no_data_lbl.show()
            return
        try:
            df = pd.read_csv(path)
        except Exception:
            self._master_table.hide()
            self._quant_no_data_lbl.show()
            return
        self._quant_no_data_lbl.hide()
        self._master_table.show()
        self._master_table.setRowCount(len(df))
        self._master_table.setColumnCount(len(df.columns))
        self._master_table.setHorizontalHeaderLabels(list(df.columns))
        self._master_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        for r, row in df.iterrows():
            for c, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._master_table.setItem(r, c, item)

    def _export_master_table(self):
        src = RESULTS / "quantification" / "master_table.csv"
        if not src.exists():
            QMessageBox.information(self, "Export", "No master table to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Master Table", "", "CSV files (*.csv)")
        if path:
            shutil.copy2(src, path)
            QMessageBox.information(self, "Export", f"Saved to {path}")

    def _render_learning_curves(self):
        if not self._lc_canvas or not _init_mpl():
            return
        self._lc_canvas.ax.clear()
        summary = self._data.get("summary")
        if summary is None or not {"animal_id", "day", "context"}.issubset(summary.columns):
            self._lc_canvas.ax.text(0.5, 0.5, "Run Report Generation to view learning curves.", ha="center", va="center")
            self._lc_canvas.draw()
            return
        state_cols = [c for c in summary.columns if c.startswith("state_") and c.endswith("_frac")]
        if not state_cols:
            self._lc_canvas.draw()
            return
        a_mean = summary[summary["context"].astype(str).str.upper().str.startswith("A")][state_cols].mean()
        b_mean = summary[summary["context"].astype(str).str.upper().str.startswith("B")][state_cols].mean()
        fear_col = (a_mean - b_mean).idxmax() if not a_mean.empty and not b_mean.empty else state_cols[0]
        rows = []
        for (animal, day), grp in summary.groupby(["animal_id", "day"]):
            a = grp[grp["context"].astype(str).str.upper().str.startswith("A")][fear_col].mean()
            b = grp[grp["context"].astype(str).str.upper().str.startswith("B")][fear_col].mean()
            if pd.notna(a) and pd.notna(b):
                rows.append({"animal_id": str(animal), "day": day, "disc_ratio": (a - b) / (a + b + 1e-6)})
        if not rows:
            self._lc_canvas.ax.text(0.5, 0.5, "No A/B context pairs found.", ha="center", va="center")
            self._lc_canvas.draw()
            return
        df = pd.DataFrame(rows)
        for aid, grp in df.groupby("animal_id"):
            alpha = 0.35 if self._lc_indiv_chk.isChecked() else 0
            if alpha:
                g = grp.sort_values("day")
                self._lc_canvas.ax.plot(g["day"], g["disc_ratio"], color="#999", alpha=alpha, linewidth=0.8)
        mean = df.groupby("day")["disc_ratio"].mean()
        self._lc_canvas.ax.plot(mean.index, mean.values, marker="o", color="#1a73e8", linewidth=2.5, label="Mean")
        self._lc_canvas.ax.axhline(0, color="#999", linestyle="--", linewidth=0.8)
        self._lc_canvas.ax.set_title("Fear Conditioning Learning Curve")
        self._lc_canvas.ax.set_xlabel("Day")
        self._lc_canvas.ax.set_ylabel("Discrimination Ratio")
        self._lc_canvas.ax.legend()
        self._lc_canvas.fig.tight_layout()
        self._lc_canvas.draw()


class SettingsView(QWidget):
    settings_changed = pyqtSignal(dict)
    navigate_help = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        _set_top = QHBoxLayout()
        t = QLabel("Settings")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        _set_top.addWidget(t)
        _set_hbtn = QPushButton("?")
        _set_hbtn.setFixedSize(20, 20)
        _set_hbtn.setFlat(True)
        _set_hbtn.setToolTip("Open Help for Settings")
        _set_hbtn.setCursor(Qt.PointingHandCursor)
        _set_hbtn.setStyleSheet(
            "QPushButton{border:1px solid #aaa;border-radius:10px;color:#555;"
            "background:#f5f5f5;font-size:10px;font-weight:bold;}"
            "QPushButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
        )
        _set_hbtn.clicked.connect(lambda: self.navigate_help.emit("settings"))
        _set_top.addWidget(_set_hbtn)
        _set_top.addStretch()
        lay.addLayout(_set_top)
        form = QGridLayout()
        r = 0

        def row(label, widget):
            nonlocal r
            form.addWidget(QLabel(label), r, 0)
            form.addWidget(widget, r, 1)
            r += 1

        ab = cfg.get("arena_bounds", _DEFAULT_CFG["arena_bounds"])
        self._xmin = QSpinBox(); self._xmin.setRange(0, 9999); self._xmin.setValue(ab["x_min"])
        self._ymin = QSpinBox(); self._ymin.setRange(0, 9999); self._ymin.setValue(ab["y_min"])
        self._xmax = QSpinBox(); self._xmax.setRange(0, 9999); self._xmax.setValue(ab["x_max"])
        self._ymax = QSpinBox(); self._ymax.setRange(0, 9999); self._ymax.setValue(ab["y_max"])
        row("Arena x_min", self._xmin)
        row("Arena y_min", self._ymin)
        row("Arena x_max", self._xmax)
        row("Arena y_max", self._ymax)

        def dir_row(label, key):
            nonlocal r
            le = QLineEdit(self.cfg.get(key, ""))
            b = QPushButton("Browse...")
            b.clicked.connect(lambda: self._browse(le))
            h = QHBoxLayout()
            h.addWidget(le)
            h.addWidget(b)
            form.addWidget(QLabel(label), r, 0)
            form.addLayout(h, r, 1)
            r += 1
            return le

        self._results = dir_row("Results directory", "results_dir")
        self._raw = dir_row("Raw videos directory", "raw_videos_dir")

        # Metadata CSV file picker
        self._meta_le = QLineEdit(self.cfg.get("metadata_csv_path", ""))
        meta_browse = QPushButton("Browse...")
        meta_browse.clicked.connect(lambda: self._browse_file(self._meta_le))
        meta_h = QHBoxLayout()
        meta_h.addWidget(self._meta_le)
        meta_h.addWidget(meta_browse)
        form.addWidget(QLabel("Metadata CSV file"), r, 0)
        form.addLayout(meta_h, r, 1)
        r += 1

        self._ctx_groups = QLineEdit(str(self.cfg.get("context_groups", "A,B,C")))
        row("Context groups (comma-separated)", self._ctx_groups)
        self._fps = QSpinBox()
        self._fps.setRange(1, 256)
        self._fps.setValue(int(self.cfg.get("fps", 30)))
        row("FPS", self._fps)

        _umap_tip = (
            "Number of UMAP output dimensions before HDBSCAN clustering.\n"
            "Lower values (3–5) run faster and produce coarser clusters.\n"
            "Higher values (10–15) preserve more structure. Default: 10."
        )
        self._umap_dims = QSpinBox()
        self._umap_dims.setRange(2, 50)
        self._umap_dims.setValue(int(self.cfg.get("umap_dims", 10)))
        self._umap_dims.setToolTip(_umap_tip)
        row("UMAP dimensions", self._umap_dims)

        _hms_tip = (
            "HDBSCAN min_samples controls how conservative cluster borders are.\n"
            "0 = use the same value as min_cluster_size (recommended default).\n"
            "Lower values produce more clusters with softer borders."
        )
        self._hdbscan_min_samples = QSpinBox()
        self._hdbscan_min_samples.setRange(0, 500)
        self._hdbscan_min_samples.setValue(int(self.cfg.get("hdbscan_min_samples", 0)))
        self._hdbscan_min_samples.setToolTip(_hms_tip)
        row("HDBSCAN min_samples", self._hdbscan_min_samples)

        lay.addLayout(form)

        save = QPushButton("Save Settings")
        save.clicked.connect(self._save)
        lay.addWidget(save)
        lay.addStretch()

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "Select Directory", le.text())
        if d:
            le.setText(d)

    def _browse_file(self, le):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", le.text(), "CSV files (*.csv)")
        if path:
            le.setText(path)

    def _save(self):
        self.cfg["arena_bounds"] = {
            "x_min": self._xmin.value(),
            "y_min": self._ymin.value(),
            "x_max": self._xmax.value(),
            "y_max": self._ymax.value(),
        }
        self.cfg["results_dir"] = self._results.text()
        self.cfg["raw_videos_dir"] = self._raw.text()
        self.cfg["metadata_csv_path"] = self._meta_le.text()
        self.cfg["context_groups"] = self._ctx_groups.text().strip() or "A,B,C"
        self.cfg["fps"] = self._fps.value()
        self.cfg["umap_dims"] = self._umap_dims.value()
        self.cfg["hdbscan_min_samples"] = self._hdbscan_min_samples.value()
        _save_cfg(self.cfg)
        self.settings_changed.emit(self.cfg)
        QMessageBox.information(self, "Settings", "Saved.")


class NavBtn(QPushButton):
    """Sidebar navigation button styled to match the design spec."""
    def __init__(self, text):
        icon = _NAV_ICONS.get(text, "·")
        super().__init__(f"  {icon}   {text}")
        self.setCheckable(True)
        self.setFixedHeight(38)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 0 18px;
                border: none;
                border-left: 3px solid transparent;
                background: transparent;
                font-size: 13px;
                color: #6B6B6B;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton:hover {
                background: rgba(0,0,0,0.03);
                color: #1A1A1A;
            }
            QPushButton:checked {
                border-left-color: #4E79A7;
                background: rgba(78,121,167,0.08);
                color: #1A1A1A;
                font-weight: 600;
            }
        """)




class ExportResultsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.resize(600, 520)
        self._worker = None
        lay = QVBoxLayout(self)
        self._checks = {}
        options = [
            ("summary", "Summary table (summary_table.csv)"),
            ("animal", "Animal scalars (animal_scalars.csv)"),
            ("states", "State profiles (state_summary.csv)"),
            ("transitions", "Transition matrices (transition_table.csv)"),
            ("motifs", "Motif enrichment (motifs.csv)"),
            ("plots", "All comparison plots (PNG files)"),
            ("clips", "Video clips (copies clips/ directory)"),
            ("pdf", "Full report (single PDF summary)"),
        ]
        for key, text in options:
            cb = QCheckBox(text)
            self._checks[key] = cb
            lay.addWidget(cb)
        self._run = QPushButton("Choose Destination and Export")
        self._run.clicked.connect(self._start)
        lay.addWidget(self._run)
        self._prog = QProgressBar()
        self._prog.setRange(0, 1)
        lay.addWidget(self._prog)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("background:#111;color:#ddd;")
        lay.addWidget(self._log)
        b = QDialogButtonBox(QDialogButtonBox.Close)
        b.rejected.connect(self.reject)
        lay.addWidget(b)

    def _start(self):
        opts = {k: v.isChecked() for k, v in self._checks.items()}
        if not any(opts.values()):
            QMessageBox.information(self, "Export", "Select at least one output.")
            return
        d = QFileDialog.getExistingDirectory(self, "Select destination", str(ROOT))
        if not d:
            return
        self._run.setEnabled(False)
        self._prog.setRange(0, 0)
        self._worker = ExportWorker(opts, Path(d))
        self._worker.log.connect(lambda t: self._log.insertPlainText(t))
        self._worker.done.connect(self._done)
        self._worker.start()

    def _done(self, ok):
        self._prog.setRange(0, 1)
        self._prog.setValue(1 if ok else 0)
        self._run.setEnabled(True)
        QMessageBox.information(self, "Export", "Export complete." if ok else "Export failed.")


def export_pdf_report():
    if not _init_mpl():
        raise RuntimeError("matplotlib is required for PDF export.")

    summary = RESULTS / "comparison" / "summary_table.csv"
    state_summary = RESULTS / "characterization" / "state_summary.csv"
    transition = RESULTS / "comparison" / "transition_table.csv"
    animal_scalars = RESULTS / "comparison" / "animal_scalars.csv"
    motifs = RESULTS / "comparison" / "motifs.csv"
    cluster_info = RESULTS / "shared" / "cluster_info.json"
    fi_path = RESULTS / "features" / "index.json"

    df = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    ss = pd.read_csv(state_summary) if state_summary.exists() else pd.DataFrame()
    tt = pd.read_csv(transition) if transition.exists() else pd.DataFrame()
    asc = pd.read_csv(animal_scalars) if animal_scalars.exists() else pd.DataFrame()
    mot = pd.read_csv(motifs) if motifs.exists() else pd.DataFrame()
    ci = json.loads(cluster_info.read_text(encoding="utf-8")) if cluster_info.exists() else {"n_clusters": 0}
    fi = json.loads(fi_path.read_text(encoding="utf-8")) if fi_path.exists() else {}

    out = RESULTS / f"VIEB_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(figsize=(8.3, 11.7))
        n_videos = len(df) if not df.empty else 0
        n_frames = sum(int(v.get("n_frames", 0)) for v in fi.values()) if isinstance(fi, dict) else 0
        n_states = int(ci.get("n_clusters", 0))
        noise = "-"
        if not df.empty:
            sc = [c for c in df.columns if c.startswith("state_") and c.endswith("_frac")]
            if sc:
                noise = f"{(1 - df[sc].sum(axis=1).mean()) * 100:.1f}%"
        ax.axis("off")
        ax.text(0.1, 0.9, "VIEB Dataset Overview", fontsize=20, weight="bold")
        ax.text(0.1, 0.78, f"Videos: {n_videos}\nFrames: {n_frames:,}\nStates: {n_states}\nNoise: {noise}", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8))
        if not df.empty and "context" in df.columns:
            sc = [c for c in df.columns if c.startswith("state_") and c.endswith("_frac")]
            by_ctx = df.groupby("context")[sc].mean()
            by_ctx.plot(kind="bar", ax=ax)
            ax.set_title("State occupancy by context")
        else:
            ax.text(0.5, 0.5, "No summary_table.csv available", ha="center", va="center")
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if not tt.empty and "context" in tt.columns:
            n = int(ci.get("n_clusters", 0))
            for idx, ctx in enumerate(["A", "B"]):
                sub = tt[tt["context"] == ctx]
                cols, matrix_n = _transition_matrix_cols(sub, n)
                if cols and matrix_n and not sub.empty:
                    mat = sub[cols].mean().values.reshape(matrix_n, matrix_n)
                    im = axes[idx].imshow(mat, cmap="Blues", aspect="auto")
                    axes[idx].set_title(f"Context {ctx}")
                    fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8))
        if not ss.empty:
            mcols = [c for c in ["mean_centroid_speed", "mean_angular_vel", "mean_body_length_px", "mean_elongation", "mean_entropy"] if c in ss.columns]
            mat = ss[mcols].values
            im = ax.imshow(mat, cmap="viridis", aspect="auto")
            ax.set_yticks(range(len(ss)))
            ax.set_yticklabels(ss["state"] if "state" in ss.columns else range(len(ss)))
            ax.set_xticks(range(len(mcols)))
            ax.set_xticklabels(mcols, rotation=45, ha="right")
            ax.set_title("Kinematic profile heatmap")
            fig.colorbar(im, ax=ax)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8))
        if not asc.empty and {"freeze_auc", "mean_discrimination_ratio"}.issubset(asc.columns):
            ax.scatter(asc["freeze_auc"], asc["mean_discrimination_ratio"])
            for _, r in asc.iterrows():
                ax.annotate(str(r.get("animal_id", "")), (r["freeze_auc"], r["mean_discrimination_ratio"]))
            ax.set_xlabel("AUC")
            ax.set_ylabel("Discrimination ratio")
            ax.set_title("Animal scalars")
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")
        top = mot.sort_values("enrichment_ratio", ascending=False).head(10) if not mot.empty else pd.DataFrame()
        if not top.empty:
            table_data = top[["motif", "type", "enrichment_ratio"]].astype(str).values.tolist()
            tbl = ax.table(cellText=table_data, colLabels=["Motif", "Type", "Enrichment"], loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.5)
            ax.set_title("Top 10 enriched motifs")
        else:
            ax.text(0.5, 0.5, "No motif data available", ha="center", va="center")
        pdf.savefig(fig); plt.close(fig)
    return out


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._init_project()
        self.cfg = _load_cfg()
        self.setWindowTitle("VIEB - Video Interpreter for Experimental Behavior")
        self.setMinimumSize(1024, 768)
        w, h = self.cfg.get("window_size", [1280, 800])
        self.resize(w, h)
        self._pulse_idx = 0
        self._running = False
        self._pipeline_running = False
        self._clip_running = False
        self._clip_worker = None
        self._clip_log_buf: list[str] = []
        self._initial_load_done = False
        self._cached_data = None
        self._crv = None
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.timeout.connect(self._load_data)
        self._build()
        self._load_data()
        self._start_file_watcher()
        QTimer.singleShot(200, self._maybe_onboarding)

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        ml = QHBoxLayout(central)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)

        # ── Sidebar ────────────────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("""
            QWidget#sidebar {
                background: #F4F4F4;
                border-right: 1px solid #E5E5E5;
            }
        """)
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 18, 0, 10)
        sl.setSpacing(0)

        # Brand
        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(18, 0, 18, 20)
        logo = QLabel("VIEB")
        logo.setStyleSheet(
            "font-family:'Consolas','IBM Plex Mono',monospace;"
            "font-size:16px;font-weight:600;letter-spacing:2px;color:#1A1A1A;"
            "background:transparent;border:none;"
        )
        ver = QLabel("v1.0")
        ver.setStyleSheet(
            "font-family:'Consolas','IBM Plex Mono',monospace;"
            "font-size:10px;color:#9B9B9B;background:transparent;border:none;"
        )
        brand_row.addWidget(logo)
        brand_row.addWidget(ver)
        brand_row.addStretch()
        sl.addLayout(brand_row)

        # Project name row
        proj_row = QHBoxLayout()
        proj_row.setContentsMargins(18, 0, 14, 12)
        proj_row.setSpacing(4)
        self._proj_name_lbl = QLabel("—")
        self._proj_name_lbl.setStyleSheet(
            "font-size:11px;font-weight:600;color:#3A3A3A;"
            "background:transparent;border:none;"
        )
        proj_row.addWidget(self._proj_name_lbl, stretch=1)
        self._proj_btn = QToolButton()
        self._proj_btn.setText("⊿")
        self._proj_btn.setFixedSize(22, 22)
        self._proj_btn.setCursor(Qt.PointingHandCursor)
        self._proj_btn.setStyleSheet(
            "QToolButton{border:none;background:transparent;"
            "color:#9B9B9B;font-size:12px;border-radius:3px;}"
            "QToolButton:hover{color:#1A1A1A;background:rgba(0,0,0,0.05);}"
        )
        self._proj_btn.clicked.connect(self._open_project_menu)
        proj_row.addWidget(self._proj_btn)
        sl.addLayout(proj_row)
        self._refresh_project_label()

        # Section label
        ws_lbl = QLabel("WORKSPACE")
        ws_lbl.setStyleSheet(
            "font-size:10px;font-weight:600;letter-spacing:2px;color:#9B9B9B;"
            "padding:6px 18px;background:transparent;border:none;"
            "text-transform:uppercase;"
        )
        sl.addWidget(ws_lbl)

        # Nav buttons
        self._nav = {}
        for name in _NAV_VIEWS:
            b = NavBtn(name)
            b.clicked.connect(lambda _, n=name: self._switch(n))
            sl.addWidget(b)
            self._nav[name] = b
        sl.addStretch()

        # Footer
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#E5E5E5;background:#E5E5E5;border:none;max-height:1px;")
        sl.addWidget(sep)
        self._sb_footer = QLabel("No project loaded")
        self._sb_footer.setStyleSheet(
            "font-family:'Consolas','IBM Plex Mono',monospace;"
            "font-size:10px;color:#9B9B9B;padding:10px 18px;"
            "background:transparent;border:none;line-height:1.6;"
        )
        self._sb_footer.setWordWrap(True)
        sl.addWidget(self._sb_footer)

        reload_btn = QPushButton("⟳  Reload data")
        reload_btn.setToolTip(
            "Re-read all results from disk.\n"
            "Use this after running compare.py or characterize.py from the terminal."
        )
        reload_btn.setStyleSheet(
            "QPushButton {"
            "  background:transparent; border:none; color:#9B9B9B;"
            "  font-size:11px; padding:6px 18px; text-align:left;"
            "}"
            "QPushButton:hover { color:#1a73e8; background:#EBEBEB; }"
        )
        reload_btn.clicked.connect(self._manual_reload)
        sl.addWidget(reload_btn)

        ml.addWidget(sidebar)

        self._stack = QStackedWidget()
        ml.addWidget(self._stack)
        self._views = {}

        def add(name, widget):
            self._views[name] = widget
            self._stack.addWidget(widget)

        self._ov = OverviewView()
        self._ov.export_requested.connect(self._show_export_dialog)
        self._ov.load_previous_requested.connect(self._load_session)
        self._ov.cohort_path_changed.connect(self._on_cohort_path_changed)
        self._ov.navigate_help.connect(self._navigate_to_help)
        add("Overview", self._ov)

        self._dlc = DLCSetupView(self.cfg)
        self._dlc.navigate_pipeline.connect(lambda: self._switch("Pipeline"))
        self._views["DLC Setup"] = self._dlc
        self._stack.addWidget(self._dlc)

        self._pv = RunPipelineView(self.cfg)
        self._pv.pipeline_done.connect(self._load_data)
        self._pv.worker_running.connect(self._set_running)
        self._pv.navigate_dlc.connect(lambda: self._switch("DLC Setup"))
        self._pv.cluster_finished.connect(self._show_cluster_runs)
        self._pv.navigate_help.connect(self._navigate_to_help)
        add("Pipeline", self._pv)

        if _HAS_CLUSTER_RUNS_VIEW:
            self._crv = _ClusterRunsView(self.cfg)
            add("Cluster Runs", self._crv)
        else:
            self._crv = None

        self._sv = BrowseStatesView(self.cfg)
        self._sv.navigate_to_pipeline.connect(lambda: self._switch("Pipeline"))
        self._sv.request_clip_generation.connect(self._start_background_clip_generation)
        add("Browse States", self._sv)

        if _HAS_ANALYSIS_VIEW:
            self._av = _AnalysisView(self.cfg)
            self._av.worker_running.connect(self._set_running)
            add("Analysis", self._av)
        else:
            self._av = None

        self._vv = ValidationView(self.cfg)
        self._vv.navigate_to_pipeline.connect(lambda: self._switch("Pipeline"))
        self._vv.navigate_help.connect(self._navigate_to_help)
        add("Validation", self._vv)

        self._qv = QuantificationView(self.cfg)
        add("Quantification", self._qv)

        self._setv = SettingsView(self.cfg)
        self._setv.settings_changed.connect(self._settings_changed)
        self._setv.navigate_help.connect(self._navigate_to_help)
        add("Settings", self._setv)

        self._hv = HelpView()
        add("Help", self._hv)

        self._build_status_bar()
        self._switch(self.cfg.get("last_view", "Overview"))

    def _build_status_bar(self):
        sb: QStatusBar = self.statusBar()
        sb.setSizeGripEnabled(False)

        mono = "font-family:'Consolas','Courier New',monospace;font-size:11px;"

        self._sb_run    = QLabel("Last run: —")
        self._sb_vid    = QLabel("Videos: —")
        self._sb_states = QLabel("States: —")
        self._sb_noise  = QLabel("Noise: —")
        self._sb_noise.setToolTip("Frames below confidence threshold excluded from analysis")
        self._sb_dot    = QLabel("●")
        self._sb_stage  = QLabel("idle")

        for w in (self._sb_run, self._sb_vid, self._sb_states, self._sb_noise, self._sb_dot, self._sb_stage):
            w.setStyleSheet(mono + "color:#6B6B6B;background:transparent;border:none;")

        self._sb_dot.setStyleSheet(mono + "color:#9B9B9B;background:transparent;border:none;")

        # Left section
        sb.addWidget(self._sb_run)
        sb.addWidget(self._make_sb_sep())

        # Center section (permanent = right-aligned push)
        sb.addWidget(self._sb_vid)
        sb.addWidget(self._make_sb_sep())
        sb.addWidget(self._sb_states)
        sb.addWidget(self._make_sb_sep())
        sb.addWidget(self._sb_noise)

        # Right section
        sb.addPermanentWidget(self._sb_stage)
        sb.addPermanentWidget(self._make_sb_sep())
        sb.addPermanentWidget(self._sb_dot)

    def _make_sb_sep(self) -> QLabel:
        sep = QLabel("·")
        sep.setStyleSheet(
            "font-family:'Consolas','Courier New',monospace;"
            "font-size:11px;color:#C8C8C8;background:transparent;border:none;"
            "padding:0 4px;"
        )
        return sep

    def _settings_changed(self, cfg):
        self.cfg = cfg
        _save_cfg(self.cfg)
        self._refresh_view_labels()
        self._load_data()

    def _refresh_view_labels(self) -> None:
        """Re-read vocabulary labels from config and push them to live views."""
        if self._av is not None and hasattr(self._av, "refresh_labels"):
            self._av.refresh_labels()
        if hasattr(self._ov, "refresh_labels"):
            self._ov.refresh_labels()

    def _on_cohort_path_changed(self, path: str):
        self.cfg["cohort_csv_path"] = path
        _save_cfg(self.cfg)
        self._load_data()

    def _set_running(self, running: bool):
        self._pipeline_running = running
        self._sync_running()

    def _sync_running(self):
        self._running = self._pipeline_running or self._clip_running
        if self._running:
            self._pulse_timer.start(300)
        else:
            self._pulse_timer.stop()
            self._sb_dot.setStyleSheet("color:#999;")

    def _start_background_clip_generation(self, _sid: int):
        if self._clip_worker and self._clip_worker.isRunning():
            self.statusBar().showMessage("Clip generation already running in background.", 5000)
            return
        self._clip_log_buf.clear()
        self._clip_worker = ClipGenerationWorker(self.cfg)
        self._clip_worker.log.connect(self._on_clip_log)
        self._clip_worker.done.connect(self._on_clip_done)
        self._clip_running = True
        self._sync_running()
        self.statusBar().showMessage("Background clip generation started.", 5000)
        self._clip_worker.start()

    def _on_clip_log(self, text):
        self._clip_log_buf.append(text)
        if text.strip():
            self.statusBar().showMessage("Generating clips in background...", 2000)

    def _on_clip_done(self, ok):
        self._clip_running = False
        self._sync_running()
        if ok:
            self.statusBar().showMessage("Clip generation complete.", 7000)
            self._load_data()
        else:
            self.statusBar().showMessage("Clip generation failed — see error details.", 10000)
            self._show_clip_error_dialog()

    def _show_clip_error_dialog(self):
        from PyQt5.QtWidgets import QMessageBox
        log_text = "".join(self._clip_log_buf).strip() or "(no output captured)"
        msg = QMessageBox(self)
        msg.setWindowTitle("Clip Generation Failed")
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Clip generation failed.")
        msg.setDetailedText(log_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def _pulse(self):
        mono = "font-family:'Consolas','Courier New',monospace;font-size:11px;background:transparent;border:none;"
        if not self._running:
            self._sb_dot.setStyleSheet(mono + "color:#9B9B9B;")
            self._sb_stage.setText("idle")
            return
        self._pulse_idx = (self._pulse_idx + 1) % 2
        color = "#27AE60" if self._pulse_idx else "#6FCF97"
        self._sb_dot.setStyleSheet(mono + f"color:{color};")
        self._sb_stage.setText("running")

    def _show_cluster_runs(self):
        if self._crv is not None:
            self._crv.refresh()
        self._switch("Cluster Runs")

    def _switch(self, name):
        if name not in self._views:
            return
        for n, b in self._nav.items():
            b.setChecked(n == name)
        self._stack.setCurrentWidget(self._views[name])
        self.cfg["last_view"] = name

    def _navigate_to_help(self, section_id: str):
        self._switch("Help")
        self._hv.scroll_to_section(section_id)

    # ── File watcher — auto-refresh when pipeline writes new results ──────────

    _WATCH_FILES = [
        "shared/cluster_info.json",
        "comparison/summary_table.csv",
        "characterization/state_summary.csv",
    ]

    def _start_file_watcher(self) -> None:
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_watched_file_changed)
        self._watch_result_files()

    def _watch_result_files(self) -> None:
        for rel in self._WATCH_FILES:
            p = str(RESULTS / rel)
            if Path(p).exists() and p not in self._watcher.files():
                self._watcher.addPath(p)

    def _on_watched_file_changed(self, path: str) -> None:
        # Qt removes a path from the watch list when the file is replaced;
        # re-add it and debounce the reload to 1 s so rapid writes don't spam.
        if Path(path).exists():
            self._watcher.addPath(path)
        if not self._reload_timer.isActive():
            self._reload_timer.start(1000)
        self.statusBar().showMessage("Results updated — reloading…", 3000)

    def _manual_reload(self) -> None:
        self._watch_result_files()   # pick up any new files that weren't there before
        self._load_data()
        self.statusBar().showMessage("Reloading results from disk…", 2000)

    def _load_data(self):
        self._loader = DataLoader(self.cfg.get("cohort_csv_path", ""))
        self._loader.loaded.connect(self._on_loaded)
        self._loader.error.connect(lambda e: self.statusBar().showMessage(f"Load error: {e}", 6000))
        self._loader.start()

    def _on_loaded(self, data):
        self._watch_result_files()   # add any newly created result files
        summary = data.get("summary")
        ci = data.get("cluster_info")
        self._cached_data = data
        if summary is not None:
            self._sb_vid.setText(f"Videos: {len(summary)}")
        if ci:
            self._sb_states.setText(f"States: {ci.get('n_clusters', '-')}")
        p = RESULTS / "comparison" / "summary_table.csv"
        if p.exists():
            self._sb_run.setText(f"Last run: {_fmt_ts(p.stat().st_mtime)}")
        if summary is not None:
            state_cols = [c for c in summary.columns if c.startswith("state_") and c.endswith("_frac")]
            if state_cols:
                noise = (1 - summary[state_cols].sum(axis=1).mean()) * 100
                self._sb_noise.setText(f"Noise: {noise:.1f}%")
            # Update sidebar footer
            n_animals = summary["animal_id"].nunique() if "animal_id" in summary.columns else "—"
            n_days = summary["day"].nunique() if "day" in summary.columns else "—"
            self._sb_footer.setText(
                f"Project\n{self.cfg.get('project_name', 'VIEB')}\n"
                f"{n_animals} animals · {n_days} days"
            )

        self._pv.estimate_times(data)
        self._pv.update_from_cfg()
        self._pv.update_cluster_quality(data)
        if not self._initial_load_done:
            self._initial_load_done = True
            has_results = any(data.get(k) is not None for k in ("summary", "state_summary", "motifs", "animal_scalars"))
            self._ov.show_load_banner(has_results)
        self._ov.update_data(data)
        self._sv.update_data(data)
        self._vv.update_data(data)
        self._qv.update_data(data)
        if self._av is not None:
            self._av.update_data(data)

    def _load_session(self):
        if self._cached_data:
            self._ov.show_load_banner(False)
            self._ov.update_data(self._cached_data)
            self._sv.update_data(self._cached_data)
            self._vv.update_data(self._cached_data)
            self._qv.update_data(self._cached_data)
            if self._av is not None:
                self._av.update_data(self._cached_data)
            self.statusBar().showMessage("Previous session results loaded.", 3000)

    def _show_export_dialog(self):
        dlg = ExportResultsDialog(self)
        dlg.exec_()

    def _maybe_onboarding(self):
        if self.cfg.get("onboarding_complete"):
            return
        self.cfg["onboarding_complete"] = True
        _save_cfg(self.cfg)
        QMessageBox.information(
            self,
            "Welcome to VIEB",
            "Welcome to VIEB. To get started, go to Pipeline and press Run.",
        )

    # ── Project management ────────────────────────────────────────────────────

    def _init_project(self):
        """Ensure an active project is selected before the window is built.

        On first launch (no app_config.json), silently migrates the existing
        ROOT/config.json to projects/<slug>/config.json and writes app_config.json.
        On subsequent launches with multiple projects and no default, shows
        ProjectSelectorDialog.
        """
        app_cfg = _load_app_config()

        # ── First launch: migrate existing config ──────────────────────────
        if not app_cfg.get("projects"):
            project_dir = _migrate_to_project()
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            # Derive a human-friendly name from what was in the config
            try:
                cfg_text = (project_dir / "config.json").read_text(encoding="utf-8")
                name = json.loads(cfg_text).get("project_name", "Luna Fear Conditioning")
            except Exception:
                name = "Luna Fear Conditioning"
            app_cfg = {
                "projects": [
                    {"name": name, "path": str(project_dir), "last_opened": now}
                ],
                "active_project": str(project_dir),
                "default_project": "",
            }
            _save_app_config(app_cfg)
            return

        projects = app_cfg.get("projects", [])
        default = app_cfg.get("default_project", "")
        active = app_cfg.get("active_project", "")

        # ── Default project set: load silently ────────────────────────────
        if default and Path(default).exists():
            if default != active:
                app_cfg["active_project"] = default
                _save_app_config(app_cfg)
            return

        # ── Single project: load silently ─────────────────────────────────
        if len(projects) == 1:
            p = projects[0]["path"]
            if p != active:
                app_cfg["active_project"] = p
                _save_app_config(app_cfg)
            return

        # ── Active project already set and valid ──────────────────────────
        if active and Path(active).exists():
            return

        # ── Multiple projects, none active: show selector ─────────────────
        try:
            from views.project_selector import ProjectSelectorDialog
            dlg = ProjectSelectorDialog(app_cfg, None)
            dlg.exec_()
            # selected_path was written to app_config.json inside _select()
        except Exception:
            pass

    def _refresh_project_label(self):
        """Update the sidebar project name label from the active project's config."""
        app_cfg = _load_app_config()
        active = app_cfg.get("active_project", "")
        name = "—"
        if active:
            cfg_path = Path(active) / "config.json"
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                    name = cfg.get("project_name", Path(active).name)
                except Exception:
                    name = Path(active).name
            else:
                # Find the name from the projects list
                for p in app_cfg.get("projects", []):
                    if p.get("path") == active:
                        name = p.get("name", name)
                        break
        # Truncate with ellipsis
        if len(name) > 22:
            name = name[:19] + "..."
        self._proj_name_lbl.setText(name)

        # Style the button based on project count
        n = len(app_cfg.get("projects", []))
        if n <= 1:
            self._proj_btn.setToolTip("Add a new project")
            self._proj_btn.setStyleSheet(
                "QToolButton{border:none;background:transparent;"
                "color:#C8C8C8;font-size:12px;border-radius:3px;}"
                "QToolButton:hover{color:#9B9B9B;background:rgba(0,0,0,0.04);}"
            )
        else:
            self._proj_btn.setToolTip("Switch project")
            self._proj_btn.setStyleSheet(
                "QToolButton{border:none;background:transparent;"
                "color:#9B9B9B;font-size:12px;border-radius:3px;}"
                "QToolButton:hover{color:#1A1A1A;background:rgba(0,0,0,0.05);}"
            )

    def _open_project_menu(self):
        app_cfg = _load_app_config()
        projects = app_cfg.get("projects", [])
        active = app_cfg.get("active_project", "")

        if len(projects) <= 1:
            # Single project: open NewProjectDialog directly
            try:
                from views.project_selector import NewProjectDialog
                dlg = NewProjectDialog(app_cfg, self)
                if dlg.exec_() == QDialog.Accepted and dlg.created_path:
                    self._do_switch(dlg.created_path)
            except Exception:
                pass
            return

        # Multiple projects: show dropdown menu
        menu = QMenu(self)
        for proj in projects:
            path = proj.get("path", "")
            pname = proj.get("name", "Unnamed")
            action = menu.addAction(pname)
            action.setCheckable(True)
            action.setChecked(path == active)
            action.triggered.connect(
                lambda checked=False, p=path: self._switch_project(p)
            )
        menu.addSeparator()
        menu.addAction("New Project...").triggered.connect(self._new_project_from_menu)
        menu.addAction("Manage Projects...").triggered.connect(self._manage_projects)

        btn_pos = self._proj_btn.mapToGlobal(self._proj_btn.rect().bottomLeft())
        menu.exec_(btn_pos)

    def _switch_project(self, path: str):
        active = _load_app_config().get("active_project", "")
        if path == active:
            return
        if self._pipeline_running or self._clip_running:
            result = QMessageBox.question(
                self, "Switch Project",
                "A pipeline is currently running. Switch project anyway?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if result != QMessageBox.Yes:
                return
        self._do_switch(path)

    def _do_switch(self, path: str):
        """Write the new active_project to app_config.json and restart."""
        app_cfg = _load_app_config()
        app_cfg["active_project"] = path
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        for p in app_cfg.get("projects", []):
            if p.get("path") == path:
                p["last_opened"] = now
        _save_app_config(app_cfg)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def _new_project_from_menu(self):
        try:
            from views.project_selector import NewProjectDialog
            app_cfg = _load_app_config()
            dlg = NewProjectDialog(app_cfg, self)
            if dlg.exec_() == QDialog.Accepted and dlg.created_path:
                self._do_switch(dlg.created_path)
        except Exception:
            pass

    def _manage_projects(self):
        try:
            from views.project_selector import ProjectSelectorDialog
            app_cfg = _load_app_config()
            dlg = ProjectSelectorDialog(app_cfg, self)
            if dlg.exec_() == QDialog.Accepted and dlg.selected_path:
                current = _load_app_config().get("active_project", "")
                if dlg.selected_path != current:
                    self._do_switch(dlg.selected_path)
                else:
                    self._refresh_project_label()
        except Exception:
            pass

    def closeEvent(self, e):
        self.cfg["window_size"] = [self.width(), self.height()]
        _save_cfg(self.cfg)
        super().closeEvent(e)


_APP_QSS = """
/* ── Global reset ─────────────────────────────────────────── */
QWidget {
    font-family: 'Segoe UI', -apple-system, Arial, sans-serif;
    font-size: 13px;
    color: #1A1A1A;
}
QLabel  { border: none; background: transparent; }
QGroupBox { border: none; }
QGroupBox::title { subcontrol-origin: margin; }

/* ── Main window ──────────────────────────────────────────── */
QMainWindow { background: #FAFAFA; }

/* ── Buttons ──────────────────────────────────────────────── */
QPushButton {
    background: #FFFFFF;
    border: 1px solid #D4D4D4;
    border-radius: 4px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
    color: #1A1A1A;
    min-height: 26px;
}
QPushButton:hover   { background: #F0F0F0; }
QPushButton:pressed { background: #E8E8E8; }
QPushButton:disabled { color: #B0B0B0; border-color: #E5E5E5; }

/* Primary blue button — use setProperty("primary","true") */
QPushButton[primary="true"] {
    background: #4E79A7;
    border-color: #4E79A7;
    color: #FFFFFF;
}
QPushButton[primary="true"]:hover   { background: #426490; border-color: #426490; }
QPushButton[primary="true"]:pressed { background: #365478; }

/* ── Inputs ───────────────────────────────────────────────── */
QLineEdit, QTextEdit {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 12px;
    color: #1A1A1A;
    selection-background-color: rgba(78,121,167,0.20);
}
QLineEdit:focus, QTextEdit:focus { border-color: #4E79A7; }

QComboBox {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 4px;
    padding: 5px 28px 5px 10px;
    font-size: 12px;
    color: #1A1A1A;
    min-height: 26px;
}
QComboBox:focus { border-color: #4E79A7; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    selection-background-color: rgba(78,121,167,0.12);
    outline: none;
}

QSpinBox {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}
QSpinBox:focus { border-color: #4E79A7; }

/* ── Status bar ───────────────────────────────────────────── */
QStatusBar {
    background: #F4F4F4;
    border-top: 1px solid #E5E5E5;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 11px;
    color: #6B6B6B;
    min-height: 28px;
}
QStatusBar::item { border: none; }

/* ── Tables ───────────────────────────────────────────────── */
QTableWidget, QTableView {
    background: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 4px;
    gridline-color: #F0F0F0;
    font-size: 12px;
    outline: none;
}
QTableWidget::item, QTableView::item {
    padding: 6px 10px;
    border: none;
    color: #1A1A1A;
}
QTableWidget::item:selected, QTableView::item:selected {
    background: rgba(78,121,167,0.12);
    color: #1A1A1A;
}
QHeaderView::section {
    background: #FBFBFB;
    border: none;
    border-bottom: 1px solid #E5E5E5;
    border-right: 1px solid #F0F0F0;
    padding: 7px 10px;
    font-size: 10px;
    font-weight: 600;
    color: #6B6B6B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Tabs ─────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #E5E5E5;
    border-top: none;
    background: #FFFFFF;
    border-radius: 0 0 6px 6px;
}
QTabBar::tab {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 9px 16px;
    font-size: 13px;
    color: #6B6B6B;
    margin-right: 2px;
}
QTabBar::tab:selected {
    color: #1A1A1A;
    border-bottom-color: #4E79A7;
    font-weight: 600;
}
QTabBar::tab:hover { color: #1A1A1A; }

/* ── Scroll bars ──────────────────────────────────────────── */
QScrollBar:vertical {
    background: transparent;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #D4D4D4;
    border-radius: 5px;
    min-height: 20px;
    margin: 2px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal {
    background: transparent;
    height: 10px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #D4D4D4;
    border-radius: 5px;
    min-width: 20px;
    margin: 2px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }

/* ── Frames / cards ───────────────────────────────────────── */
QFrame[frameShape="1"] {
    border: 1px solid #E5E5E5;
    border-radius: 6px;
    background: #FFFFFF;
}
QSplitter::handle { background: #E5E5E5; }

/* ── Checkboxes ───────────────────────────────────────────── */
QCheckBox { spacing: 6px; font-size: 12px; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #D4D4D4;
    border-radius: 3px;
    background: #FFFFFF;
}
QCheckBox::indicator:checked {
    background: #4E79A7;
    border-color: #4E79A7;
}

/* ── Sliders ──────────────────────────────────────────────── */
QSlider::groove:horizontal {
    height: 3px;
    background: #E5E5E5;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #4E79A7;
    border: none;
    width: 14px; height: 14px;
    border-radius: 7px;
    margin: -6px 0;
}
QSlider::sub-page:horizontal { background: #4E79A7; border-radius: 2px; }

/* ── Progress bar ─────────────────────────────────────────── */
QProgressBar {
    background: #ECECEC;
    border: none;
    border-radius: 2px;
    height: 4px;
    text-align: center;
    font-size: 10px;
    color: transparent;
}
QProgressBar::chunk { background: #4E79A7; border-radius: 2px; }

/* ── Tool tips ────────────────────────────────────────────── */
QToolTip {
    background: #1A1A1A;
    color: #FAFAFA;
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 12px;
}

/* ── Message boxes ────────────────────────────────────────── */
QMessageBox { background: #FFFFFF; }

/* ── Dialog ───────────────────────────────────────────────── */
QDialog { background: #FAFAFA; }
"""


def main():
    # Fix OpenCV/Qt plugin conflict: cv2 ships its own Qt5 platform plugins and
    # may set QT_QPA_PLATFORM_PLUGIN_PATH to its own cv2/qt/plugins directory.
    # We override it here — after cv2 has been imported — to point Qt at
    # PyQt5's own plugins so that libqxcb.so loads correctly.
    try:
        import importlib.util as _ilu
        _spec = _ilu.find_spec("PyQt5")
        if _spec and _spec.submodule_search_locations:
            _base = next(iter(_spec.submodule_search_locations))
            _plugins = os.path.join(_base, "Qt5", "plugins")
            if os.path.isdir(_plugins):
                os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _plugins
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("VIEB")
    app.setStyle("Fusion")
    app.setStyleSheet(_APP_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
