"""Manage versioned clustering runs in results/runs/.

Pure Python module (no Qt, no heavy ML imports). Used by both compare.py (CLI)
and views/cluster_runs.py (GUI).
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ClusterRunConfig:
    """Configuration for a single clustering run."""

    min_cluster_size: int = 50
    min_samples: int = 0  # 0 = auto
    umap_dims: int = 10
    hdbscan_sample: int = 300_000
    fps: float = 30.0
    validate: bool = False

    def resolve_min_samples(self) -> int:
        if self.min_samples > 0:
            return self.min_samples
        return max(10, min(100, self.min_cluster_size // 10))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["resolved_min_samples"] = self.resolve_min_samples()
        return d


@dataclass
class ClusterRunManifest:
    """Enriched manifest for a completed (or in-progress) run."""

    run_id: str = ""
    status: str = "queued"
    date: str = ""
    started_at: str = ""
    finished_at: str = ""
    runtime_seconds: float = 0.0

    # Config (requested)
    min_cluster_size: int = 50
    min_samples_requested: int = 0
    min_samples_resolved: int = 0
    umap_dims: int = 10
    hdbscan_sample: int = 300_000
    fps: float = 30.0
    validate: bool = False

    # Results
    n_clusters: int = 0
    mean_confidence: float = 0.0
    low_confidence_frac: float = 0.0
    noise_frac: float = 0.0
    assignment_method: str = ""

    # Diagnostics summary
    largest_state_occupancy: float = 0.0
    health_status: str = ""
    warnings_count: int = 0
    warnings: list = field(default_factory=list)

    # Error / user metadata
    error_message: str = ""
    notes: str = ""
    saved: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ClusterRunManifest:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_legacy(cls, d: dict) -> ClusterRunManifest:
        """Parse old-format run_manifest.json into ClusterRunManifest."""
        return cls(
            run_id=d.get("run_id", ""),
            status="completed" if d.get("n_clusters", 0) > 0 else "unknown",
            date=d.get("date", ""),
            min_cluster_size=d.get("min_cluster_size", 50),
            min_samples_requested=d.get("hdbscan_min_samples", 0),
            min_samples_resolved=d.get("hdbscan_min_samples", 0),
            umap_dims=d.get("umap_dims", 10),
            hdbscan_sample=d.get("hdbscan_sample", 300_000),
            n_clusters=d.get("n_clusters", 0),
            mean_confidence=d.get("mean_confidence", 0.0),
            low_confidence_frac=d.get("low_confidence_frac", 0.0),
            noise_frac=d.get("noise_frac", 0.0),
            saved=d.get("saved", False),
        )


# ---------------------------------------------------------------------------
# ClusterRunManager
# ---------------------------------------------------------------------------

class ClusterRunManager:
    """Manage versioned cluster runs in results/runs/."""

    def __init__(self, results_dir: str | Path, config_path: str | Path | None = None):
        self._results = Path(results_dir)
        self._shared = self._results / "shared"
        self._runs = self._results / "runs"
        self._config_path = Path(config_path) if config_path else None

    # --- ID generation ---

    def create_run_id(self, config: ClusterRunConfig) -> str:
        n = self._next_run_n()
        now = datetime.now()
        ms = config.resolve_min_samples()
        return (
            f"run_{n:03d}_{now.strftime('%Y%m%d_%H%M')}"
            f"_mcs{config.min_cluster_size}_ms{ms}_umap{config.umap_dims}"
        )

    def _next_run_n(self) -> int:
        max_n = 0
        if self._runs.is_dir():
            for name in os.listdir(self._runs):
                if name.startswith("run_") and (self._runs / name).is_dir():
                    try:
                        n = int(name.split("_")[1])
                        max_n = max(max_n, n)
                    except (IndexError, ValueError):
                        pass
        return max_n + 1

    # --- Run CRUD ---

    def save_run(self, run_id: str, source_dir: Path | None = None) -> Path:
        """Copy outputs from source_dir (default: shared/) to runs/run_id/."""
        source = source_dir or self._shared
        run_dir = self._runs / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        for fname in os.listdir(source):
            src = source / fname
            if src.is_file():
                shutil.copy2(src, run_dir / fname)

        # Also copy diagnostics artifacts into the run dir
        for diag_name, dest_name in [
            ("cluster_diagnostics.json", "cluster_diagnostics.json"),
            ("cluster_overview.png", "cluster_overview.png"),
            ("state_duration_summary.csv", "state_duration_summary.csv"),
        ]:
            diag_src = self._results / "diagnostics" / diag_name
            if diag_src.exists():
                shutil.copy2(diag_src, run_dir / dest_name)

        # Enrich manifest with diagnostics summary if available
        self._enrich_manifest_with_diagnostics(run_dir)

        return run_dir

    def list_runs(self) -> list[ClusterRunManifest]:
        """List all runs, sorted by run number."""
        manifests: list[ClusterRunManifest] = []
        if not self._runs.is_dir():
            return manifests
        for entry in sorted(os.listdir(self._runs)):
            run_path = self._runs / entry
            if not run_path.is_dir():
                continue
            mp = run_path / "run_manifest.json"
            if not mp.exists():
                continue
            try:
                d = json.loads(mp.read_text(encoding="utf-8"))
                if "status" in d and "min_samples_requested" in d:
                    manifests.append(ClusterRunManifest.from_dict(d))
                else:
                    manifests.append(ClusterRunManifest.from_legacy(d))
            except Exception:
                pass
        return manifests

    def load_run_manifest(self, run_id: str) -> ClusterRunManifest | None:
        mp = self._runs / run_id / "run_manifest.json"
        if not mp.exists():
            return None
        try:
            d = json.loads(mp.read_text(encoding="utf-8"))
            if "status" in d and "min_samples_requested" in d:
                return ClusterRunManifest.from_dict(d)
            return ClusterRunManifest.from_legacy(d)
        except Exception:
            return None

    def set_active_run(self, run_id: str) -> None:
        """Copy run outputs to shared/, update config.json."""
        run_dir = self._runs / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_id}")

        self._clear_cluster_outputs_from_shared()
        self._shared.mkdir(parents=True, exist_ok=True)

        for fname in os.listdir(run_dir):
            src = run_dir / fname
            if src.is_file():
                shutil.copy2(src, self._shared / fname)

        self._update_config({
            "active_cluster_run": run_id,
            "current_run_id": run_id,
            "current_run_saved": True,
        })

    def get_active_run(self) -> str:
        cfg = self._load_config()
        return cfg.get("active_cluster_run", cfg.get("current_run_id", ""))

    def delete_run(self, run_id: str) -> None:
        run_dir = self._runs / run_id
        if run_dir.is_dir():
            shutil.rmtree(run_dir)

    def get_run_dir(self, run_id: str) -> Path:
        return self._runs / run_id

    def comparison_table(self) -> list[dict]:
        """Return list of dicts suitable for DataFrame construction."""
        return [m.to_dict() for m in self.list_runs()]

    # --- Internal helpers ---

    def _clear_cluster_outputs_from_shared(self) -> None:
        if not self._shared.is_dir():
            return
        for p in list(self._shared.glob("*.npy")) + list(self._shared.glob("*.pkl")):
            p.unlink(missing_ok=True)
        for name in ("cluster_info.json", "run_manifest.json", "validation_report.json"):
            p = self._shared / name
            if p.exists():
                p.unlink()

    def _enrich_manifest_with_diagnostics(self, run_dir: Path) -> None:
        """Read cluster_diagnostics.json and merge summary fields into the run manifest."""
        diag_path = run_dir / "cluster_diagnostics.json"
        manifest_path = run_dir / "run_manifest.json"
        if not diag_path.exists() or not manifest_path.exists():
            return
        try:
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["largest_state_occupancy"] = diag.get("largest_state_frac", 0.0)
            manifest["health_status"] = diag.get("health_status", "")
            warn_list = diag.get("warnings", [])
            manifest["warnings_count"] = len(warn_list)
            manifest["warnings"] = [
                w.get("message", str(w)) if isinstance(w, dict) else str(w)
                for w in warn_list
            ]
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_config(self) -> dict:
        cfg_path = self._resolve_config_path()
        if cfg_path and cfg_path.exists():
            try:
                return json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _update_config(self, updates: dict) -> None:
        cfg_path = self._resolve_config_path()
        if cfg_path is None:
            return
        cfg = self._load_config()
        cfg.update(updates)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def _resolve_config_path(self) -> Path | None:
        if self._config_path:
            return self._config_path
        # Fall back to config.json next to this module (same as compare.py)
        return Path(os.path.dirname(os.path.abspath(__file__))) / "config.json"


# ---------------------------------------------------------------------------
# ClusterRunQueue
# ---------------------------------------------------------------------------

class ClusterRunQueue:
    """Sequential queue of ClusterRunConfig objects."""

    def __init__(self, configs: list[ClusterRunConfig] | None = None):
        self._configs: list[ClusterRunConfig] = list(configs or [])
        self._current_index: int = 0
        self._stop_after_current: bool = False
        self._cancelled: bool = False

    def add(self, config: ClusterRunConfig) -> None:
        self._configs.append(config)

    def remove(self, index: int) -> None:
        if 0 <= index < len(self._configs):
            self._configs.pop(index)

    def duplicate(self, index: int) -> None:
        if 0 <= index < len(self._configs):
            original = self._configs[index]
            copy = ClusterRunConfig(**asdict(original))
            self._configs.insert(index + 1, copy)

    def stop_after_current(self) -> None:
        self._stop_after_current = True

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def configs(self) -> list[ClusterRunConfig]:
        return list(self._configs)

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def is_done(self) -> bool:
        return (
            self._current_index >= len(self._configs)
            or self._cancelled
            or self._stop_after_current
        )

    def next_config(self) -> ClusterRunConfig | None:
        if self._stop_after_current or self._cancelled:
            return None
        if self._current_index >= len(self._configs):
            return None
        cfg = self._configs[self._current_index]
        self._current_index += 1
        return cfg
