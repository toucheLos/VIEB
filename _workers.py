#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Background worker QThread classes extracted from gui.py for VIEB."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import QObject, QThread, pyqtSignal

from _utils import (
    ROOT, RESULTS, CLIPS, _wsl_python, _wsl_path, wsl_cuml_available,
    _state_colors, _MPL, _CV2
)

try:
    from cohort_loader import load_cohort_excel
except ImportError:
    load_cohort_excel = None

if _MPL:
    from _utils import Figure, FigureCanvas, plt, PdfPages, mpl_cm, mpimg


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

            meta_p = ROOT / "metadata.csv"
            data["metadata"] = pd.read_csv(meta_p) if meta_p.exists() else None

            data["cohort"] = None
            if self._cohort_path:
                cp = Path(self._cohort_path)
                if cp.exists():
                    try:
                        if cp.suffix.lower() in (".xlsx", ".xls", ".xlsm"):
                            if load_cohort_excel is not None:
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

    def _run_subprocess(self, args):
        p = subprocess.Popen(
            [sys.executable, *args],
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
        rc = p.wait()
        return rc == 0

    def _run_cluster_wsl(self, fps: float, mcs: int) -> bool:
        """Run compare.py --cluster inside WSL2 using venv_wsl (GPU via cuML)."""
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
        assert p.stdout is not None
        for line in p.stdout:
            self.log.emit(line)
        return p.wait() == 0

    def run(self):
        cap = _Capture()
        cap.text.connect(self.log)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = cap
        ok_all = True
        cluster_bundle_ran = False
        try:
            sys.path.insert(0, str(ROOT))
            fps = float(self.cfg.get("fps", 30))
            mcs = int(self.cfg.get("min_cluster_size", 2000))
            collapse_threshold = float(self.cfg.get("collapse_threshold", 0.5))
            use_wavelets = bool(self.cfg.get("use_wavelets", True))
            enable_collapse = bool(self.cfg.get("enable_state_collapse", False))
            export_clips = bool(self.cfg.get("export_clips", False))

            for sid in self.stage_ids:
                if sid == 7 and not enable_collapse:
                    self.stage_done.emit(7, True)
                    continue

                if sid in (3, 4, 5, 6):
                    if cluster_bundle_ran:
                        continue
                    cluster_bundle_ran = True
                    for b in (3, 4, 5, 6):
                        self.stage_started.emit(b)

                    try:
                        if wsl_cuml_available():
                            # Delegate to WSL2 where cuML / GPU UMAP+HDBSCAN are available
                            ok = self._run_cluster_wsl(fps, mcs)
                        else:
                            from compare import cmd_cluster
                            cmd_cluster(fps=fps, min_cluster_size=mcs)
                            ok = True
                        if ok:
                            for b in (3, 4, 5, 6):
                                self.stage_done.emit(b, True)
                        else:
                            raise RuntimeError("WSL2 clustering subprocess returned non-zero exit code.")
                    except (Exception, SystemExit) as _exc:
                        msg = (
                            f"Clustering exited: {_exc}"
                            if isinstance(_exc, SystemExit)
                            else traceback.format_exc()
                        )
                        print(msg)
                        for b in (3, 4, 5, 6):
                            self.stage_done.emit(b, False)
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
                        from compare import cmd_extract

                        cmd_extract(fps=fps, use_wavelets=use_wavelets)
                    elif sid == 7:
                        from compare import cmd_collapse

                        cmd_collapse(threshold=collapse_threshold)
                    elif sid == 8:
                        from compare import cmd_report

                        cmd_report(fps=fps)
                    elif sid == 9:
                        from compare import cmd_summarize

                        cmd_summarize()
                    elif sid == 10:
                        from compare import cmd_motifs

                        cmd_motifs()
                    elif sid == 11:
                        from characterize import cmd_clips, cmd_summarize as csum

                        csum(fps=fps)
                        if export_clips:
                            cmd_clips(fps=fps)
                    self.stage_done.emit(sid, True)
                except (Exception, SystemExit) as _exc:
                    msg = (
                        f"Stage {sid} exited: {_exc}"
                        if isinstance(_exc, SystemExit)
                        else traceback.format_exc()
                    )
                    print(msg)
                    self.stage_done.emit(sid, False)
                    ok_all = False
                    break
        finally:
            sys.stdout, sys.stderr = old_out, old_err
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


class _CohortWorker(QThread):
    """Background worker that runs behavioral_fingerprint.py or plot_cohort.py."""
    log  = pyqtSignal(str)
    done = pyqtSignal(bool)

    def __init__(self, args: list[str]):
        super().__init__()
        self._args = args

    def run(self):
        import subprocess
        p = subprocess.Popen(
            [sys.executable, *self._args],
            cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        assert p.stdout is not None
        for line in p.stdout:
            self.log.emit(line)
        self.done.emit(p.wait() == 0)
