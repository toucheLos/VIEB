"""Background worker for pipeline runs launched from the GUI.

The pipeline takes seconds to hours depending on the project, so running it on
the GUI thread would freeze the window and make the app look crashed. This wraps
it in a QThread and reports progress, completion and failure as signals.

Failures are delivered as an `failed` signal rather than raised: an exception on
a worker thread would otherwise vanish, leaving a Run button that never comes
back. Anything the backend raises reaches the user as a message.
"""

from __future__ import annotations

import traceback

from PyQt5.QtCore import QThread, pyqtSignal


class PipelineWorker(QThread):
    """Runs `representation.pipeline.run` off the GUI thread."""

    progress = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, pose_dir, out_dir, options, parent=None):
        super().__init__(parent)
        self.pose_dir = pose_dir
        self.out_dir = out_dir
        self.options = dict(options)

    def run(self):
        try:
            self.progress.emit("loading pose")
            sessions, bodyparts = self._load()
            if not sessions:
                self.failed.emit(
                    f"No pose files under {self.pose_dir!r}.\n\n"
                    f"The v2 pipeline starts from continuous per-frame "
                    f"keypoints, so DeepLabCut must run first:\n"
                    f"    python setup_dlc_training.py --analyze"
                )
                return

            from representation import run_registry
            from representation.pipeline import run as run_pipeline

            result = run_pipeline(
                sessions, bodyparts=bodyparts,
                progress=lambda stage: self.progress.emit(stage),
                **self.options,
            )

            run_registry.record(
                self.out_dir,
                latent_method=self.options.get("latent_method", "pca"),
                params={k: v for k, v in self.options.items()
                        if k != "progress"},
                metrics=result["metrics"],
                report=result["report"].get("latent"),
                source="gui",
            )
            self.progress.emit("done")
            self.finished_ok.emit(result)

        except Exception as exc:                      # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n"
                             f"{traceback.format_exc()}")

    def _load(self):
        from representation.pose_loader import find_pose_files, load_sessions

        paths = find_pose_files(self.pose_dir)
        if not paths:
            return [], None
        sessions, bodyparts, _ = load_sessions(paths)
        return sessions, bodyparts
