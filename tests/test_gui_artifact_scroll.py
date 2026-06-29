import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_transitions_motifs_page_uses_scroll_area():
    from pathlib import Path

    source = Path("views/analysis.py").read_text(encoding="utf-8")
    start = source.index("def _build_tab2")
    end = source.index("def _load_tab2")
    tab2 = source[start:end]

    assert "QScrollArea()" in tab2
    assert "scroll.setWidgetResizable(True)" in tab2
    assert "_scroll_content_widget()" in tab2
    assert "setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)" in tab2
    assert "setMaximumHeight(250)" not in tab2


def test_binary_artifacts_are_not_text_previewed():
    from views.artifacts import binary_preview_disabled

    assert binary_preview_disabled("Video", "clips/state_0/clip_001.mp4")
    assert binary_preview_disabled("Model", "shared/clusterer.pkl")
    assert binary_preview_disabled("HDF5", "pose/output.h5")
    assert not binary_preview_disabled("CSV", "comparison/summary_table.csv")
