import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_QAPP = None


def _app():
    global _QAPP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    if _QAPP is None:
        _QAPP = QApplication.instance() or QApplication([])
    return _QAPP


# ------------------------------------------------------------------ sidebar

def test_sidebar_collapses_to_icons_and_back():
    _app()
    from app import theme
    from app.navigation import Navigation

    nav = Navigation()
    assert nav.width() == theme.SIDEBAR_WIDTH
    label = nav.buttons["Overview"].text()

    nav.set_collapsed(True)
    assert nav.width() == theme.SIDEBAR_COLLAPSED_WIDTH
    assert nav.buttons["Overview"].text() != label
    assert nav.buttons["Overview"].toolTip() == "Overview"

    nav.set_collapsed(False)
    assert nav.width() == theme.SIDEBAR_WIDTH
    assert nav.buttons["Overview"].text() == label


def test_collapsing_preserves_the_active_page():
    _app()
    from app.navigation import Navigation

    nav = Navigation()
    nav.set_active("Analysis")
    nav.set_collapsed(True)
    assert nav.active() == "Analysis"
    nav.set_collapsed(False)
    assert nav.active() == "Analysis"


def test_collapsed_buttons_keep_an_accessible_name():
    # The visible text becomes a glyph; the destination must still be
    # announced rather than living only in a hover tooltip.
    _app()
    from app.navigation import Navigation

    nav = Navigation()
    nav.set_collapsed(True)
    assert nav.buttons["Cluster Runs"].accessibleName() == "Cluster Runs"


def test_every_nav_item_has_an_icon():
    from app.nav_button import NAV_ICONS
    from app.navigation import NAV_ITEMS
    for name in NAV_ITEMS:
        assert name in NAV_ICONS, name


def test_set_active_is_exclusive():
    _app()
    from app.navigation import Navigation

    nav = Navigation()
    nav.set_active("Help")
    checked = [n for n, b in nav.buttons.items() if b.isChecked()]
    assert checked == ["Help"]


# ----------------------------------------------------------------- pipeline

def test_pipeline_marks_stages_done_from_checkpoints(tmp_path):
    # Status is read off the filesystem so CLI and batch runs show up too.
    _app()
    import numpy as np
    from pages.pipeline import PipelinePage

    page = PipelinePage()
    page.set_out_dir(str(tmp_path))
    assert page.refresh() == 0
    assert page.rows["align"].state == "pending"

    np.savez(os.path.join(str(tmp_path), "aligned.npz"), x=np.zeros(4))
    np.savez(os.path.join(str(tmp_path), "scores.npz"), x=np.zeros(4))
    assert page.refresh() == 2
    assert page.rows["align"].state == "done"
    assert page.rows["latent"].state == "done"
    assert page.rows["embed"].state == "pending"


def test_pipeline_has_v2_stages_not_v1_stages():
    _app()
    from pages.pipeline import STAGES
    assert [k for k, *_ in STAGES] == ["align", "latent", "embed", "cluster"]


def test_pipeline_run_request_logs_the_cli_equivalent():
    _app()
    from pages.pipeline import PipelinePage

    page = PipelinePage()
    page.rows["cluster"]._run_button.click()
    assert "cli cluster" in page.terminal.toPlainText()


# ----------------------------------------------------------------- settings

def test_settings_round_trip(tmp_path, monkeypatch):
    _app()
    from pages import settings as settings_mod

    path = str(tmp_path / "cfg.json")
    monkeypatch.setattr(settings_mod, "config_path", lambda: path)

    page = settings_mod.SettingsPage()
    page.min_cluster_size.setValue(321)
    page.alpha.setValue(0.5)
    page.save()

    with open(path) as fh:
        stored = json.load(fh)
    assert stored["min_cluster_size"] == 321
    assert stored["alpha"] == 0.5
    assert settings_mod.load_settings()["min_cluster_size"] == 321


def test_settings_fall_back_to_defaults_when_keys_missing(tmp_path, monkeypatch):
    # A config written by an older version must not drop newer keys.
    _app()
    from pages import settings as settings_mod

    path = str(tmp_path / "cfg.json")
    monkeypatch.setattr(settings_mod, "config_path", lambda: path)
    with open(path, "w") as fh:
        json.dump({"min_cluster_size": 77}, fh)

    values = settings_mod.load_settings()
    assert values["min_cluster_size"] == 77
    assert values["alpha"] == settings_mod.DEFAULTS["alpha"]


def test_corrupt_settings_do_not_crash(tmp_path, monkeypatch):
    _app()
    from pages import settings as settings_mod

    path = str(tmp_path / "cfg.json")
    monkeypatch.setattr(settings_mod, "config_path", lambda: path)
    with open(path, "w") as fh:
        fh.write("{ not json")
    assert settings_mod.load_settings() == settings_mod.DEFAULTS


# ----------------------------------------------------------------- overview

def test_overview_summarises_the_latest_run(tmp_path):
    _app()
    from pages.overview import OverviewPage
    from representation import run_registry

    out = str(tmp_path)
    run_registry.record(out, "pca", {}, metrics={
        "n_states": 5, "noise_frac": 0.10,
        "clustered_only": {"state_entropy": 0.81}})
    run_registry.record(out, "diffusion", {}, metrics={
        "n_states": 9, "noise_frac": 0.22,
        "clustered_only": {"state_entropy": 0.93}})

    page = OverviewPage()
    page.out_edit.setText(out)
    assert page.refresh() == 2
    assert page.cards["latent"].value() == "diffusion"   # newest
    assert page.cards["states"].value() == "9"
    assert page.cards["noise"].value() == "22.0%"


def test_overview_is_calm_with_no_runs(tmp_path):
    _app()
    from pages.overview import OverviewPage

    page = OverviewPage()
    page.out_edit.setText(str(tmp_path / "nothing"))
    assert page.refresh() == 0
    assert page.cards["states"].value() == "-"


# ---------------------------------------------------------------- artifacts

def test_artifacts_lists_what_is_on_disk(tmp_path):
    _app()
    import numpy as np
    from pages.artifacts import ArtifactsPage

    np.savez(os.path.join(str(tmp_path), "aligned.npz"), x=np.zeros(8))
    with open(os.path.join(str(tmp_path), "runs.json"), "w") as fh:
        fh.write("[]")

    page = ArtifactsPage()
    page.out_edit.setText(str(tmp_path))
    assert page.refresh() == 2
    names = {page.table.item(r, 0).text() for r in range(page.table.rowCount())}
    assert names == {"aligned.npz", "runs.json"}
    kinds = {page.table.item(r, 1).text() for r in range(page.table.rowCount())}
    assert "Aligned pose" in kinds


def test_artifacts_empty_state(tmp_path):
    _app()
    from pages.artifacts import ArtifactsPage

    page = ArtifactsPage()
    page.out_edit.setText(str(tmp_path / "nothing"))
    assert page.refresh() == 0
    assert page.empty_label.isVisibleTo(page)


# --------------------------------------------------------------------- help

def test_help_documents_both_metric_conventions():
    # The two conventions are easy to misread, so Help must state both.
    _app()
    from pages.help import SECTIONS

    text = " ".join(body for _title, body in SECTIONS)
    assert "v1 convention" in text and "clustered only" in text
    assert "noise/clustered speed ratio" in text
