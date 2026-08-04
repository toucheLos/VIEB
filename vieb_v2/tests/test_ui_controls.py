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


def test_nav_is_the_union_of_v1_and_v2_pages():
    # v1's information architecture (Overview/Pipeline/Cluster Runs/Analysis/
    # Artifacts/Settings/Help) plus v2's own States & Motifs and Journeys.
    _app()
    from app.navigation import NAV_ITEMS
    assert NAV_ITEMS == ["Overview", "Pipeline", "States & Motifs", "Journeys",
                         "Cluster Runs", "Analysis", "Artifacts", "Settings",
                         "Help"]


def test_every_nav_item_still_swaps_pages():
    _app()
    from app.main_window import MainWindow
    from app.navigation import NAV_ITEMS

    window = MainWindow()
    for name in NAV_ITEMS:
        window.nav.buttons[name].click()
        assert window.stack.currentWidget() is window.pages[name]
        checked = [n for n, b in window.nav.buttons.items() if b.isChecked()]
        assert checked == [name]


def test_analysis_page_offers_both_latent_methods():
    _app()
    from pages.analysis import AnalysisPage

    page = AnalysisPage()
    assert page.latent_method() == "pca"           # default
    page.diffusion_radio.setChecked(True)
    assert page.latent_method() == "diffusion"
    page.pca_radio.setChecked(True)
    assert page.latent_method() == "pca"


def test_parameter_fields_follow_the_selected_method():
    # Showing diffusion bandwidth controls next to a PCA run would invite
    # setting parameters that do nothing.
    _app()
    from pages.analysis import AnalysisPage

    page = AnalysisPage()
    page.pca_radio.setChecked(True)
    assert page.pca_params.isVisibleTo(page)
    assert not page.diffusion_params.isVisibleTo(page)

    page.diffusion_radio.setChecked(True)
    assert page.diffusion_params.isVisibleTo(page)
    assert not page.pca_params.isVisibleTo(page)


def test_options_match_pipeline_run_signature():
    # The page passes its options straight through as kwargs, so a rename in
    # the pipeline must not silently break the GUI.
    _app()
    import inspect

    from pages.analysis import AnalysisPage
    from representation.pipeline import run as pipeline_run

    accepted = set(inspect.signature(pipeline_run).parameters)
    assert set(AnalysisPage().options()).issubset(accepted)


def test_diffusion_defaults_to_alpha_one():
    # alpha=1 is the density-normalised default; anything lower lets sampling
    # density compress the slow behaviors.
    _app()
    from pages.analysis import AnalysisPage
    assert AnalysisPage().options()["alpha"] == 1.0


def test_cluster_runs_page_lists_recorded_runs(tmp_path):
    _app()
    from pages.cluster_runs import ClusterRunsPage
    from representation import run_registry

    out = str(tmp_path)
    run_registry.record(out, "pca", {}, metrics={"n_states": 5,
                                                 "noise_frac": 0.1})
    run_registry.record(out, "diffusion", {}, metrics={"n_states": 9,
                                                       "noise_frac": 0.2})

    page = ClusterRunsPage()
    page.out_edit.setText(out)
    assert page.refresh() == 2
    # Newest first, and the latent method is visible -- the reason the page
    # exists.
    assert page.table.item(0, 2).text() == "diffusion"
    assert page.table.item(1, 2).text() == "pca"


def test_cluster_runs_page_is_empty_without_a_registry(tmp_path):
    _app()
    from pages.cluster_runs import ClusterRunsPage

    page = ClusterRunsPage()
    page.out_edit.setText(str(tmp_path / "nothing"))
    assert page.refresh() == 0
    assert page.empty_label.isVisibleTo(page)


def test_worker_reports_missing_pose_instead_of_raising(tmp_path):
    # A worker-thread exception would vanish and leave the Run button dead.
    _app()
    from app.worker import PipelineWorker

    errors = []
    worker = PipelineWorker(str(tmp_path / "no-pose"), str(tmp_path),
                            {"latent_method": "pca"})
    worker.failed.connect(errors.append)
    worker.run()                      # run synchronously; no thread needed

    assert errors and "setup_dlc_training.py --analyze" in errors[0]


def test_worker_surfaces_backend_errors(tmp_path):
    _app()
    from app.worker import PipelineWorker

    errors = []
    worker = PipelineWorker(str(tmp_path), str(tmp_path),
                            {"latent_method": "nonsense"})
    worker.failed.connect(errors.append)
    worker.run()
    assert errors            # reported, not raised
