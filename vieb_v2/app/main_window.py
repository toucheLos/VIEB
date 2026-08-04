"""Top-level application window for VIEB v2."""

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from app import theme
from app.navigation import NAV_ITEMS, Navigation
from pages.analysis import AnalysisPage
from pages.artifacts import ArtifactsPage
from pages.cluster_runs import ClusterRunsPage
from pages.help import HelpPage
from pages.journeys import JourneysPage
from pages.overview import OverviewPage
from pages.pipeline import PipelinePage
from pages.settings import SettingsPage
from pages.states_motifs import StatesMotifsPage

# Nav label -> page class. Keys must match app.navigation.NAV_ITEMS.
_PAGE_CLASSES = {
    "Overview": OverviewPage,
    "Pipeline": PipelinePage,
    "States & Motifs": StatesMotifsPage,
    "Journeys": JourneysPage,
    "Cluster Runs": ClusterRunsPage,
    "Analysis": AnalysisPage,
    "Artifacts": ArtifactsPage,
    "Settings": SettingsPage,
    "Help": HelpPage,
}


class MainWindow(QMainWindow):
    """Shell window: a sidebar on the left, a stack of pages on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VIEB v2")
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        central = QWidget()
        central.setStyleSheet(theme.page_background())
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.nav = Navigation()
        self.nav.page_selected.connect(self.show_page)
        layout.addWidget(self.nav)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, stretch=1)

        self.pages = {}
        for name in NAV_ITEMS:
            page = _PAGE_CLASSES[name]()
            self.pages[name] = page
            self.stack.addWidget(page)

        self._connect_pages()

        self.setCentralWidget(central)
        self.show_page(NAV_ITEMS[0])

    def _connect_pages(self):
        """Wire cross-page updates so nothing needs a manual Refresh."""
        analysis = self.pages.get("Analysis")
        settings = self.pages.get("Settings")

        # A finished run should appear in the history and summary immediately.
        if analysis is not None:
            analysis.run_recorded = self._on_run_recorded

        # Saved settings become the starting values for the next run.
        if settings is not None and analysis is not None:
            original_save = settings.save

            def save_and_apply():
                path = original_save()
                self._apply_settings(settings.values())
                return path

            settings.save = save_and_apply
            self._apply_settings(settings.values())

        out_dir = self._current_out_dir()
        if out_dir:
            self.nav.set_output_label(out_dir)

    def _apply_settings(self, values):
        """Push saved defaults into the pages that consume them."""
        analysis = self.pages.get("Analysis")
        if analysis is not None:
            analysis.pose_edit.setText(values["pose_dir"])
            analysis.out_edit.setText(values["out_dir"])
            analysis.var_threshold.setValue(values["var_threshold"])
            analysis.n_components.setValue(values["n_components"])
            analysis.alpha.setValue(values["alpha"])
            analysis.n_lags.setValue(values["n_lags"])
            analysis.lag_stride.setValue(values["lag_stride"])
            analysis.min_cluster_size.setValue(values["min_cluster_size"])

        for name in ("Cluster Runs", "Overview", "Artifacts"):
            page = self.pages.get(name)
            if page is not None and hasattr(page, "out_edit"):
                page.out_edit.setText(values["out_dir"])

        pipeline = self.pages.get("Pipeline")
        if pipeline is not None:
            pipeline.set_out_dir(values["out_dir"])

        self.nav.set_output_label(values["out_dir"])
        self._refresh_views()

    def _current_out_dir(self):
        analysis = self.pages.get("Analysis")
        return analysis.out_edit.text().strip() if analysis else None

    def _on_run_recorded(self):
        self._refresh_views()
        overview = self.pages.get("Overview")
        if overview is not None:
            runs = overview.refresh()
            self.nav.set_footer(f"{runs} run(s) recorded")

    def _refresh_views(self):
        for name in ("Cluster Runs", "Overview", "Artifacts", "Pipeline"):
            page = self.pages.get(name)
            if page is not None and hasattr(page, "refresh"):
                page.refresh()

    def show_page(self, name):
        """Swap the content area to `name` and highlight its nav item."""
        page = self.pages.get(name)
        if page is None:
            return
        self.stack.setCurrentWidget(page)
        self.nav.set_active(name)
        # Pages that read the filesystem should be current when shown, not
        # whatever they held when the window was built.
        if name in ("Overview", "Artifacts", "Cluster Runs", "Pipeline"):
            if hasattr(page, "refresh"):
                page.refresh()
