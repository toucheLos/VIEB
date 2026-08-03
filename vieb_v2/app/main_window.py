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
from pages.journeys import JourneysPage
from pages.overview import OverviewPage
from pages.states_motifs import StatesMotifsPage

# Nav label -> page class. Keys must match app.navigation.NAV_ITEMS.
_PAGE_CLASSES = {
    "Overview": OverviewPage,
    "States & Motifs": StatesMotifsPage,
    "Journeys": JourneysPage,
    "Analysis": AnalysisPage,
    "Artifacts": ArtifactsPage,
    "Cluster Runs": ClusterRunsPage,
}


class MainWindow(QMainWindow):
    """Shell window: a sidebar on the left, a stack of pages on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VIEB v2")
        self.resize(1200, 800)
        self.setMinimumSize(900, 600)

        central = QWidget()
        central.setStyleSheet(f"background:{theme.CONTENT_BG};")
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

        # A run started on Analysis should show up in Cluster Runs without the
        # user having to hunt for a Refresh button.
        analysis, runs = self.pages.get("Analysis"), self.pages.get("Cluster Runs")
        if analysis is not None and runs is not None:
            analysis.run_recorded = runs.refresh

        self.setCentralWidget(central)
        self.show_page(NAV_ITEMS[0])

    def show_page(self, name):
        """Swap the content area to `name` and highlight its nav item."""
        page = self.pages.get(name)
        if page is None:
            return
        self.stack.setCurrentWidget(page)
        self.nav.set_active(name)
