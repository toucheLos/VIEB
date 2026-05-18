# View modules for VIEB GUI
from views.overview import OverviewView
from views.pipeline import RunPipelineView
from views.dlc_setup import DLCSetupView
from views.browse_states import BrowseStatesView
from views.validation import ValidationView
from views.quantification import QuantificationView
from views.advanced import AdvancedView
from views.settings import SettingsView
from views.analysis import AnalysisView

__all__ = [
    "OverviewView",
    "RunPipelineView",
    "DLCSetupView",
    "BrowseStatesView",
    "ValidationView",
    "QuantificationView",
    "AdvancedView",
    "SettingsView",
    "AnalysisView",
]
