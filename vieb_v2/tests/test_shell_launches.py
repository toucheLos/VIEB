import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Module-level reference: Qt aborts if the QApplication is garbage collected.
_QAPP = None


def _app():
    global _QAPP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    if _QAPP is None:
        _QAPP = QApplication.instance() or QApplication([])
    return _QAPP


def test_shell_launches():
    _app()
    from app.main_window import MainWindow
    from app.navigation import NAV_ITEMS

    window = MainWindow()

    assert window.windowTitle() == "VIEB v2"
    assert list(window.pages) == NAV_ITEMS
    assert window.stack.count() == len(NAV_ITEMS)

    # Overview is the landing page and starts highlighted.
    assert window.stack.currentWidget() is window.pages["Overview"]
    assert window.nav.buttons["Overview"].isChecked()


def test_nav_switches_pages_and_highlights():
    _app()
    from app.main_window import MainWindow
    from app.navigation import NAV_ITEMS

    window = MainWindow()

    for name in NAV_ITEMS:
        window.nav.buttons[name].click()

        assert window.stack.currentWidget() is window.pages[name]

        checked = [n for n, b in window.nav.buttons.items() if b.isChecked()]
        assert checked == [name]
