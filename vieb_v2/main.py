#!/usr/bin/env python3
"""Entry point for the VIEB v2 GUI shell.

    python main.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication  # noqa: E402

from app.main_window import MainWindow  # noqa: E402


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
