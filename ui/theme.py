from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

class Theme:
    # Colors
    COLOR_MAIN_BG = "#1E1E1E"
    COLOR_SIDEBAR_BG = "#252526"
    COLOR_PRIMARY = "#007ACC"
    COLOR_TEAL = "#4EC9B0"
    COLOR_ORANGE = "#CE9178"
    COLOR_TEXT_MAIN = "#D4D4D4"
    COLOR_TEXT_MUTED = "#858585"
    COLOR_BORDER = "#3E3E42"
    COLOR_HOVER = "#2A2D2E"
    COLOR_SELECTION = "#37373D"
    COLOR_RED = "#F44336"

    @staticmethod
    def apply_theme(app: QApplication):
        app.setStyle("Fusion")
        
        # We also define a global QSS for specific widgets
        qss = f"""
        QMainWindow {{
            background-color: {Theme.COLOR_MAIN_BG};
        }}
        QWidget {{
            color: {Theme.COLOR_TEXT_MAIN};
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 14px;
        }}
        /* Sidebar */
        QFrame#Sidebar {{
            background-color: {Theme.COLOR_SIDEBAR_BG};
            border-right: 1px solid {Theme.COLOR_BORDER};
        }}
        QPushButton#SidebarButton {{
            background-color: transparent;
            border: none;
            color: {Theme.COLOR_TEXT_MUTED};
            text-align: left;
            padding: 10px 15px;
            font-weight: 500;
        }}
        QPushButton#SidebarButton:hover {{
            color: {Theme.COLOR_TEXT_MAIN};
            background-color: {Theme.COLOR_HOVER};
        }}
        QPushButton#SidebarButton:checked {{
            color: {Theme.COLOR_TEXT_MAIN};
            border-left: 2px solid {Theme.COLOR_PRIMARY};
            background-color: {Theme.COLOR_HOVER};
        }}
        
        /* Dashboard Cards */
        QFrame#LibraryCard {{
            background-color: {Theme.COLOR_SIDEBAR_BG};
            border: 1px solid {Theme.COLOR_BORDER};
            border-radius: 6px;
        }}
        QFrame#LibraryCard:hover {{
            border: 1px solid {Theme.COLOR_PRIMARY};
            background-color: {Theme.COLOR_HOVER};
        }}
        QLabel#CardTitle {{
            font-size: 16px;
            font-weight: bold;
            color: {Theme.COLOR_PRIMARY};
        }}
        QLabel#CardTag {{
            background-color: {Theme.COLOR_SELECTION};
            color: {Theme.COLOR_TEAL};
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 11px;
        }}

        /* General UI Elements */
        QGroupBox {{
            border: 1px solid {Theme.COLOR_BORDER};
            margin-top: 20px;
            border-radius: 4px;
            padding-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: {Theme.COLOR_TEAL};
        }}
        QPushButton {{
            background-color: {Theme.COLOR_PRIMARY};
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            background-color: #0062A3;
        }}
        QPushButton#SecondaryButton {{
            background-color: {Theme.COLOR_SELECTION};
            color: {Theme.COLOR_TEXT_MAIN};
        }}
        QPushButton#SecondaryButton:hover {{
            background-color: {Theme.COLOR_BORDER};
        }}
        QLineEdit, QComboBox, QSpinBox {{
            background-color: #3C3C3C;
            border: 1px solid {Theme.COLOR_BORDER};
            color: {Theme.COLOR_TEXT_MAIN};
            padding: 4px;
            border-radius: 2px;
        }}
        QTableWidget {{
            background-color: {Theme.COLOR_MAIN_BG};
            gridline-color: {Theme.COLOR_BORDER};
            border: 1px solid {Theme.COLOR_BORDER};
        }}
        QHeaderView::section {{
            background-color: {Theme.COLOR_SIDEBAR_BG};
            padding: 4px;
            border: 1px solid {Theme.COLOR_BORDER};
        }}
        """
        app.setStyleSheet(qss)
