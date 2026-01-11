from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

class Theme:
    # Current State
    is_dark = True
    
    # Palette - Dark
    DARK = {
        "BG": "#1E1E1E", "SIDEBAR": "#252526", "TEXT": "#D4D4D4", 
        "MUTED": "#858585", "BORDER": "#3E3E42", "HOVER": "#2A2D2E",
        "SELECTION": "#37373D", "PRIMARY": "#007ACC", "TEAL": "#4EC9B0", "ORANGE": "#CE9178"
    }

    # Palette - Light
    LIGHT = {
        "BG": "#FFFFFF", "SIDEBAR": "#F3F3F3", "TEXT": "#333333", 
        "MUTED": "#666666", "BORDER": "#E5E5E5", "HOVER": "#E8E8E8",
        "SELECTION": "#D0E8F5", "PRIMARY": "#007ACC", "TEAL": "#008080", "ORANGE": "#A0522D"
    }
    
    # Accessors for current theme (backward compatibility fallback)
    COLOR_MAIN_BG = DARK["BG"]
    COLOR_SIDEBAR_BG = DARK["SIDEBAR"]
    COLOR_PRIMARY = DARK["PRIMARY"]
    COLOR_TEAL = DARK["TEAL"]
    COLOR_ORANGE = DARK["ORANGE"]
    COLOR_TEXT_MAIN = DARK["TEXT"]
    COLOR_TEXT_MUTED = DARK["MUTED"]
    COLOR_BORDER = DARK["BORDER"]
    COLOR_HOVER = DARK["HOVER"]
    COLOR_SELECTION = DARK["SELECTION"]
    COLOR_RED = "#F44336"

    @staticmethod
    def toggle_theme(app: QApplication):
        Theme.is_dark = not Theme.is_dark
        Theme.apply_theme(app)

    @staticmethod
    def apply_theme(app: QApplication):
        app.setStyle("Fusion")
        
        c = Theme.DARK if Theme.is_dark else Theme.LIGHT
        
        # Update static vars for other classes (simple hack, though re-instantiation usually required for full effect)
        Theme.COLOR_MAIN_BG = c["BG"]
        Theme.COLOR_SIDEBAR_BG = c["SIDEBAR"]
        Theme.COLOR_TEXT_MAIN = c["TEXT"]
        Theme.COLOR_TEXT_MUTED = c["MUTED"]
        Theme.COLOR_BORDER = c["BORDER"]
        
        qss = f"""
        QMainWindow, QDialog {{
            background-color: {c["BG"]};
        }}
        QWidget {{
            color: {c["TEXT"]};
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 14px;
        }}
        /* Sidebar */
        QFrame#Sidebar {{
            background-color: {c["SIDEBAR"]};
            border-right: 1px solid {c["BORDER"]};
        }}
        QPushButton#SidebarButton {{
            background-color: transparent;
            border: none;
            color: {c["MUTED"]};
            text-align: left;
            padding: 10px 15px;
            font-weight: 500;
        }}
        QPushButton#SidebarButton:hover {{
            color: {c["TEXT"]};
            background-color: {c["HOVER"]};
        }}
        QPushButton#SidebarButton:checked {{
            color: {c["TEXT"]};
            border-left: 2px solid {c["PRIMARY"]};
            background-color: {c["HOVER"]};
        }}
        
        /* Dashboard Cards */
        QFrame#LibraryCard {{
            background-color: {c["SIDEBAR"]};
            border: 1px solid {c["BORDER"]};
            border-radius: 6px;
        }}
        QFrame#LibraryCard:hover {{
            border: 1px solid {c["PRIMARY"]};
            background-color: {c["HOVER"]};
        }}
        QLabel#CardTitle {{
            font-size: 16px;
            font-weight: bold;
            color: {c["PRIMARY"]};
        }}
        QLabel#CardTag {{
            background-color: {c["SELECTION"]};
            color: {c["TEAL"]};
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 11px;
        }}

        /* General UI Elements */
        QGroupBox {{
            border: 1px solid {c["BORDER"]};
            margin-top: 20px;
            border-radius: 4px;
            padding-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
            color: {c["TEAL"]};
        }}
        QPushButton {{
            background-color: {c["PRIMARY"]};
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            background-color: #0062A3;
        }}
        QPushButton#SecondaryButton {{
            background-color: {c["SELECTION"]};
            color: {c["TEXT"]};
        }}
        QPushButton#SecondaryButton:hover {{
            background-color: {c["BORDER"]};
        }}
        
        /* Input Fields */
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
            background-color: {c["BG"]};
            border: 1px solid {c["BORDER"]};
            color: {c["TEXT"]};
            padding: 4px;
            border-radius: 2px;
            selection-background-color: {c["SELECTION"]};
        }}
        
        /* Lists and Tables */
        QListWidget {{
            background-color: {c["BG"]};
            border: 1px solid {c["BORDER"]};
            color: {c["TEXT"]};
        }}
        QListWidget::item:selected {{
            background-color: {c["SELECTION"]};
            color: {c["TEAL"]};
        }}
        QListWidget::item:hover {{
            background-color: {c["HOVER"]};
        }}
        
        QTableWidget, QTableView {{
            background-color: {c["BG"]};
            gridline-color: {c["BORDER"]};
            border: 1px solid {c["BORDER"]};
            color: {c["TEXT"]};
            selection-background-color: {c["SELECTION"]};
            selection-color: {c["TEXT"]};
        }}
        QHeaderView::section {{
            background-color: {c["SIDEBAR"]};
            padding: 4px;
            border: 1px solid {c["BORDER"]};
            color: {c["TEXT"]};
        }}
        QHeaderView::section:horizontal {{
            border-bottom: 1px solid {c["BORDER"]};
        }}
        QHeaderView::section:vertical {{
            border-right: 1px solid {c["BORDER"]};
        }}
        
        /* Scroll Area & Splitter */
        QScrollArea {{
            background-color: {c["BG"]};
            border: none;
        }}
        QSplitter::handle {{
            background-color: {c["BORDER"]};
        }}
        
        /* Menus */
        QMenuBar {{
            background-color: {c["BG"]};
            color: {c["TEXT"]};
            border-bottom: 1px solid {c["BORDER"]};
        }}
        QMenuBar::item:selected {{
            background-color: {c["HOVER"]};
        }}
        QMenu {{
            background-color: {c["BG"]};
            border: 1px solid {c["BORDER"]};
            color: {c["TEXT"]};
        }}
        QMenu::item {{
            padding: 5px 20px;
        }}
        QMenu::item:selected {{
            background-color: {c["HOVER"]};
            color: {c["TEXT"]};
        }}
        """
        app.setStyleSheet(qss)
