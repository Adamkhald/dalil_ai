from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                               QFrame, QPushButton, QLabel, QStackedWidget, QMessageBox, QApplication)
from PySide6.QtCore import Qt
from dalil_ai.ui.theme import Theme
from dalil_ai.ui.dashboard import Dashboard
from dalil_ai.ui.dashboard import Dashboard

# Safe Imports for Modules
try:
    from dalil_ai.ui.sklearn_view import SklearnView
except ImportError:
    SklearnView = None

try:
    from dalil_ai.ui.pytorch_view import PyTorchView
except ImportError:
    PyTorchView = None

try:
    from dalil_ai.ui.tensorflow_view import TensorflowView
except ImportError:
    TensorflowView = None

try:
    from dalil_ai.ui.mediapipe_view import MediaPipeView
except ImportError:
    MediaPipeView = None

try:
    from dalil_ai.ui.rl_studio_view import RLStudioView
except ImportError:
    RLStudioView = None

class PlaceholderView(QWidget):
    def __init__(self, module_name):
        super().__init__()
        layout = QVBoxLayout(self)
        lbl = QLabel(f"⚠️ {module_name} Module Not Installed")
        lbl.setStyleSheet(f"font-size: 20px; color: {Theme.COLOR_TEXT_MUTED}; font-weight: bold;")
        lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl)
        
        lbl_sub = QLabel("This module was excluded from the build to save space.")
        lbl_sub.setStyleSheet(f"font-size: 14px; color: {Theme.COLOR_TEXT_MUTED};")
        lbl_sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_sub)


from PySide6.QtGui import QIcon
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dalil AI - Deep Learning & RL Suite")
        
        # Robust Icon Loading
        # Assuming main_window.py is in /ui/ and icon is in / (root)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        icon_path = os.path.join(project_root, "Dalil_ai.ico")
        
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
        self.resize(1280, 800)
        self.init_ui()

    def init_ui(self):
        # Apply Theme
        Theme.apply_theme(QApplication.instance())

        # Main Layout container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Sidebar ---
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(250)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Branding
        brand_box = QFrame()
        brand_box.setFixedHeight(60)
        brand_box.setStyleSheet(f"background-color: {Theme.COLOR_MAIN_BG}; border-bottom: 1px solid {Theme.COLOR_BORDER};")
        brand_layout = QHBoxLayout(brand_box)
        lbl_brand = QLabel("🧠 Dalil AI")
        lbl_brand.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {Theme.COLOR_TEXT_MAIN};")
        brand_layout.addWidget(lbl_brand)
        sidebar_layout.addWidget(brand_box)

        # Nav Buttons
        self.nav_buttons = []
        nav_items = [("Home", "🏠"), ("Scikit-Learn", "🔬"), ("PyTorch", "🔥"), 
                     ("TensorFlow", "🧠"), ("MediaPipe", "📸"), ("RL Studio", "🎮")]
        
        for name, icon in nav_items:
            btn = QPushButton(f"{icon}  {name}")
            btn.setObjectName("SidebarButton")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, n=name: self.navigate_to(n))
            self.nav_buttons.append(btn)
            sidebar_layout.addWidget(btn)
        
        sidebar_layout.addStretch()
        
        # Bottom Settings
        btn_settings = QPushButton("⚙  Settings")
        btn_settings.setObjectName("SidebarButton")
        sidebar_layout.addWidget(btn_settings)

        main_layout.addWidget(self.sidebar)

        # --- Content Area ---
        content_layout = QVBoxLayout()
        content_layout.setSpacing(0)
        content_layout.setContentsMargins(0,0,0,0)
        
        # Custom Title Header (inside content to look integrated)
        self.header = QFrame()
        self.header.setFixedHeight(40)
        self.header.setStyleSheet(f"background-color: {Theme.COLOR_MAIN_BG}; border-bottom: 1px solid {Theme.COLOR_BORDER};")
        header_layout = QHBoxLayout(self.header)
        self.lbl_page_title = QLabel("Dashboard")
        self.lbl_page_title.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.lbl_page_title)
        header_layout.addStretch()
        content_layout.addWidget(self.header)
        
        # View Stack
        self.stack = QStackedWidget()
        content_layout.addWidget(self.stack)
        
        main_layout.addLayout(content_layout)

        # Initialize Views
        self.view_dashboard = Dashboard()
        self.view_dashboard.library_selected.connect(self.navigate_to)
        
        self.view_sklearn = SklearnView() if SklearnView else PlaceholderView("Scikit-Learn")
        self.view_pytorch = PyTorchView() if PyTorchView else PlaceholderView("PyTorch")
        self.view_tensorflow = TensorflowView() if TensorflowView else PlaceholderView("TensorFlow")
        self.view_mediapipe = MediaPipeView() if MediaPipeView else PlaceholderView("MediaPipe")
        self.view_rl = RLStudioView() if RLStudioView else PlaceholderView("RL Studio")
        
        # Add to stack
        self.stack.addWidget(self.view_dashboard) # Index 0
        self.stack.addWidget(self.view_sklearn)   # Index 1
        self.stack.addWidget(self.view_pytorch)   # Index 2
        self.stack.addWidget(self.view_tensorflow)# Index 3
        self.stack.addWidget(self.view_mediapipe) # Index 4
        self.stack.addWidget(self.view_rl)        # Index 5

        # Map names to indices
        self.view_map = {
            "Home": 0,
            "Scikit-Learn": 1,
            "PyTorch": 2,
            "TensorFlow": 3,
            "MediaPipe": 4,
            "RL Studio": 5
        }

        # Set default
        self.nav_buttons[0].setChecked(True)

    def navigate_to(self, page_name):
        self.lbl_page_title.setText(page_name)
        
        # Update Sidebar State
        for btn in self.nav_buttons:
            if page_name in btn.text():
                btn.setChecked(True)
            else:
                btn.setChecked(False)

        # Switch Stack
        if page_name in self.view_map:
            self.stack.setCurrentIndex(self.view_map[page_name])
        else:
            # Fallback (shouldn't happen with updated button connection logic, but good for safety)
            # Check if name is in any key
            found = False
            for key, idx in self.view_map.items():
                if key in page_name:
                    self.stack.setCurrentIndex(idx)
                    found = True
                    break
            if not found:
                 QMessageBox.information(self, "Info", f"Navigating to {page_name}")
