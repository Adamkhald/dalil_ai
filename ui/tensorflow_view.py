from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QFrame, QStackedWidget, QFileDialog, 
                               QComboBox, QSpinBox, QMessageBox, QTextEdit, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal
from dalil_ai.ui.theme import Theme

try:
    from dalil_ai.core.tensorflow_logic import TensorFlowPipeline
except ImportError:
    TensorFlowPipeline = None

# Worker Thread for TF Training to avoid freezing UI
class TFTrainerThread(QThread):
    progress = Signal(str)
    finished = Signal()
    
    def __init__(self, pipeline, epochs):
        super().__init__()
        self.pipeline = pipeline
        self.epochs = epochs
        
    def run(self):
        # We pass a lambda/function that emits signals to the pipeline's callback mechanism
        def update_ui(msg):
            self.progress.emit(msg)
            
        self.pipeline.train(epochs=self.epochs, update_callback=update_ui)
        self.finished.emit()

class TensorflowView(QWidget):
    def __init__(self):
        super().__init__()
        self.pipeline = TensorFlowPipeline() if TensorFlowPipeline else None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        if not self.pipeline:
             lbl = QLabel("TensorFlow not found. Please install tensorflow.")
             lbl.setAlignment(Qt.AlignCenter)
             main_layout.addWidget(lbl)
             return

        # Sidebar
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet(f"background-color: {Theme.COLOR_SIDEBAR_BG}; border-right: 1px solid {Theme.COLOR_BORDER};")
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(0, 10, 0, 0)
        
        self.steps = ["1. Dataset", "2. Architecture", "3. Train", "4. Export"]
        self.step_buttons = []
        
        for i, step in enumerate(self.steps):
            btn = QPushButton(step)
            btn.setObjectName("SidebarButton")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self.switch_step(idx))
            self.step_buttons.append(btn)
            side_layout.addWidget(btn)
            
        side_layout.addStretch()
        main_layout.addWidget(self.sidebar)
        
        # Content
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)
        
        self.create_data_page()
        self.create_arch_page()
        self.create_train_page()
        self.create_export_page()
        
        self.switch_step(0)

    def switch_step(self, index):
        for i, btn in enumerate(self.step_buttons):
            btn.setChecked(i == index)
        self.stack.setCurrentIndex(index)

    def create_data_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 1: TF Load Data")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_ORANGE}; font-weight: bold;")
        layout.addWidget(lbl)
        
        layout.addWidget(QLabel("Select an Image Directory (Grid/Folder format)."))
        
        btn = QPushButton("Select Folder")
        btn.clicked.connect(self.load_data)
        layout.addWidget(btn)
        
        self.lbl_data_status = QLabel("No Data.")
        layout.addWidget(self.lbl_data_status)
        layout.addStretch()
        self.stack.addWidget(page)
        
    def create_arch_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 2: Model Architecture")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_ORANGE}; font-weight: bold;")
        layout.addWidget(lbl)
        
        layout.addWidget(QLabel("Choose Transfer Learning Base:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["mobilenet_v2", "resnet50", "Custom CNN"])
        layout.addWidget(self.combo_model)
        
        btn = QPushButton("Build Model")
        btn.clicked.connect(self.build_model)
        layout.addWidget(btn)
        
        self.lbl_model_status = QLabel("")
        layout.addWidget(self.lbl_model_status)
        layout.addStretch()
        self.stack.addWidget(page)

    def create_train_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        lbl = QLabel("Step 3: Train")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_ORANGE}; font-weight: bold;")
        layout.addWidget(lbl)
        
        layout.addWidget(QLabel("Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setValue(5)
        layout.addWidget(self.spin_epochs)
        
        btn = QPushButton("Start Training")
        btn.clicked.connect(self.start_training)
        layout.addWidget(btn)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background-color: #222; color: #ddd;")
        layout.addWidget(self.log_area)
        
        layout.addStretch()
        self.stack.addWidget(page)

    def create_export_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        lbl = QLabel("Step 4: Export (TFLite)")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_ORANGE}; font-weight: bold;")
        layout.addWidget(lbl)
        
        btn = QPushButton("Convert & Save .tflite")
        btn.clicked.connect(self.export_model)
        layout.addWidget(btn)
        
        layout.addStretch()
        self.stack.addWidget(page)

    # --- Logic ---
    def load_data(self):
        d = QFileDialog.getExistingDirectory(self, "Select Data")
        if d:
            try:
                msg = self.pipeline.load_data(d)
                self.lbl_data_status.setText(msg)
                self.switch_step(1)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def build_model(self):
        m = self.combo_model.currentText()
        try:
            msg = self.pipeline.build_model(m)
            self.lbl_model_status.setText(msg)
            self.switch_step(2)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def start_training(self):
        epochs = self.spin_epochs.value()
        self.log_area.append("Starting training...")
        self.thread = TFTrainerThread(self.pipeline, epochs)
        self.thread.progress.connect(self.log_area.append)
        self.thread.finished.connect(lambda: self.switch_step(3))
        self.thread.start()

    def export_model(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save TFLite", "model.tflite", "TFLite (*.tflite)")
        if path:
            try:
                msg = self.pipeline.export_tflite(path)
                QMessageBox.information(self, "Success", msg)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
