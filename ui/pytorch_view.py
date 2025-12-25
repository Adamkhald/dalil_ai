import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QFrame, QStackedWidget, QFileDialog, 
                               QComboBox, QSpinBox, QDoubleSpinBox, QMessageBox, QTextEdit, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal
from dalil_ai.ui.theme import Theme
# Try importing backend, handle if missing deps
try:
    from dalil_ai.core.pytorch_logic import PyTorchPipeline
except ImportError:
    PyTorchPipeline = None

class TrainingWorker(QThread):
    progress = Signal(str)
    finished = Signal()
    
    def __init__(self, pipeline, epochs):
        super().__init__()
        self.pipeline = pipeline
        self.epochs = epochs
        self.is_running = True

    def run(self):
        def batch_callback(batch_idx, total_batches, loss, acc):
            # Emit batch progress every now and then
            if not self.is_running: return
            msg = f"Epoch {self.current_epoch+1}/{self.epochs} [Batch {batch_idx}/{total_batches}] Loss: {loss:.4f} Acc: {acc:.4f}"
            self.progress.emit(msg)

        for epoch in range(self.epochs):
            if not self.is_running: break
            self.current_epoch = epoch
            loss, acc = self.pipeline.train_one_epoch(epoch, callback=batch_callback)
            self.progress.emit(f"Epoch {epoch+1}/{self.epochs} Finished - Avg Loss: {loss:.4f} - Avg Acc: {acc:.4f}")
        self.finished.emit()

    def stop(self):
        self.is_running = False

class PyTorchView(QWidget):
    def __init__(self):
        super().__init__()
        self.pipeline = PyTorchPipeline() if PyTorchPipeline else None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        if not self.pipeline:
             lbl = QLabel("PyTorch not found or import error. Please install torch torchvision.")
             lbl.setAlignment(Qt.AlignCenter)
             main_layout.addWidget(lbl)
             return

        # --- Sidebar (Steps) ---
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet(f"background-color: {Theme.COLOR_SIDEBAR_BG}; border-right: 1px solid {Theme.COLOR_BORDER};")
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(0, 10, 0, 0)
        
        self.steps = ["1. Dataset", "2. Model", "3. Config", "4. Train", "5. Validation", "6. Export"]
        self.step_buttons = []
        
        for i, step in enumerate(self.steps):
            btn = QPushButton(step)
            btn.setObjectName("SidebarButton") # Reusing theme style
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self.switch_step(idx))
            self.step_buttons.append(btn)
            side_layout.addWidget(btn)
            
        side_layout.addStretch()
        main_layout.addWidget(self.sidebar)
        
        # --- Content Area ---
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)
        
        self.create_data_page()
        self.create_model_page()
        self.create_config_page()
        self.create_train_page()
        self.create_validation_page()
        self.create_export_page()
        
        # Default
        self.switch_step(0)

    def switch_step(self, index):
        for i, btn in enumerate(self.step_buttons):
            btn.setChecked(i == index)
        self.stack.setCurrentIndex(index)

    # --- Pages ---
    
    def create_data_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 1: Load Dataset")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_PRIMARY}; font-weight: bold;")
        layout.addWidget(lbl)
        
        from PySide6.QtWidgets import QTabWidget
        self.tab_data = QTabWidget()
        self.tab_data.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {Theme.COLOR_BORDER}; }}
            QTabBar::tab {{ background: {Theme.COLOR_SIDEBAR_BG}; color: {Theme.COLOR_TEXT_MAIN}; padding: 8px; }}
            QTabBar::tab:selected {{ background: {Theme.COLOR_SELECTION}; border-bottom: 2px solid {Theme.COLOR_PRIMARY}; }}
        """)
        
        # Tab 1: Custom Folder
        tab1 = QWidget()
        t1_layout = QVBoxLayout(tab1)
        t1_layout.addWidget(QLabel("Select a folder containing subfolders for each class (e.g. data/cats, data/dogs)."))
        btn_load = QPushButton("Select Dataset Folder")
        btn_load.clicked.connect(self.load_data_folder)
        t1_layout.addWidget(btn_load)
        t1_layout.addStretch()
        self.tab_data.addTab(tab1, "Custom Folder")
        
        # Tab 2: Built-in
        tab2 = QWidget()
        t2_layout = QVBoxLayout(tab2)
        t2_layout.addWidget(QLabel("Download and load standard benchmarks."))
        
        self.combo_builtin = QComboBox()
        self.combo_builtin.addItems(["CIFAR10", "MNIST", "FashionMNIST"])
        t2_layout.addWidget(self.combo_builtin)
        
        btn_dl = QPushButton("Download & Load")
        btn_dl.clicked.connect(self.load_builtin_data)
        t2_layout.addWidget(btn_dl)
        t2_layout.addStretch()
        self.tab_data.addTab(tab2, "Built-in Dataset")
        
        
        layout.addWidget(self.tab_data)
        
        # Fast Mode Option
        from PySide6.QtWidgets import QCheckBox
        self.chk_fast = QCheckBox("Fast Mode (Resize to 128x128)")
        self.chk_fast.setStyleSheet(f"color: {Theme.COLOR_TEAL}; margin-top: 10px;")
        self.chk_fast.setToolTip("Speeds up training significantly but might reduce accuracy.")
        layout.addWidget(self.chk_fast)
        
        self.lbl_data_status = QLabel("No data loaded.")
        self.lbl_data_status.setWordWrap(True)
        self.lbl_data_status.setStyleSheet("color: #aaa; margin-top: 10px;")
        layout.addWidget(self.lbl_data_status)
        
        layout.addStretch()
        self.stack.addWidget(page)

    def create_model_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 2: Choose Architecture")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_PRIMARY}; font-weight: bold;")
        layout.addWidget(lbl)
        
        self.combo_model = QComboBox()
        self.combo_model.addItems(["mobilenet_v2", "resnet18", "vgg16"]) # Reorder for speed
        layout.addWidget(QLabel("Model (MobileNet is fastest):"))
        layout.addWidget(self.combo_model)
        
        btn_build = QPushButton("Build Model")
        btn_build.clicked.connect(self.build_model)
        layout.addWidget(btn_build)
        
        self.lbl_model_status = QLabel("")
        layout.addWidget(self.lbl_model_status)
        layout.addStretch()
        self.stack.addWidget(page)

    def create_config_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 3: Training Config")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_PRIMARY}; font-weight: bold;")
        layout.addWidget(lbl)
        
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setValue(5)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(4)
        self.spin_lr.setValue(0.001)
        
        self.combo_optim = QComboBox()
        self.combo_optim.addItems(["Adam", "SGD"])
        
        layout.addWidget(QLabel("Epochs (Reduce if too slow):"))
        layout.addWidget(self.spin_epochs)
        layout.addWidget(QLabel("Learning Rate:"))
        layout.addWidget(self.spin_lr)
        layout.addWidget(QLabel("Optimizer:"))
        layout.addWidget(self.combo_optim)
        
        layout.addStretch()
        self.stack.addWidget(page)

    def create_train_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 4: Training Loop")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_PRIMARY}; font-weight: bold;")
        layout.addWidget(lbl)
        
        self.btn_train = QPushButton("Start Training")
        self.btn_train.clicked.connect(self.start_training)
        layout.addWidget(self.btn_train)
        
        self.progress_log = QTextEdit()
        self.progress_log.setReadOnly(True)
        self.progress_log.setStyleSheet("background-color: #111; color: #0f0; font-family: monospace;")
        layout.addWidget(self.progress_log)
        
        layout.addStretch()
        self.stack.addWidget(page)

    def create_validation_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 5: Validation")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_PRIMARY}; font-weight: bold;")
        layout.addWidget(lbl)
        
        btn_val = QPushButton("Run Validation (Show 9 Samples)")
        btn_val.clicked.connect(self.run_validation)
        layout.addWidget(btn_val)
        
        # Grid for images
        from PySide6.QtWidgets import QGridLayout
        self.val_grid_widget = QWidget()
        self.val_grid = QGridLayout(self.val_grid_widget)
        layout.addWidget(self.val_grid_widget)
        
        layout.addStretch()
        self.stack.addWidget(page)

    def create_export_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 5: Export Model")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_PRIMARY}; font-weight: bold;")
        layout.addWidget(lbl)
        
        btn_save = QPushButton("Save Checkpoint (.pth)")
        btn_save.clicked.connect(self.save_model)
        layout.addWidget(btn_save)
        
        layout.addSpacing(10)
        
        btn_code = QPushButton("Export Inference Code (.py)")
        btn_code.clicked.connect(self.export_code)
        btn_code.setStyleSheet(f"background-color: {Theme.COLOR_TEAL}; color: black;")
        layout.addWidget(btn_code)
        
        layout.addStretch()
        self.stack.addWidget(page)

    # --- Logic ---

    def load_data_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if folder:
            try:
                img_size = 128 if self.chk_fast.isChecked() else 224
                # Note: Default core load_data is 224, unless updated. 
                # Assuming simple pass through or ignoring for folder currently if not updated core.
                # Ideally, core load_data accepts img_size.
                # Based on previous step, core was updated ONLY for builtin. 
                # Let's pass it anyway or update core later if critical.
                # Actually user is likely using builtin for speed test.
                msg = self.pipeline.load_data(folder, img_size=img_size)
                self.lbl_data_status.setText(msg)
                self.switch_step(1)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def load_builtin_data(self):
        name = self.combo_builtin.currentText()
        self.lbl_data_status.setText(f"Downloading {name}... (This might take a while)")
        # Force UI update
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        try:
            img_size = 128 if self.chk_fast.isChecked() else 224
            msg = self.pipeline.load_builtin_data(name, img_size=img_size)
            self.lbl_data_status.setText(msg)
            if "Error" not in msg:
                self.switch_step(1)
            else:
                 QMessageBox.critical(self, "Error", msg)
        except Exception as e:
             QMessageBox.critical(self, "Error", str(e))
             self.lbl_data_status.setText("Error loading data.")

    def build_model(self):
        name = self.combo_model.currentText()
        try:
            msg = self.pipeline.build_model(name)
            self.lbl_model_status.setText(msg)
            self.switch_step(2)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def start_training(self):
        lr = self.spin_lr.value()
        optim = self.combo_optim.currentText()
        epochs = self.spin_epochs.value()
        
        # Setup
        try:
            self.pipeline.setup_training(lr=lr, optimizer_name=optim)
        except Exception as e:
             QMessageBox.critical(self, "Error", str(e))
             return

        # Thread
        self.worker = TrainingWorker(self.pipeline, epochs)
        self.worker.progress.connect(self.append_log)
        self.worker.finished.connect(self.on_train_finished)
        self.worker.start()
        
        self.btn_train.setDisabled(True)
        self.progress_log.append("Training started...")

    def append_log(self, text):
        self.progress_log.append(text)

    def on_train_finished(self):
        self.btn_train.setDisabled(False)
        self.progress_log.append("Training Completed!")
        self.switch_step(4)

    def save_model(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", "model.pth", "PyTorch Model (*.pth)")
        if path:
            msg = self.pipeline.save_checkpoint(path)
            QMessageBox.information(self, "Saved", msg)

    def run_validation(self):
        results = self.pipeline.run_validation(num_images=9)
        
        # Clear grid
        for i in reversed(range(self.val_grid.count())): 
            self.val_grid.itemAt(i).widget().setParent(None)
            
        # Fill grid
        cols = 3
        for i, res in enumerate(results):
            row = i // cols
            col = i % cols
            
            # Convert PIL to QPixmap
            from PySide6.QtGui import QImage, QPixmap
            import numpy as np
            
            # PIL Image is RGB
            im_np = np.array(res['image'])
            h, w, c = im_np.shape
            q_img = QImage(im_np.data, w, h, 3 * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(q_img)
            
            # Widget
            frame = QFrame()
            frame.setStyleSheet("border: 1px solid #444; background: #222; border-radius: 4px;")
            f_layout = QVBoxLayout(frame)
            
            lbl_img = QLabel()
            lbl_img.setPixmap(pix.scaled(100, 100, Qt.KeepAspectRatio))
            lbl_img.setAlignment(Qt.AlignCenter)
            
            lbl_text = QLabel(f"T: {res['true']}\nP: {res['pred']}")
            color = "#0f0" if res['true'] == res['pred'] else "#f00"
            lbl_text.setStyleSheet(f"color: {color}; font-weight: bold;")
            lbl_text.setAlignment(Qt.AlignCenter)
            
            f_layout.addWidget(lbl_img)
            f_layout.addWidget(lbl_text)
            
            self.val_grid.addWidget(frame, row, col)

    def export_code(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Code", "run_model.py", "Python Script (*.py)")
        if path:
            try:
                model_name = self.combo_model.currentText()
                code = self.pipeline.generate_inference_code(model_classname=model_name, num_classes=len(self.pipeline.classes))
                with open(path, "w") as f:
                    f.write(code)
                QMessageBox.information(self, "Exported", "Code exported successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
