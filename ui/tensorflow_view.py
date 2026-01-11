from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QFrame, QStackedWidget, QFileDialog, 
                               QComboBox, QSpinBox, QMessageBox, QTextEdit, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal
from dalil_ai.ui.theme import Theme
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

try:
    from dalil_ai.core.tensorflow_logic import TensorFlowPipeline
except ImportError:
    TensorFlowPipeline = None

# Worker Thread for TF Training to avoid freezing UI
class TFTrainerThread(QThread):
    progress = Signal(str)
    finished = Signal()
    
    def __init__(self, pipeline, epochs, opt, lr):
        super().__init__()
        self.pipeline = pipeline
        self.epochs = epochs
        self.opt = opt
        self.lr = lr
        
    def run(self):
        # We pass a lambda/function that emits signals to the pipeline's callback mechanism
        def update_ui(msg):
            self.progress.emit(msg)
            
        self.pipeline.train(epochs=self.epochs, optimizer_name=self.opt, learning_rate=self.lr, update_callback=update_ui)
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
        
        self.steps = ["1. Dataset", "2. Architecture", "3. Train", "4. Predict", "5. Export"]
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
        self.create_prediction_page()
        self.create_export_page()
        
        self.switch_step(0)

    def switch_step(self, index):
        for i, btn in enumerate(self.step_buttons):
            btn.setChecked(i == index)
        self.stack.setCurrentIndex(index)

    def create_data_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 1: Data Type & Source")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_ORANGE}; font-weight: bold;")
        layout.addWidget(lbl)
        
        # Mode Selection
        layout.addWidget(QLabel("Select Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems([
            "Image Classification", 
            "Tabular Regression (CSV)", 
            "Tabular Classification (CSV)", 
            "Text Classification (CSV)",
            "Timeseries Forecasting (CSV)"
        ])
        self.combo_mode.currentTextChanged.connect(self.on_mode_change)
        layout.addWidget(self.combo_mode)
        
        layout.addSpacing(15)
        
        self.lbl_instruction = QLabel("Select an Image Directory (Grid/Folder format).")
        layout.addWidget(self.lbl_instruction)
        
        self.btn_load = QPushButton("Select Folder")
        self.btn_load.clicked.connect(self.load_data)
        layout.addWidget(self.btn_load)
        
        self.lbl_data_status = QLabel("No Data.")
        layout.addWidget(self.lbl_data_status)
        
        # Built-in Section
        layout.addSpacing(15)
        layout.addWidget(QLabel("Or Load Built-in Keras Dataset:"))
        self.combo_builtin = QComboBox()
        self.combo_builtin.addItems([
            "Select...", 
            "MNIST", "Fashion MNIST", "CIFAR-10", 
            "IMDB Reviews", "California Housing", 
            "Iris Plants", "Sine Wave (Synthetic)"
        ])
        self.combo_builtin.currentIndexChanged.connect(self.load_builtin)
        layout.addWidget(self.combo_builtin)
        
        layout.addStretch()
        self.stack.addWidget(page)
        
    def load_builtin(self):
        name = self.combo_builtin.currentText()
        if "Select" in name: return
        try:
             msg = self.pipeline.load_builtin_data(name)
             self.lbl_data_status.setText(msg)
             # Auto switch to Image mode logic if needed, but pipeline handles format
             self.switch_step(1)
        except Exception as e:
             QMessageBox.critical(self, "Error", str(e))
        
    def on_mode_change(self, text):
        if "Image" in text:
            self.lbl_instruction.setText("Select an Image Directory (Grid/Folder format).")
            self.btn_load.setText("Select Folder")
            self.combo_model.clear()
            self.combo_model.addItems(["mobilenet_v2", "resnet50"])
            self.combo_builtin.setEnabled(True)
        else:
            self.lbl_instruction.setText("Select a CSV File.")
            self.btn_load.setText("Select CSV")
            self.combo_model.clear()
            self.combo_builtin.setEnabled(False)
            
            if "Tabular" in text:
                 self.combo_model.addItem("Dense Network (Auto)")
            elif "Text" in text:
                 self.combo_model.addItem("LSTM (Text)")
            elif "Timeseries" in text:
                 self.combo_model.addItem("LSTM (Sequential)")
        
    def create_arch_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 2: Model Architecture")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_ORANGE}; font-weight: bold;")
        layout.addWidget(lbl)
        
        layout.addWidget(QLabel("Choose Architecture:"))
        self.combo_model = QComboBox()
        # Initial population (default Image)
        self.combo_model.addItems(["mobilenet_v2", "resnet50"])
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
        
        layout.addWidget(QLabel("Optimizer:"))
        self.combo_opt = QComboBox()
        self.combo_opt.addItems(["adam", "sgd", "rmsprop"])
        layout.addWidget(self.combo_opt)
        
        from PySide6.QtWidgets import QDoubleSpinBox
        layout.addWidget(QLabel("Learning Rate:"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.0001, 1.0)
        self.spin_lr.setSingleStep(0.001)
        self.spin_lr.setDecimals(4)
        self.spin_lr.setValue(0.001)
        layout.addWidget(self.spin_lr)
        
        btn = QPushButton("Start Training")
        btn.clicked.connect(self.start_training)
        layout.addWidget(btn)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background-color: #222; color: #ddd;")
        self.log_area.setMaximumHeight(150)
        layout.addWidget(self.log_area)
        
        # Plot Area
        self.plot_layout = QVBoxLayout()
        layout.addLayout(self.plot_layout)
        
        layout.addStretch()
        self.stack.addWidget(page)

    
    def create_prediction_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        lbl = QLabel("Step 4: Prediction & Analysis")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_ORANGE}; font-weight: bold;")
        layout.addWidget(lbl)
        
        # 1. Visualization (Time Series / Reg)
        layout.addWidget(QLabel("Model Analysis:"))
        btn_viz = QPushButton("Visualize Test Set Predictions")
        btn_viz.clicked.connect(self.run_visualization)
        layout.addWidget(btn_viz)
        
        self.viz_layout = QVBoxLayout()
        layout.addLayout(self.viz_layout)
        
        layout.addSpacing(20)
        
        # 2. Single Prediction
        layout.addWidget(QLabel("Single Prediction:"))
        
        # Input Area (Dynamic)
        self.pred_input_layout = QHBoxLayout()
        self.txt_input = QTextEdit() # For Text/Tabular
        self.txt_input.setPlaceholderText("Enter text or comma-separated numbers...")
        self.txt_input.setMaximumHeight(50)
        self.btn_browse_img = QPushButton("Load Image")
        self.btn_browse_img.clicked.connect(self.browse_predict_image)
        self.btn_browse_img.setVisible(False)
        
        self.pred_input_layout.addWidget(self.txt_input)
        self.pred_input_layout.addWidget(self.btn_browse_img)
        
        btn_predict = QPushButton("Predict")
        btn_predict.clicked.connect(self.run_single_prediction)
        self.pred_input_layout.addWidget(btn_predict)
        layout.addLayout(self.pred_input_layout)
        
        self.lbl_pred_result = QLabel("")
        self.lbl_pred_result.setStyleSheet(f"font-size: 16px; color: {Theme.COLOR_TEAL}; font-weight: bold;")
        layout.addWidget(self.lbl_pred_result)
        
        layout.addStretch()
        self.stack.addWidget(page)

    def create_export_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        lbl = QLabel("Step 5: Export (TFLite)")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_ORANGE}; font-weight: bold;")
        layout.addWidget(lbl)
        
        btn = QPushButton("Convert & Save .tflite")
        btn.clicked.connect(self.export_model)
        layout.addWidget(btn)
        
        layout.addStretch()
        self.stack.addWidget(page)

    # --- Logic ---
    def load_data(self):
        mode = self.combo_mode.currentText()
        
        if "Image" in mode:
            d = QFileDialog.getExistingDirectory(self, "Select Image Data")
            if d:
                try:
                    msg = self.pipeline.load_image_data(d)
                    self.lbl_data_status.setText(msg)
                    self.switch_step(1)
                except Exception as e:
                    QMessageBox.critical(self, "Error", str(e))
        else:
            # All CSV based
            f, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
            if f:
                try:
                    import pandas as pd
                    import numpy as np
                    df = pd.read_csv(f)
                    
                    if "Tabular Regression" in mode:
                         X = df.iloc[:, :-1].values.astype(np.float32)
                         y = df.iloc[:, -1].values.astype(np.float32)
                         msg = self.pipeline.load_tabular_data(X, y)
                         
                    elif "Tabular Classification" in mode:
                         X = df.iloc[:, :-1].values.astype(np.float32)
                         y = df.iloc[:, -1].values.astype(np.int32)
                         # Simple heuristics: count unique in Y
                         num_classes = len(np.unique(y))
                         msg = self.pipeline.load_tabular_class_data(X, y, num_classes)
                         
                    elif "Text" in mode:
                         # Assume 1st col text, 2nd col label
                         texts = df.iloc[:, 0].astype(str).values
                         labels = df.iloc[:, 1].values.astype(np.int32)
                         msg = self.pipeline.load_text_data(texts, labels)
                         
                    elif "Timeseries" in mode:
                         # Assume all cols features, predict next step of last col (very simple demo)
                         data = df.values.astype(np.float32)
                         # Make a dummy 3D shape (samples, 10, features)
                         # This is complex to generalize, so we just use sliding window of 1
                         X = data[:-1]
                         y = data[1:, -1] # Predict next value of last feature
                         X = np.expand_dims(X, axis=1) # (N, 1, F)
                         msg = self.pipeline.load_timeseries_data(X, y)
                    
                    self.lbl_data_status.setText(msg)
                    self.switch_step(1)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")

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
        opt = self.combo_opt.currentText()
        lr = self.spin_lr.value()
        
        self.log_area.append(f"Starting training (Opt: {opt}, LR: {lr})...")
        self.thread = TFTrainerThread(self.pipeline, epochs, opt, lr)
        self.thread.progress.connect(self.log_area.append)
        self.thread.finished.connect(self.on_train_finished)
        self.thread.start()

    def on_train_finished(self):
        self.log_area.append("Training Complete!")
        # Show plots
        fig = self.pipeline.plot_results()
        if fig:
            # Clear old plots
            for i in reversed(range(self.plot_layout.count())): 
                self.plot_layout.itemAt(i).widget().setParent(None)
                
            canvas = FigureCanvas(fig)
            self.plot_layout.addWidget(canvas)
            canvas.draw()
            
        # Update UI for prediction page based on mode
        self.update_pred_ui()
        
    def update_pred_ui(self):
        mode = self.pipeline.mode
        if "IMAGE" in mode:
            self.txt_input.setVisible(False)
            self.btn_browse_img.setVisible(True)
        else:
            self.txt_input.setVisible(True)
            self.btn_browse_img.setVisible(False)
            
            if "TABULAR" in mode:
                 self.txt_input.setPlaceholderText("Enter example: 1.2, 0.5, 3.4 ...")
            elif "TEXT" in mode:
                 self.txt_input.setPlaceholderText("Enter text to classify...")

    def run_visualization(self):
        fig = self.pipeline.visualize_predictions()
        if fig:
             for i in reversed(range(self.viz_layout.count())): 
                self.viz_layout.itemAt(i).widget().setParent(None)
             canvas = FigureCanvas(fig)
             self.viz_layout.addWidget(canvas)
             canvas.draw()
        else:
             QMessageBox.information(self, "Info", "Visualization not available for this mode or model not trained.")

    def browse_predict_image(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png)")
        if f:
             self.current_pred_image = f
             self.lbl_pred_result.setText(f"Selected: {f}")

    def run_single_prediction(self):
        mode = self.pipeline.mode
        data = None
        if "IMAGE" in mode:
             if hasattr(self, 'current_pred_image'):
                 data = self.current_pred_image
             else:
                 self.lbl_pred_result.setText("Please load an image first.")
                 return
        else:
             data = self.txt_input.toPlainText()
             if not data:
                  self.lbl_pred_result.setText("Please enter input data.")
                  return
        
        res = self.pipeline.predict_single(data)
        self.lbl_pred_result.setText(res)

    def export_model(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save TFLite", "model.tflite", "TFLite (*.tflite)")
        if path:
            try:
                msg = self.pipeline.export_tflite(path)
                QMessageBox.information(self, "Success", msg)
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
