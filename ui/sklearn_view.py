from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QFileDialog, QComboBox, QTableView, 
                               QHeaderView, QStackedWidget, QGroupBox, QFormLayout, 
                               QSpinBox, QTextEdit, QMessageBox, QFrame)
from PySide6.QtCore import Qt, QAbstractTableModel
from PySide6.QtGui import QStandardItemModel, QStandardItem
from dalil_ai.core.sklearn_logic import SklearnPipeline
from dalil_ai.ui.theme import Theme
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt

# Simple Pandas Table Model
class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super(PandasModel, self).__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

class SklearnView(QWidget):
    def __init__(self):
        super().__init__()
        self.pipeline = SklearnPipeline()
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Pipeline Sidebar (Steps)
        self.step_list = QFrame()
        self.step_list.setObjectName("Sidebar")
        self.step_list.setFixedWidth(200)
        step_layout = QVBoxLayout(self.step_list)
        step_layout.setContentsMargins(0, 10, 0, 0)
        step_layout.setSpacing(5)
        
        self.steps = [
            "1. Data Load", "2. Preprocess", "3. Feature Eng.", 
            "4. Model Selection", "5. Hyperparameters", "6. Evaluate", "7. Export"
        ]
        self.step_buttons = []
        
        for i, step in enumerate(self.steps):
            btn = QPushButton(step)
            btn.setObjectName("SidebarButton")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, idx=i: self.switch_step(idx))
            self.step_buttons.append(btn)
            step_layout.addWidget(btn)
            
        step_layout.addStretch()
        main_layout.addWidget(self.step_list)

        # 2. Main Content Area
        self.content_area = QStackedWidget()
        main_layout.addWidget(self.content_area)

        # Initialize Step Pages
        self.create_data_load_page()
        self.create_preprocess_page()
        self.create_feature_page()
        self.create_model_page()
        self.create_params_page()
        self.create_eval_page()
        self.create_export_page()

        # Set default
        self.step_buttons[0].setChecked(True)
        self.content_area.setCurrentIndex(0)

    def switch_step(self, index):
        for i, btn in enumerate(self.step_buttons):
            btn.setChecked(i == index)
        self.content_area.setCurrentIndex(index)

    def create_data_load_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 1: Load Dataset")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_PRIMARY}; font-weight: bold;")
        layout.addWidget(lbl)
        
        btn_box = QHBoxLayout()
        self.btn_load = QPushButton("Load CSV/Excel")
        self.btn_load.clicked.connect(self.load_dataset)
        self.lbl_path = QLabel("No file loaded")
        btn_box.addWidget(self.btn_load)
        btn_box.addWidget(self.lbl_path)
        btn_box.addStretch()
        layout.addLayout(btn_box)
        
        self.table_preview = QTableView()
        layout.addWidget(self.table_preview)
        
        self.content_area.addWidget(page)

    def create_preprocess_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 2: Preprocessing")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_PRIMARY}; font-weight: bold;")
        layout.addWidget(lbl)
        
        form_layout = QFormLayout()
        self.combo_target = QComboBox()
        self.combo_imputer = QComboBox()
        self.combo_imputer.addItems(["mean", "median", "most_frequent"])
        
        form_layout.addRow("Select Target Column:", self.combo_target)
        form_layout.addRow("Missing Value Strategy:", self.combo_imputer)
        
        layout.addLayout(form_layout)
        
        btn_process = QPushButton("Run Preprocessing")
        btn_process.clicked.connect(self.run_preprocess)
        layout.addWidget(btn_process)
        
        self.lbl_process_status = QLabel("")
        layout.addWidget(self.lbl_process_status)
        layout.addStretch()
        
        self.content_area.addWidget(page)

    def create_feature_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        lbl = QLabel("Step 3: Feature Engineering")
        lbl.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(lbl)
        layout.addWidget(QLabel("Automatic scaling and encoding is handled in Step 2 for this demo."))
        layout.addStretch()
        self.content_area.addWidget(page)

    def create_model_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 4: Model Selection")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_PRIMARY}; font-weight: bold;")
        layout.addWidget(lbl)
        
        form = QFormLayout()
        self.combo_task = QComboBox()
        self.combo_task.addItems(["Classification", "Regression"])
        self.combo_task.currentTextChanged.connect(self.update_models)
        
        self.combo_model = QComboBox()
        # Default
        self.update_models("Classification")
        
        form.addRow("Task Type:", self.combo_task)
        form.addRow("Algorithm:", self.combo_model)
        layout.addLayout(form)
        
        btn_train = QPushButton("Train Model")
        btn_train.clicked.connect(self.run_training)
        layout.addWidget(btn_train)
        
        self.lbl_train_status = QLabel("")
        layout.addWidget(self.lbl_train_status)
        layout.addStretch()
        
        self.content_area.addWidget(page)

    def create_params_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        lbl = QLabel("Step 5: Hyperparameters")
        lbl.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(lbl)
        layout.addWidget(QLabel("Using default parameters for selected model."))
        layout.addStretch()
        self.content_area.addWidget(page)

    def create_eval_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        lbl = QLabel("Step 6: Evaluation")
        lbl.setStyleSheet(f"font-size: 18px; color: {Theme.COLOR_PRIMARY}; font-weight: bold;")
        layout.addWidget(lbl)
        
        self.btn_eval = QPushButton("Evaluate Results")
        self.btn_eval.clicked.connect(self.run_evaluation)
        layout.addWidget(self.btn_eval)
        
        self.txt_metrics = QTextEdit()
        self.txt_metrics.setReadOnly(True)
        self.txt_metrics.setMaximumHeight(150)
        layout.addWidget(self.txt_metrics)
        
        self.plot_area = QVBoxLayout()
        layout.addLayout(self.plot_area)
        
        self.content_area.addWidget(page)

    def create_export_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        lbl = QLabel("Step 7: Export")
        lbl.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(lbl)
        layout.addWidget(QLabel("Export trained model (.pkl) or results report."))
        
        btn_save = QPushButton("Save Model")
        layout.addWidget(btn_save)
        
        layout.addSpacing(10)
        btn_code = QPushButton("Export Source Code (.py)")
        btn_code.clicked.connect(self.export_code)
        btn_code.setStyleSheet(f"background-color: {Theme.COLOR_TEAL}; color: black;")
        layout.addWidget(btn_code)
        
        layout.addStretch()
        self.content_area.addWidget(page)
    
    # ... (rest of logic) ...

    def export_code(self):
        code = self.pipeline.generate_code()
        path, _ = QFileDialog.getSaveFileName(self, "Export Code", "pipeline_script.py", "Python Script (*.py)")
        if path:
            try:
                with open(path, "w") as f:
                    f.write(code)
                QMessageBox.information(self, "Success", f"Code exported to {path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    # --- Logic Connectors ---
    
    def load_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
        if path:
            self.lbl_path.setText(path)
            try:
                df = self.pipeline.load_data(path)
                model = PandasModel(df)
                self.table_preview.setModel(model)
                
                # Update cols
                cols = self.pipeline.get_columns()
                self.combo_target.clear()
                self.combo_target.addItems(cols)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def run_preprocess(self):
        target = self.combo_target.currentText()
        if not target:
            return
        
        strat = self.combo_imputer.currentText()
        try:
            msg = self.pipeline.preprocess_data(target_col=target, impute_strategy=strat)
            self.lbl_process_status.setText(msg)
            self.switch_step(3) # Move to model selection
        except Exception as e:
             QMessageBox.critical(self, "Error", str(e))

    def update_models(self, task):
        self.combo_model.clear()
        if task == "Classification":
            self.combo_model.addItems(["Logistic Regression", "Random Forest", "SVM"])
        else:
            self.combo_model.addItems(["Linear Regression", "Random Forest", "SVR"])

    def run_training(self):
        model_name = self.combo_model.currentText()
        task = self.combo_task.currentText()
        
        try:
            self.pipeline.select_model(model_name, task)
            msg = self.pipeline.train_model()
            self.lbl_train_status.setText(msg)
            self.switch_step(5) # Evalu
        except Exception as e:
             QMessageBox.critical(self, "Error", str(e))

    def run_evaluation(self):
        try:
            metrics, report = self.pipeline.evaluate_model()
            txt = "Metrics:\n"
            for k,v in metrics.items():
                txt += f"{k}: {v}\n"
            if report:
                txt += "\nClassification Report:\n" + str(report)
            
            self.txt_metrics.setText(txt)
            
            # Plot
            fig = self.pipeline.plot_results()
            if fig:
                # Clear previous plot if any (simple way)
                while self.plot_area.count():
                    item = self.plot_area.takeAt(0)
                    widget = item.widget()
                    if widget: widget.deleteLater()
                
                canvas = FigureCanvasQTAgg(fig)
                self.plot_area.addWidget(canvas)
                canvas.draw()
                
        except Exception as e:
             QMessageBox.critical(self, "Error", str(e))
