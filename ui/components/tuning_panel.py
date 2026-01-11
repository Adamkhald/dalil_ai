from PySide6.QtWidgets import (QWidget, QVBoxLayout, QFormLayout, QSpinBox, 
                               QDoubleSpinBox, QCheckBox, QComboBox, QLabel, QGroupBox)

class TuningPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        
        self.group = QGroupBox("Hyperparameters")
        self.group.setLayout(self.form_layout)
        self.layout.addWidget(self.group)
        self.layout.addStretch()
        
        self.inputs = {}

    def update_for_model(self, model_name):
        self.clear_layout()
        self.inputs = {}
        
        if "Random Forest" in model_name:
            self.add_spin("n_estimators", 100, 10, 1000, 10, "Number of Trees")
            self.add_spin("max_depth", 0, 0, 100, 1, "Max Depth (0=None)")
            self.add_spin("min_samples_split", 2, 2, 20, 1, "Min Samples Split")
            
        elif "SVM" in model_name or "SVR" in model_name:
            self.add_double_spin("C", 1.0, 0.01, 100.0, "Regularization (C)")
            self.add_combo("kernel", ["rbf", "linear", "poly", "sigmoid"], "Kernel")
            
        elif "Logistic" in model_name or "Linear" in model_name:
            self.add_check("fit_intercept", True, "Fit Intercept")
            if "Logistic" in model_name:
                self.add_double_spin("C", 1.0, 0.01, 100.0, "Inverse Regularization (C)")
        
        else:
            lbl = QLabel("No tunable parameters for this model demo.")
            self.form_layout.addRow(lbl)

    def add_spin(self, name, val, min_v, max_v, step, label):
        sb = QSpinBox()
        sb.setRange(min_v, max_v)
        sb.setValue(val)
        sb.setSingleStep(step)
        self.form_layout.addRow(label, sb)
        self.inputs[name] = sb

    def add_double_spin(self, name, val, min_v, max_v, label):
        sb = QDoubleSpinBox()
        sb.setRange(min_v, max_v)
        sb.setValue(val)
        sb.setSingleStep(0.1)
        self.form_layout.addRow(label, sb)
        self.inputs[name] = sb

    def add_check(self, name, val, label):
        cb = QCheckBox()
        cb.setChecked(val)
        self.form_layout.addRow(label, cb)
        self.inputs[name] = cb

    def add_combo(self, name, items, label):
        cb = QComboBox()
        cb.addItems(items)
        self.form_layout.addRow(label, cb)
        self.inputs[name] = cb

    def clear_layout(self):
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)

    def get_params(self):
        params = {}
        for name, widget in self.inputs.items():
            if isinstance(widget, QSpinBox):
                val = widget.value()
                if name == "max_depth" and val == 0:
                    val = None
                params[name] = val
            elif isinstance(widget, QDoubleSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                params[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
        return params
