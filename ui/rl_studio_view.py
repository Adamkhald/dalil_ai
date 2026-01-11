from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QFrame, QComboBox, QProgressBar, QMessageBox, QGraphicsView, QGraphicsScene)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
from dalil_ai.ui.theme import Theme
import time

try:
    from dalil_ai.core.rl_logic import RLPipeline
    import numpy as np
except ImportError:
    RLPipeline = None

# --- Worker Threads ---

class RLTrainWorker(QThread):
    progress = Signal(int, str) # percent, message
    frame_ready = Signal(QImage) # For live preview
    finished = Signal(str)
    
    def __init__(self, pipeline, env_id, algo, steps, live_preview=False):
        super().__init__()
        self.pipeline = pipeline
        self.env_id = env_id
        self.algo = algo
        self.steps = steps
        self.live_preview = live_preview
        
    def run(self):
        def cb_prog(pct, msg):
            self.progress.emit(pct, msg)
            
        def cb_frame(frame_np):
            # Convert numpy array (H, W, 3) to QImage
            height, width, channel = frame_np.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame_np.data, width, height, bytes_per_line, QImage.Format_RGB888)
            # Copy to ensure data persistence across threads
            self.frame_ready.emit(q_img.copy())

        frame_cb = cb_frame if self.live_preview else None
        result_msg = self.pipeline.train_agent(self.env_id, self.algo, self.steps, cb_prog, frame_cb)
        self.finished.emit(result_msg)

class RLVisualizerWorker(QThread):
    frame_ready = Signal(QImage)
    finished = Signal()
    
    def __init__(self, pipeline, env_id):
        super().__init__()
        self.pipeline = pipeline
        self.env_id = env_id
        self.running = True
        
    def run(self):
        # Run 5 episodes of preview
        data_gen = self.pipeline.run_preview(self.env_id, episodes=5)
        
        for frame in data_gen:
            if not self.running: break
            if frame is None: break
            
            # Convert numpy array (H, W, 3) RGB to QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            self.frame_ready.emit(q_img)
            
            # Limit FPS roughly
            time.sleep(0.033) # ~30 FPS
            
        self.finished.emit()

    def stop(self):
        self.running = False


class RLStudioView(QWidget):
    def __init__(self):
        super().__init__()
        self.pipeline = RLPipeline() if RLPipeline else None
        self.train_worker = None
        self.viz_worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Header
        top_bar = QHBoxLayout()
        logo = QLabel("🎮 RL Studio")
        logo.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {Theme.COLOR_TEAL};")
        top_bar.addWidget(logo)
        top_bar.addStretch()
        
        if not self.pipeline:
             lbl_err = QLabel("Missing deps: install gymnasium stable-baselines3 shimmy pygame")
             lbl_err.setStyleSheet("color: red;")
             top_bar.addWidget(lbl_err)
        
        main_layout.addLayout(top_bar)
        
        # Content Split
        content = QHBoxLayout()
        
        # --- Left: Config ---
        config_frame = QFrame()
        config_frame.setFixedWidth(320)
        config_frame.setStyleSheet(f"background-color: {Theme.COLOR_SIDEBAR_BG}; border-radius: 8px;")
        cf_layout = QVBoxLayout(config_frame)
        
        cf_layout.addWidget(QLabel("1. Environment (Gymnasium)"))
        self.combo_env = QComboBox()
        # Updated env versions and added new ones
        self.combo_env.addItems(["CartPole-v1", "LunarLander-v3", "Pendulum-v1", "Acrobot-v1", "BipedalWalker-v3", "MountainCar-v0"])
        cf_layout.addWidget(self.combo_env)
        
        cf_layout.addSpacing(15)
        cf_layout.addWidget(QLabel("2. Agent (Algorithm)"))
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["PPO", "DQN", "A2C", "SAC"])
        cf_layout.addWidget(self.combo_algo)
        
        cf_layout.addSpacing(15)
        cf_layout.addWidget(QLabel("3. Training Steps"))
        self.combo_steps = QComboBox()
        self.combo_steps.addItems(["1000", "5000", "10000", "50000", "100000"])
        cf_layout.addWidget(self.combo_steps)
        
        cf_layout.addSpacing(15)
        from PySide6.QtWidgets import QCheckBox
        self.chk_preview = QCheckBox("Live Preview (Slower)")
        self.chk_preview.setStyleSheet(f"color: {Theme.COLOR_TEAL};")
        self.chk_preview.setToolTip("Watch the agent learn in real-time. Warning: slows down training.")
        cf_layout.addWidget(self.chk_preview)
        
        cf_layout.addSpacing(20)
        
        # Training Controls
        self.btn_train = QPushButton("Start Training 🚀")
        self.btn_train.setMinimumHeight(40)
        self.btn_train.clicked.connect(self.start_training)
        cf_layout.addWidget(self.btn_train)
        
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #aaa; font-size: 12px; margin-top: 5px;")
        cf_layout.addWidget(self.status_label)
        
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.pbar.setTextVisible(True)
        self.pbar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {Theme.COLOR_TEAL}; }}")
        cf_layout.addWidget(self.pbar)
        
        cf_layout.addStretch()
        
        # Viz & Eval Controls
        cf_layout.addWidget(QLabel("4. Tools"))
        
        btn_layout = QHBoxLayout()
        self.btn_viz = QPushButton("Preview 🎥")
        self.btn_viz.setStyleSheet(f"background-color: {Theme.COLOR_ORANGE}; color: black;")
        self.btn_viz.clicked.connect(self.start_visualization)
        
        self.btn_eval = QPushButton("Evaluate 📊")
        self.btn_eval.setStyleSheet(f"background-color: {Theme.COLOR_PRIMARY}; color: white;")
        self.btn_eval.clicked.connect(self.start_evaluation)
        
        btn_layout.addWidget(self.btn_viz)
        btn_layout.addWidget(self.btn_eval)
        cf_layout.addLayout(btn_layout)

        # Export Code
        self.btn_export = QPushButton("Export Code (.py) 📜")
        self.btn_export.clicked.connect(self.export_code)
        cf_layout.addWidget(self.btn_export)

        # Log Area
        cf_layout.addWidget(QLabel("Logs & Metrics"))
        from PySide6.QtWidgets import QTextEdit
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(150)
        self.log_area.setStyleSheet("font-size: 11px;")
        cf_layout.addWidget(self.log_area)
        
        content.addWidget(config_frame)
        
        # --- Right: Render Area ---
        viz_container = QFrame()
        viz_container.setStyleSheet("background-color: #000; border: 2px solid #444;")
        vc_layout = QVBoxLayout(viz_container)
        
        self.lbl_screen = QLabel("Agent View")
        self.lbl_screen.setAlignment(Qt.AlignCenter)
        self.lbl_screen.setStyleSheet("color: #555; font-size: 20px; font-weight: bold;")
        # Fix for resizing issue: prevent label from forcing layout expansion
        from PySide6.QtWidgets import QSizePolicy
        self.lbl_screen.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        vc_layout.addWidget(self.lbl_screen)
        
        content.addWidget(viz_container)
        
        main_layout.addLayout(content)

    # --- Logic ---
    
    def reset_preview(self):
        # Stop any running workers
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.terminate()
            self.train_worker.wait()
        if self.viz_worker and self.viz_worker.isRunning():
            self.viz_worker.stop()
            self.viz_worker.wait()
            
        # Clear screen
        self.lbl_screen.clear()
        self.lbl_screen.setText("Initializing...")

    def start_training(self):
        if not self.pipeline: return
        
        # Reset Logic
        self.reset_preview()
        
        env = self.combo_env.currentText()
        algo = self.combo_algo.currentText()
        steps = int(self.combo_steps.currentText())
        live = self.chk_preview.isChecked()
        
        self.btn_train.setEnabled(False)
        self.btn_viz.setEnabled(False)
        self.btn_eval.setEnabled(False)
        
        self.pbar.setValue(0)
        self.status_label.setText("Initializing training...")
        if live:
            self.status_label.setText("Initializing training (Preview ON)...")
        
        self.train_worker = RLTrainWorker(self.pipeline, env, algo, steps, live_preview=live)
        self.train_worker.progress.connect(self.on_train_progress)
        if live:
            self.train_worker.frame_ready.connect(self.update_frame)
        self.train_worker.finished.connect(self.on_train_finished)
        self.train_worker.start()

    def on_train_progress(self, pct, msg):
        self.pbar.setValue(pct)
        self.status_label.setText(msg)

    def on_train_finished(self, msg):
        self.btn_train.setEnabled(True)
        self.btn_viz.setEnabled(True)
        self.btn_eval.setEnabled(True)
        self.status_label.setText(msg)
        
        if "failed" in msg or "Error" in msg:
             QMessageBox.critical(self, "Error", msg)
        else:
             QMessageBox.information(self, "RL Studio", msg)

    def start_visualization(self):
        if not self.pipeline: return
        self.reset_preview()
        
        env = self.combo_env.currentText()
        
        self.btn_viz.setEnabled(False)
        self.status_label.setText("Rendering preview...")
        
        self.viz_worker = RLVisualizerWorker(self.pipeline, env)
        self.viz_worker.frame_ready.connect(self.update_frame)
        self.viz_worker.finished.connect(self.on_viz_finished)
        self.viz_worker.start()

    def start_evaluation(self):
        if not self.pipeline: return
        env = self.combo_env.currentText()
        
        self.log("Evaluating agent (10 episodes)...")
        # Inline for now, can be threaded later
        result = self.pipeline.evaluate_agent(env)
        
        if isinstance(result, dict):
            txt = "--- Evaluation Results ---\n"
            for k, v in result.items():
                txt += f"{k}: {v}\n"
            self.log(txt)
        else:
            self.log(str(result))
            
    def export_code(self):
        if not self.pipeline: return
        code = self.pipeline.generate_code()
        
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Export RL Code", "rl_agent.py", "Python Script (*.py)")
        if path:
            try:
                with open(path, "w") as f:
                    f.write(code)
                self.log(f"Code saved to {path}")
            except Exception as e:
                self.log(f"Error saving code: {e}")

    def log(self, msg):
        self.log_area.append(msg)
        
    def update_frame(self, q_img):
        # Scale to fit the container, not the label itself (avoid loop)
        parent = self.lbl_screen.parentWidget()
        if parent:
            container_size = parent.size()
            w = container_size.width() - 4 
            h = container_size.height() - 4
            pixmap = QPixmap.fromImage(q_img)
            self.lbl_screen.setPixmap(pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_viz_finished(self):
        self.btn_viz.setEnabled(True)
        self.status_label.setText("Preview finished.")
