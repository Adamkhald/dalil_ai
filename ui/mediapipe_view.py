from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QFrame, QComboBox, QMessageBox, QSizePolicy)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from dalil_ai.ui.theme import Theme
import cv2
import sys

try:
    from dalil_ai.core.mediapipe_logic import MediaPipeProcessor
    MP_AVAILABLE = True
except Exception as e:
    print(f"MediaPipe import failed: {e}")
    MediaPipeProcessor = None
    MP_AVAILABLE = False

class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    error_occurred = Signal(str)
    
    def __init__(self, processor, mode="Face Detection", cam_index=0):
        super().__init__()
        self.processor = processor
        self.mode = mode
        self.cam_index = cam_index
        self.running = True

    def update_mode(self, new_mode):
        self.mode = new_mode

    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            self.error_occurred.emit("Could not open webcam.")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process Frame
            try:
                annotated_frame = self.processor.process_frame(frame, self.mode)
                
                # Convert to Qt Image (H, W, 3)
                h, w, ch = annotated_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(annotated_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                self.frame_ready.emit(qt_img.copy()) # Copy to be safe
                
            except Exception as e:
                print(f"MP Error: {e}")
                
        cap.release()

    def stop(self):
        self.running = False


class MediaPipeView(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker = None
        self.processor = None
        
        # Init processor if deps available and import succeeded
        if MP_AVAILABLE and MediaPipeProcessor:
            try:
                self.processor = MediaPipeProcessor()
            except Exception as e:
                print(f"Error init MP: {e}")
                self.processor = None

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        
        # Header
        top_bar = QHBoxLayout()
        logo = QLabel("🧿 MediaPipe Lab")
        logo.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {Theme.COLOR_TEAL};")
        top_bar.addWidget(logo)
        top_bar.addStretch()
        
        if not MediaPipeProcessor:
            lbl_err = QLabel("Missing deps: install mediapipe opencv-python")
            lbl_err.setStyleSheet("color: red;")
            top_bar.addWidget(lbl_err)
            
        main_layout.addLayout(top_bar)
        
        # Main Content
        content = QHBoxLayout()
        
        # --- Sidebar ---
        sidebar = QFrame()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet(f"background-color: {Theme.COLOR_SIDEBAR_BG}; border-radius: 8px;")
        sb_layout = QVBoxLayout(sidebar)
        
        sb_layout.addWidget(QLabel("1. Vision Solution"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Face Detection", "Face Mesh", "Hands", "Pose"])
        self.combo_mode.currentTextChanged.connect(self.on_mode_changed)
        sb_layout.addWidget(self.combo_mode)
        
        sb_layout.addSpacing(20)
        
        sb_layout.addWidget(QLabel("2. Controls"))
        self.btn_start = QPushButton("Start Webcam 📷")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet(f"background-color: {Theme.COLOR_PRIMARY}; color: white;")
        self.btn_start.clicked.connect(self.toggle_camera)
        sb_layout.addWidget(self.btn_start)
        
        sb_layout.addStretch()
        content.addWidget(sidebar)
        
        # --- Video Area ---
        video_frame = QFrame()
        video_frame.setStyleSheet("background-color: #000; border: 2px solid #444;")
        vf_layout = QVBoxLayout(video_frame)
        
        self.lbl_video = QLabel("Webcam Off")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("color: #555; font-size: 20px; font-weight: bold;")
        self.lbl_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        vf_layout.addWidget(self.lbl_video)
        
        content.addWidget(video_frame)
        
        main_layout.addLayout(content)

    def toggle_camera(self):
        if not self.processor:
            QMessageBox.critical(self, "Error", "MediaPipe backend not loaded.")
            return

        if self.worker and self.worker.isRunning():
            # Stop
            self.worker.stop()
            self.worker.wait()
            self.worker = None
            self.btn_start.setText("Start Webcam 📷")
            self.btn_start.setStyleSheet(f"background-color: {Theme.COLOR_PRIMARY}; color: white;")
            self.lbl_video.clear()
            self.lbl_video.setText("Webcam Off")
        else:
            # Start
            mode = self.combo_mode.currentText()
            self.worker = VideoWorker(self.processor, mode)
            self.worker.frame_ready.connect(self.update_image)
            self.worker.error_occurred.connect(self.on_error)
            self.worker.start()
            
            self.btn_start.setText("Stop Webcam ⏹")
            self.btn_start.setStyleSheet(f"background-color: {Theme.COLOR_RED}; color: white;")

    def on_mode_changed(self, text):
        if self.worker:
            self.worker.update_mode(text)

    def update_image(self, q_img):
        # Intelligent Scaling
        parent = self.lbl_video.parentWidget()
        if parent:
            container_size = parent.size()
            w = container_size.width() - 4
            h = container_size.height() - 4
            
            pixmap = QPixmap.fromImage(q_img)
            self.lbl_video.setPixmap(pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_error(self, msg):
        QMessageBox.critical(self, "Camera Error", msg)
        self.toggle_camera() # Reset UI state

    def closeEvent(self, event):
        # Cleanup
        if self.worker:
            self.worker.stop()
            self.worker.wait()
        if self.processor:
            self.processor.close()
        super().closeEvent(event)
