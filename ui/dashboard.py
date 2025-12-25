from PySide6.QtWidgets import (QWidget, QGridLayout, QVBoxLayout, QLabel, 
                               QFrame, QPushButton, QScrollArea, QHBoxLayout)
from PySide6.QtCore import Qt, Signal
from dalil_ai.ui.theme import Theme

class LibraryCard(QFrame):
    clicked = Signal(str)

    def __init__(self, title, description, tags, icon_text="📊"):
        super().__init__()
        self.setObjectName("LibraryCard")
        self.title = title
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(280, 180) # Fixed size for grid consistency

        layout = QVBoxLayout(self)
        
        # Icon + Title
        header = QLabel(f"{icon_text}  {title}")
        header.setObjectName("CardTitle")
        layout.addWidget(header)
        
        # Desc
        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {Theme.COLOR_TEXT_MUTED}; font-size: 12px;")
        layout.addWidget(desc)
        
        layout.addStretch()
        
        # Tags
        tag_layout = QGridLayout()
        col = 0
        row = 0
        for tag in tags:
            t = QLabel(tag)
            t.setObjectName("CardTag")
            t.setAlignment(Qt.AlignCenter)
            tag_layout.addWidget(t, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        layout.addLayout(tag_layout)

    def mousePressEvent(self, event):
        self.clicked.emit(self.title)
        super().mousePressEvent(event)

class Dashboard(QWidget):
    library_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Main Layout (Wrapper)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        
        # Scroll Area for ENTIRE page
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"background-color: {Theme.COLOR_MAIN_BG};") # Ensure bg matches
        
        # Content Widget
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(30)
        
        # 1. Header
        lbl_welcome = QLabel("Welcome to Dalil AI 🚀")
        lbl_welcome.setStyleSheet("font-size: 32px; font-weight: bold; color: white;")
        content_layout.addWidget(lbl_welcome)
        
        lbl_sub = QLabel("Your offline, privacy-first AI research laboratory.")
        lbl_sub.setStyleSheet(f"font-size: 16px; color: {Theme.COLOR_TEXT_MUTED}; margin-bottom: 20px;")
        content_layout.addWidget(lbl_sub)

        # 2. Library Grid
        grid = QGridLayout()
        grid.setSpacing(20)
        
        libraries = [
            ("Scikit-Learn", "Classic Machine Learning algorithms for classification, regression, and clustering.", ["Beginner Friendly", "ML", "Data Science"], "🔬"),
            ("PyTorch", "Deep Learning framework with dynamic computation graphs for research.", ["Deep Learning", "Research", "Dynamic"], "🔥"),
            ("TensorFlow", "End-to-end open source platform for machine learning production.", ["Production", "Deep Learning", "Keras"], "🧠"),
            ("MediaPipe", "Cross-platform, customizable ML solutions for live and streaming media.", ["Computer Vision", "Real-time", "Pose"], "📸"),
            ("RL Studio", "Teach AI agents to play games and solve tasks via trial and error.", ["Reinforcement Learning", "Agents", "Gym"], "🎮"),
        ]
        
        row, col = 0, 0
        for title, desc, tags, icon in libraries:
            card = LibraryCard(title, desc, tags, icon)
            card.clicked.connect(self.on_card_click)
            grid.addWidget(card, row, col)
            
            col += 1
            if col > 2: # 3 columns wide
                col = 0
                row += 1
        
        content_layout.addLayout(grid)
        
        # 3. Separator
        content_layout.addSpacing(40)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {Theme.COLOR_BORDER};")
        content_layout.addWidget(sep)
        content_layout.addSpacing(10)

        # 4. Golden Information / Walkthrough
        lbl_info = QLabel("✨ The Dalil Ecosystem")
        lbl_info.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {Theme.COLOR_ORANGE};")
        content_layout.addWidget(lbl_info)
        
        walkthrough_data = [
            ("🧠 Scikit-Learn: The Foundation", 
             "Start here if you have tabular data (Excel, CSV). Use it for predicting house prices (Regression), classifying emails as spam (Classification), or grouping customer segments (Clustering). It's fast, interpretable, and mathematically robust."),
            
            ("🔥 PyTorch: Research & Flexibility", 
             "The choice of modern AI researchers. Use PyTorch when you need to build custom Neural Networks, experiment with new architectures (Transformers, CNNs), or need dynamic computation graphs. We provide a 'Fast Mode' for CPU training."),
            
            ("⚡ TensorFlow: Production Ready", 
             "Built by Google for scale. Use TensorFlow when you want to deploy your models to mobile devices (Android/iOS) or the web. Our pipeline automatically includes TFLite export logic so your models are ready for the real world."),
             
            ("👁️ MediaPipe: Real-Time Vision", 
             "Zero-latency computer vision. Unlike standard DL models that are slow, MediaPipe is optimized for streaming. Use it to track hands, map 468 face landmarks, or estimate full-body pose in real-time using just your webcam."),
             
            ("🤖 RL Studio: Agent Training", 
             "Reinforcement Learning is different—it's about learning from experience. Use this to train 'Agents' to solve dynamic tasks like balancing a pole or landing a spaceship. We use Stable-Baselines3 and Gymnasium (formerly OpenAI Gym).")
        ]
        
        for title, text in walkthrough_data:
            info_box = QFrame()
            info_box.setStyleSheet(f"background-color: {Theme.COLOR_SIDEBAR_BG}; border-radius: 8px; padding: 15px; border-left: 4px solid {Theme.COLOR_PRIMARY};")
            ib_layout = QVBoxLayout(info_box)
            
            l_title = QLabel(title)
            l_title.setStyleSheet("font-size: 16px; font-weight: bold; color: white; margin-bottom: 5px;")
            ib_layout.addWidget(l_title)
            
            l_text = QLabel(text)
            l_text.setWordWrap(True)
            l_text.setStyleSheet(f"color: {Theme.COLOR_TEXT_MAIN}; font-size: 14px; line-height: 1.4;")
            ib_layout.addWidget(l_text)
            
            content_layout.addWidget(info_box)
            content_layout.addSpacing(10)

        content_layout.addStretch()
        
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    def on_card_click(self, title):
        self.library_selected.emit(title)
