from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QTreeWidget, QTreeWidgetItem, 
                               QTextBrowser, QLabel, QSplitter)
from PySide6.QtCore import Qt
from dalil_ai.ui.theme import Theme
from dalil_ai.core.education_data import EDUCATION_DATA

class EducationView(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # --- Sidebar (Tree) ---
        sidebar_widget = QWidget()
        sidebar_widget.setStyleSheet(f"background-color: {Theme.COLOR_SIDEBAR_BG};")
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        lbl_head = QLabel("📚 Knowledge Base")
        lbl_head.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {Theme.COLOR_PRIMARY}; margin-bottom: 5px;")
        sidebar_layout.addWidget(lbl_head)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: {Theme.COLOR_SIDEBAR_BG};
                color: {Theme.COLOR_TEXT_MAIN};
                border: none;
                font-size: 14px;
            }}
            QTreeWidget::item {{
                padding: 5px;
            }}
            QTreeWidget::item:selected {{
                background-color: {Theme.COLOR_SELECTION};
                color: {Theme.COLOR_TEAL};
            }}
        """)
        self.tree.currentItemChanged.connect(self.on_item_change)
        
        # Populate Tree
        for category, items in EDUCATION_DATA.items():
            cat_item = QTreeWidgetItem(self.tree)
            cat_item.setText(0, category)
            # Make category bold? 
            # (Requires setting font manually on item, skipped for brevity, standard tree behavior is fine)
            
            for topic in items.keys():
                child = QTreeWidgetItem(cat_item)
                child.setText(0, topic)
                
        self.tree.expandAll()
        sidebar_layout.addWidget(self.tree)
        splitter.addWidget(sidebar_widget)
        
        # --- Content Area ---
        content_widget = QWidget()
        # content_widget.setStyleSheet(f"background-color: {Theme.COLOR_MAIN_BG};")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0,0,0,0)
        
        # Header for content
        self.header_frame = QLabel("Welcome to Dalil Education")
        self.header_frame.setAlignment(Qt.AlignCenter)
        self.header_frame.setStyleSheet(f"""
            background-color: {Theme.COLOR_PRIMARY}; 
            color: white; 
            font-size: 24px; 
            font-weight: bold; 
            padding: 20px;
        """)
        content_layout.addWidget(self.header_frame)
        
        self.text_view = QTextBrowser()
        self.text_view.setOpenExternalLinks(True)
        # Base CSS for the browser
        self.text_view.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {Theme.COLOR_MAIN_BG};
                color: {Theme.COLOR_TEXT_MAIN};
                border: none;
                padding: 20px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 16px;
                line-height: 1.6;
            }}
        """)
        
        content_layout.addWidget(self.text_view)
        
        # Initial Welcome Message
        welcome_html = f"""
        <center>
        <h2 style="color:{Theme.COLOR_TEAL}">Master Machine Learning</h2>
        <p>Select a topic from the sidebar to begin your journey.</p>
        <p>From Classical Regression to Deep Reinforcement Learning, we have it covered.</p>
        </center>
        """
        self.text_view.setHtml(welcome_html)
        
        splitter.addWidget(content_widget)
        splitter.setStretchFactor(1, 4) # Content much wider
        
        layout.addWidget(splitter)

    def on_item_change(self, current, previous):
        if not current: return
        
        # If it has a parent, it's a topic. If not, it's a category.
        if current.parent():
            category = current.parent().text(0)
            topic = current.text(0)
            
            content = EDUCATION_DATA.get(category, {}).get(topic, "")
            
            # Update Header
            self.header_frame.setText(topic)
            
            # Wrap content in HTML styling for consistency
            html_content = f"""
            <style>
                h1 {{ color: {Theme.COLOR_PRIMARY}; font-size: 28px; margin-bottom: 10px; }}
                h3 {{ color: {Theme.COLOR_TEAL}; font-size: 20px; margin-top: 20px; margin-bottom: 5px; }}
                p, li {{ font-size: 16px; line-height: 1.6; color: {Theme.COLOR_TEXT_MAIN}; }}
                b {{ color: {Theme.COLOR_ORANGE}; }}
                ul {{ margin-top: 0; }}
            </style>
            {content}
            """
            self.text_view.setHtml(html_content)
