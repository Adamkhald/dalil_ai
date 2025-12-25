import sys
import os

# 1. Environment Safety Setup
# Ensure local directory is in path to find 'dalil_ai' package
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Parent of main.py if main.py is in 'dalil_ai'
# If main.py is inside 'dalil_ai' folder, we need to add the parent to sys.path
# to import 'dalil_ai.ui' etc. correctly.
# Re-checking plan: default main.py was in dalil_ai/main.py.
sys.path.append(current_dir) # Add dalil_ai itself if modules are top-level there? 
# Wait, imports are `from dalil_ai.ui ...`. This implies `dalil_ai` is a package inside a root.
# If I run `python dalil_ai/main.py` from root, `dalil_ai` is a package.
# If I run `python main.py` from inside `dalil_ai`, it's weird.
# Let's assume user runs from root: `python dalil_ai/main.py`
# Then we need the root to be in sys.path.
sys.path.append(os.path.join(current_dir, ".."))

# Optional: Set PYTHONHOME/PYTHONPATH env vars if strictly requested by user custom envs
# os.environ["PYTHONPATH"] = ... 

from PySide6.QtWidgets import QApplication
from dalil_ai.ui.main_window import MainWindow

def main():
    # --- Windows Taskbar Icon Fix ---
    import ctypes
    myappid = 'dalilai.research.lab.1.0' # Arbitrary string
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass
    # --------------------------------

    app = QApplication(sys.argv)
    app.setApplicationName("Dalil AI")
    app.setOrganizationName("Deepmind-Agent")
    
    # Set Global Icon
    from PySide6.QtGui import QIcon
    
    # Robust Path Finding
    basedir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(basedir, "Dalil_ai.ico")
    
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        print(f"Warning: Icon not found at {icon_path}")
    
    # Create Main Window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
