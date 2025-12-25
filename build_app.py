import PyInstaller.__main__
import os
import shutil
import sys

def build():
    print("🚀 Starting Build for Dalil AI...")
    
    # Parse exclusions
    # Usage: python build_app.py --no-torch --no-tf
    no_torch = '--no-torch' in sys.argv
    no_tf = '--no-tf' in sys.argv
    no_cv = '--no-cv' in sys.argv
    
    if no_torch: print("👉 Excluding PyTorch")
    if no_tf: print("👉 Excluding TensorFlow")
    if no_cv: print("👉 Excluding MediaPipe/OpenCV")
    
    # 1. Clean previous builds
    if os.path.exists("build"): shutil.rmtree("build")
    if os.path.exists("dist"): shutil.rmtree("dist")
    
    # 2. Base PyInstaller arguments
    args = [
        'main.py',                       # Entry point
        '--name=DalilAI',                # Name of the executable
        '--noconsole',                   # Windowed mode (no cmd popup)
        '--clean',                       # Clean cache
        '--onedir',                      # Folder output
        '--icon=Dalil_ai.ico',           # Icon for the .exe file
        '--add-data=Dalil_ai.ico;.',     # Include icon in the dist folder for runtime
        
        # Base Imports
        '--hidden-import=pandas',
        '--hidden-import=sklearn',
        '--hidden-import=sklearn.utils._typedefs',
        '--hidden-import=sklearn.neighbors._partition_nodes',
        '--hidden-import=scipy.special.cython_special',
        '--hidden-import=matplotlib',
        '--hidden-import=PySide6',
        
        '--exclude-module=tkinter',
        '--exclude-module=jupyter',
        '--exclude-module=notebook',
    ]
    
    # 3. Conditional Hidden Imports
    if not no_torch:
        args.append('--hidden-import=torch')
        args.append('--hidden-import=torchvision')
        
    if not no_tf:
        args.append('--hidden-import=tensorflow')
        args.append('--hidden-import=keras')
        
    if not no_cv:
        args.append('--hidden-import=mediapipe')
        args.append('--hidden-import=cv2')
    
    print(f"📦 Packaging with {len(args)} args...")
    
    # 4. Run PyInstaller
    try:
        PyInstaller.__main__.run(args)
        print("\n✅ Build Successful!")
        print("📂 Executable is located in: dist/DalilAI/")
    except Exception as e:
        print(f"\n❌ Build Failed: {e}")

if __name__ == "__main__":
    build()
