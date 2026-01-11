# 🧠 Dalil AI - The Offline Research Lab
<p align="center">
  <img src="docs/logo.png" alt="Dalil AI Logo" width="120">
</p>

**Dalil AI** is a powerful, offline-first desktop platform that democratizes access to Machine Learning and Deep Learning. Built with **PySide6**, it integrates the world's best ML ecosystems into a single, unified interface. No cloud subscription required—your data stays on your machine.

---

## 🚀 Key Features

### 🧠 TensorFlow Deep Learning Hub (New!)
The most versatile module in the platform, capable of handling almost any data type:
- **5 Dynamic Modes**:
    1.  **Image Classification**: MobileNetV2 / ResNet50 Transfer Learning.
    2.  **Tabular Regression**: Predict numerical values from CSV.
    3.  **Tabular Classification**: Predict categories from CSV.
    4.  **Text Classification**: Sentiment analysis with LSTM networks.
    5.  **Time Series Forecasting**: Predict the future using sequence models.
- **Advanced Training**: Select your **Optimizer** (Adam, SGD, RMSprop), tune **Learning Rates**, and view **Live Matplotlib Plots** of your training history.
- **Analysis**: Visualize Time Series predictions vs Actual values immediately after training.
- **Production Ready**: Export any model to `.tflite` for mobile deployment.

### 🔬 Scikit-Learn Pipeline
- **7-Step Wizard**: A guided path from Data Loading to Model Export.
- **Built-in Datasets**: One-click load for MNIST, Iris, Wine, and more.
- **Auto-Preprocessing**: Smart imputation, One-Hot Encoding, and StandardScaler.
- **PDF Reporting**: Generate professional research reports with a single click.

### 📚 Education Center 2.0
- **Rich Content**: Interactive, HTML-based guides for every major algorithm.
- **Library Guide**: Deep dives into Scikit-Learn, PyTorch, and TensorFlow concepts.
- **Theory & Code**: Learn the math and the implementation side-by-side.

### 🔥 PyTorch Lab
- **No-Code Training**: Drag-and-drop training for Image Classification.
- **Visual Validation**: See your model's predictions in a grid view.

### 🎮 RL Studio (Reinforcement Learning)
- **Train Agents**: Watch PPO, DQN, and SAC agents learn to play games.
- **Environments**: LunarLander, CartPole, BipedalWalker.
- **Code Export**: Generate standalone training scripts for your experiments.

### 📸 MediaPipe Vision
- **Real-Time AI**: Face Mesh, Hand Tracking, and Pose Estimation running at 30+ FPS on CPU.

---

## 🛠️ Installation

### Option 1: Source (Recommended)
This uses your local Python environment.

```bash
# 1. Clone the repo
git clone https://github.com/Adamkhald/dalil_ai.git
cd dalil_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

### Option 2: Windows Launcher
Simply double-click `run_windows.bat` in the root folder.

---

## 📜 License
MIT License - Free for research, education, and academic use.
