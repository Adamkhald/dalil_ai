
# Detailed Educational Content for Dalil AI

EDUCATION_DATA = {
    "Libraries": {
        "Scikit-Learn": """
<h1>Scikit-Learn</h1>
<h3>The Industry Standard for Classical Machine Learning</h3>
<p><b>Scikit-learn</b> (sklearn) is the most popular Python library for traditional machine learning algorithms. Built on top of NumPy, SciPy, and matplotlib, it is designed to be simple and efficient for data mining and data analysis.</p>

<h3>Key Features:</h3>
<ul>
    <li><b>Simple API:</b> Uses a consistent `fit()`, `transform()`, and `predict()` interface across almost all algorithms.</li>
    <li><b>Preprocessing:</b> Large suite of tools for feature extraction and normalization (e.g., StandardScaler, OneHotEncoder).</li>
    <li><b>Model Selection:</b> Tools for Grid Search, Cross-Validation, and metrics (Accuracy, F1-Score, MSE).</li>
</ul>

<h3>When to use it?</h3>
<p>Use Scikit-learn for almost all tabular data problems (rows and columns) where you don't need Deep Learning. It is perfect for Regression, Classification, and Clustering on structured data.</p>
""",
        "PyTorch": """
<h1>PyTorch</h1>
<h3>Deep Learning for Researchers</h3>
<p>Developed by Meta (Facebook) AI, <b>PyTorch</b> is a deep learning framework known for its flexibility and ease of use. It uses <i>dynamic computation graphs</i>, meaning you can modify the behavior of your neural network on the fly during execution.</p>

<h3>Why is it popular?</h3>
<ul>
    <li><b>Pythonic:</b> It feels like writing standard Python code. Debugging is easy.</li>
    <li><b>Research Favorite:</b> Most academic papers today publish code in PyTorch.</li>
    <li><b>GPU Acceleration:</b> Seamlessly moves tensors to CUDA (NVIDIA GPUs) for massive speedups.</li>
</ul>
""",
        "TensorFlow": """
<h1>TensorFlow</h1>
<h3>Production-Grade Deep Learning</h3>
<p>Created by Google, <b>TensorFlow</b> is an end-to-end open source platform for machine learning. While PyTorch is favored in research, TensorFlow (especially with Keras) is often favored in industry deployment.</p>

<h3>Key Features:</h3>
<ul>
    <li><b>Keras Integration:</b> High-level API that makes building networks incredibly fast and easy.</li>
    <li><b>TFX (TensorFlow Extended):</b> A full production platform for managing ML pipelines.</li>
    <li><b>Deployment:</b> TensorFlow Lite (Mobile), TensorFlow.js (Web), and TensorFlow Serving (Cloud) are best-in-class.</li>
</ul>
"""
    },
    "Supervised Learning (Regression)": {
        "Linear Regression": """
<h1>Linear Regression</h1>
<h3>The Foundation of Prediction</h3>
<p>Linear Regression is the simplest form of machine learning. It assumes a linear relationship between the input variables (X) and the single output variable (y).</p>

<h3>How it works</h3>
<p>The model attempts to learn a line <b>y = wx + b</b> (where <i>w</i> is the weight/slope and <i>b</i> is the intercept) that minimizes the error between prediction and actual data.</p>

<h3>Mathematical objective</h3>
<p>It minimizes the <b>Residual Sum of Squares (RSS)</b>: the sum of the squared differences between the observed value and the predicted value.</p>

<h3>Pros & Cons</h3>
<ul>
    <li>✅ Extremely fast and interpretable.</li>
    <li>✅ Works well on small, simple datasets.</li>
    <li>❌ Cannot model complex, non-linear relationships.</li>
    <li>❌ Sensitive to outliers.</li>
</ul>
""",
        "Support Vector Regression (SVR)": """
<h1>Support Vector Regression (SVR)</h1>
<h3>Robust Regression</h3>
<p>SVR uses the same principles as SVM (for classification) but for regression. Instead of minimizing error directly, it tries to fit the error within a certain threshold (epsilon).</p>

<h3>The "Kernel Trick"</h3>
<p>SVR can easily solve non-linear problems by mapping data into higher dimensions using kernels (RBF, Polynomial). This allows it to fit curves to data that looks linear in higher space.</p>

<h3>Use Cases</h3>
<p>Stock price prediction, time-series forecasting where robustness to outliers is needed.</p>
""",
        "Random Forest Regressor": """
<h1>Random Forest Regressor</h1>
<h3>The Power of Ensembles</h3>
<p>A Random Forest creates thousands of "Decision Trees" on random subsets of the data and averages their predictions.</p>

<h3>Wisdom of the Crowds</h3>
<p>A single decision tree is prone to "overfitting" (memorizing the data). By averaging many uncorrelated trees, the Random Forest drastically reduces variance and provides a stable, accurate prediction.</p>

<h3>Pros</h3>
<p>It handles non-linear data well, requires very little data preprocessing (no scaling needed), and gives feature importance scores.</p>
"""
    },
    "Supervised Learning (Classification)": {
         "Logistic Regression": """
<h1>Logistic Regression</h1>
<h3>The Binary Classifier</h3>
<p>Despite the name, this is used for <b>Classification</b> (Yes/No, Spam/Ham). It predicts the <i>probability</i> that an instance belongs to a class.</p>

<h3>The Sigmoid Function</h3>
<p>It outputs a value between 0 and 1 by passing the linear equation through a Sigmoid (S-shaped) function. If the probability > 0.5, the class is 1.</p>
""",
        "Support Vector Machines (SVM)": """
<h1>Support Vector Machines (SVM)</h1>
<h3>The Margin Maximizer</h3>
<p>SVM finds the "hyperplane" (boundary) that best separates the two classes. Uniquely, it focuses effectively on the data points near the boundary (Support Vectors).</p>

<h3>Key Concept: Margin</h3>
<p>SVM doesn't just separate data; it tries to create the widest possible "street" (margin) between the classes. This makes it generalize very well to new data.</p>
""",
        "K-Nearest Neighbors (KNN)": """
<h1>K-Nearest Neighbors (KNN)</h1>
<h3>The Lazy Learner</h3>
<p>KNN does not "learn" a model. Instead, it stores the entire training dataset.</p>

<h3>How it predicts</h3>
<p>When asked to predict a new point, it looks at the 'K' closest points in the training set. If K=3, and 2 neighbors are Red and 1 is Blue, it predicts Red.</p>

<h3>Pros & Cons</h3>
<ul>
    <li>✅ Very simple concept. No training time.</li>
    <li>❌ Very slow at prediction time (must search all points).</li>
    <li>❌ Sensitive to the scale of data (must normalize inputs).</li>
</ul>
"""
    },
    "Deep Learning Models": {
        "Artificial Neural Networks (ANN)": """
<h1>Artificial Neural Network (ANN)</h1>
<h3>Mimicking the Brain</h3>
<p>Also known as Multi-Layer Perceptrons (MLP). ANNs consist of layers of interconnected "neurons".</p>

<h3>Structure</h3>
<ul>
    <li><b>Input Layer:</b> Receives the raw data.</li>
    <li><b>Hidden Layers:</b> Layers where math operations occur. Each connection has a "weight".</li>
    <li><b>Output Layer:</b> Produces the final prediction.</li>
    <li><b>Activation Functions:</b> (ReLU, Sigmoid) introduce non-linearity, allowing the network to learn complex patterns.</li>
</ul>
""",
        "Convolutional Neural Networks (CNN)": """
<h1>Convolutional Neural Networks (CNN)</h1>
<h3>The Visionary</h3>
<p>Standard ANNs fail at images because they treat every pixel as independent. CNNs understand <i>spatial structure</i>.</p>

<h3>How it works</h3>
<p>It slides small filters (kernels) across the image to detect features like edges, corners, and textures. As you go deeper, it combines these to detect eyes, wheels, and eventually faces or cars.</p>
""",
        "Recurrent Neural Networks (RNN/LSTM)": """
<h1>Recurrent Neural Networks (RNN)</h1>
<h3>The Memory Master</h3>
<p>Standard networks assume inputs are independent. RNNs are designed for <b>sequences</b> (Time Series, Text, Audio).</p>

<h3>The Loop</h3>
<p>An RNN has a loop inside it. It passes the output of the previous step as an input to the current step. This gives it "memory" or context.</p>

<p><b>LSTM (Long Short-Term Memory):</b> A special kind of RNN that solves the "vanishing gradient" problem, allowing it to remember things from thousands of steps ago.</p>
"""
    },
     "Reinforcement Learning": {
        "Q-Learning": """
<h1>Q-Learning</h1>
<h3>Value-Based Learning</h3>
<p>A tabular method where the agent learns a "Q-Table". The table tells the agent: "In State S, if I take Action A, what is the expected future reward?"</p>
""",
        "Deep Q-Networks (DQN)": """
<h1>Deep Q-Networks (DQN)</h1>
<h3>From Tables to Brains</h3>
<p>Q-Learning fails when the world is too big for a table (like pixels in a video game). DQN replaces the table with a Neural Network that <i>approximates</i> the Q-values.</p>
""",
        "PPO (Proximal Policy Optimization)": """
<h1>PPO</h1>
<h3>The Stable Guardian</h3>
<p>Policy Gradient methods try to learn the "Policy" (strategy) directly. PPO is famous for being stable and reliable.</p>
<p>It uses a clever mathematical trick (clipping) to prevent the agent from changing its behavior too drastically in a single update, preventing "catastrophic forgetting".</p>
"""
    }
}
