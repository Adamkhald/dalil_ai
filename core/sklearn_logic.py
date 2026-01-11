import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import io
import matplotlib.pyplot as plt

class SklearnPipeline:
    def __init__(self):
        self.df = None
        self.target_column = None
        self.feature_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.model_type = "Classification" # or "Regression"
        self.metrics = {}

    def load_data(self, file_path):
        """Loads data from CSV or Excel."""
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            self.df = pd.read_excel(file_path)
        return self.df.head()

    def get_columns(self):
        if self.df is not None:
            return self.df.columns.tolist()
        return []

    def preprocess_data(self, target_col, ignored_cols=None, split_ratio=0.2, impute_strategy='mean', scale_data=True):
        """
        Preprocesses data: handles missing values, encoding, scaling, and splitting.
        """
        if self.df is None:
            raise ValueError("No data loaded.")
        
        self.target_column = target_col
        
        # Filter columns
        cols = [c for c in self.df.columns if c != target_col and (ignored_cols is None or c not in ignored_cols)]
        self.feature_columns = cols
        
        X = self.df[cols].copy()
        y = self.df[target_col].copy()

        # Handle Missing Values
        # Numeric columns
        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            imputer = SimpleImputer(strategy=impute_strategy)
            X[num_cols] = imputer.fit_transform(X[num_cols])
        
        # Categorical columns - Simple label encoding for now
        cat_cols = X.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            
        # Target Encoding if categorical
        if y.dtype == object or self.model_type == "Classification":
             le_y = LabelEncoder()
             y = le_y.fit_transform(y)

        # Splitting
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)

        # Scaling
        if scale_data:
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            
        return f"Data Preprocessed. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}"

    def select_model(self, model_name, task_type="Classification"):
        self.model_type = task_type
        if task_type == "Classification":
            if model_name == "Logistic Regression":
                self.model = LogisticRegression()
            elif model_name == "Random Forest":
                self.model = RandomForestClassifier()
            elif model_name == "SVM":
                self.model = SVC()
        else: # Regression
            if model_name == "Linear Regression":
                self.model = LinearRegression()
            elif model_name == "Random Forest":
                self.model = RandomForestRegressor()
            elif model_name == "SVR":
                self.model = SVR()
        
        return f"Model selected: {model_name} ({task_type})"

    def train_model(self, hyperparams=None):
        if self.model is None or self.X_train is None:
            raise ValueError("Model or data not ready.")
        
        if hyperparams:
            try:
                self.model.set_params(**hyperparams)
            except Exception as e:
                return f"Error setting params: {str(e)}"
        
        self.model.fit(self.X_train, self.y_train)
        return "Training Completed."

    def evaluate_model(self):
        if self.model is None or self.X_test is None:
            raise ValueError("Model not trained.")
        
        preds = self.model.predict(self.X_test)
        
        results = {}
        if self.model_type == "Classification":
            acc = accuracy_score(self.y_test, preds)
            results["Accuracy"] = f"{acc:.4f}"
            self.metrics = results
            report = classification_report(self.y_test, preds, output_dict=True)
            return results, report
        else:
            mse = mean_squared_error(self.y_test, preds)
            r2 = r2_score(self.y_test, preds)
            results["MSE"] = f"{mse:.4f}"
            results["R2 Score"] = f"{r2:.4f}"
            self.metrics = results
            return results, None

    def plot_results(self):
        """Returns a matplotlib figure for visualization."""
        if self.model is None:
            return None
            
        fig, ax = plt.subplots(figsize=(6, 4))
        
        if self.model_type == "Regression":
            preds = self.model.predict(self.X_test)
            ax.scatter(self.y_test, preds, alpha=0.7)
            ax.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted')
        elif self.model_type == "Classification":
            # Simple feature importances if available
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[-10:] # Top 10
                ax.barh(range(len(indices)), importances[indices], align='center')
                # We lost feature names in array conversion, just using indices for now or basic logic
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels(indices)
                ax.set_xlabel('Relative Importance')
                ax.set_title('Feature Importances')
            else:
                ax.text(0.5, 0.5, "Visualization not available\nfor this model type.", 
                        horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.tight_layout()
        return fig

    def generate_code(self):
        """Generates a standalone Python script for the current pipeline."""
        if self.model is None:
            return "# No model trained yet."
            
        model_name = self.model.__class__.__name__
        impute_strategy = "mean" # Simplification: should track this self state
        scale = True
        
        # Build the script string
        code = f'''import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# 1. Load Data
# Replace with your actual file path
file_path = "data.csv" 
try:
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
except Exception:
    print("Error loading file. Please update file_path.")
    df = pd.DataFrame() # Dummy

if not df.empty:
    target_col = "{self.target_column}"
    feature_cols = {self.feature_columns}
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # 2. Preprocessing
    # Impute
    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        imputer = SimpleImputer(strategy="{impute_strategy}")
        X[num_cols] = imputer.fit_transform(X[num_cols])
        
    # Encode Categorical
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    # Encode Target
    if y.dtype == object:
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. Model Training
    print(f"Training {model_name}...")
    model = {model_name}()
    model.fit(X_train, y_train)

    # 4. Evaluation
    preds = model.predict(X_test)
    
    if "{self.model_type}" == "Classification":
        acc = accuracy_score(y_test, preds)
        print(f"Accuracy: {{acc:.4f}}")
    else:
        mse = mean_squared_error(y_test, preds)
        print(f"MSE: {{mse:.4f}}")

    print("Success! Model trained and evaluated.")
'''
        return code
