import pandas as pd
from sklearn.datasets import fetch_california_housing, load_iris, load_wine, load_diabetes
import os

DATA_DIR = "examples_data"
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Downloading datasets to {DATA_DIR}...")

# 1. Housing (Regression)
print("Fetching California Housing...")
housing = fetch_california_housing()
df_housing = pd.DataFrame(housing.data, columns=housing.feature_names)
df_housing['MedHouseVal'] = housing.target
df_housing.to_csv(os.path.join(DATA_DIR, "housing.csv"), index=False)

# 2. Iris (Classification - Simple)
print("Fetching Iris...")
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['species'] = iris.target
df_iris['species'] = df_iris['species'].map({i: name for i, name in enumerate(iris.target_names)})
df_iris.to_csv(os.path.join(DATA_DIR, "iris.csv"), index=False)

# 3. Wine (Classification)
print("Fetching Wine...")
wine = load_wine()
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
df_wine['target'] = wine.target
df_wine.to_csv(os.path.join(DATA_DIR, "wine.csv"), index=False)

# 4. Diabetes (Regression)
print("Fetching Diabetes...")
diabetes = load_diabetes()
df_diab = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df_diab['target'] = diabetes.target
df_diab.to_csv(os.path.join(DATA_DIR, "diabetes.csv"), index=False)

print("Done! You can now load these files from the 'examples_data' folder.")
