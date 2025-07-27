import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ⚠️ Boston dataset is deprecated; use fetch_california_housing in future
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

# Show data info
print("Dataset shape:", df.shape)
print(df.head())

# Features and target
X = df.drop(columns='MEDV')
y = df['MEDV']

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R^2 score: {r2:.2f}")
