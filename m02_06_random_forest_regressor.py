import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

np.random.seed(42)

X = np.random.uniform(1, 10, size=(100, 1))
y = X.ravel() * 8 + np.random.normal(0, 5, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R2 score:{r2_score(y_test, y_pred):.4f}")