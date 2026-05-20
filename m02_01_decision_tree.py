import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42,
)
model.fit(X_train, y_train)

print(f"정확도:{model.score(X_test, y_test):.4f}")

importance = pd.Series(
    model.feature_importances_,
    index=iris.feature_names,
).sort_values(ascending=False)

print("특성 중요도")
print(importance)