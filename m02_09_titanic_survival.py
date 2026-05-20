import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df = sns.load_dataset("titanic")

print("데이터 크기:", df.shape)
print(df.head())

print("결측치")
print(df.isnull().sum())

print("생존 비율")
print(df["survived"].value_counts(normalize=True))

features = ["pclass", "sex", "age", "sibsp", "parch", "fare"]
df_model = df[features + ["survived"]].copy()

# 결측치 처리
df_model["age"] = df_model["age"].fillna(df_model["age"].median())
df_model["fare"] = df_model["fare"].fillna(df_model["fare"].median())

# 문자 데이터를 숫자로 변환
df_model["sex"] = df_model["sex"].map({"male": 0, "female": 1})

X = df_model[features]
y = df_model["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"정확도:{accuracy_score(y_test, y_pred):.4f}")
print("분류 리포트")
print(classification_report(y_test, y_pred, target_names=["사망", "생존"]))

importance = pd.Series(
    model.feature_importances_,
    index=features,
).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importance.plot(kind="barh", color="steelblue")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("titanic_importance.png")
plt.show()

new_passenger = pd.DataFrame({
    "pclass": [3],
    "sex": [0],
    "age": [25],
    "sibsp": [0],
    "parch": [0],
    "fare": [7.5],
})

prediction = model.predict(new_passenger)
proba = model.predict_proba(new_passenger)

result = "생존" if prediction[0] == 1 else "사망"

print(f"새 승객 예측:{result}")
print(f"생존 확률:{proba[0][1]:.1%}")