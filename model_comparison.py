import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# 1. Загрузка данных
try:
    X = pd.read_csv('X_preprocessed.csv')
    y = pd.read_csv('y_labels.csv').values.ravel()
except FileNotFoundError:
    print("Ошибка: Сначала выполните предобработку данных.")
    exit()

# 2. Разделение на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Словарь моделей для сравнения
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, kernel='linear'),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# 4. Обучение и сбор метрик
results = []

print("Начинаю сравнение моделей...")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC-ROC": roc_auc_score(y_test, y_prob),
        "F1-Score": f1_score(y_test, y_pred)
    })

# 5. Преобразование результатов в таблицу
results_df = pd.DataFrame(results).sort_values(by="AUC-ROC", ascending=False)
print("\nИтоговая таблица сравнения:")
print(results_df)

# 6. Визуализация
plt.figure(figsize=(10, 6))
sns.barplot(x="AUC-ROC", y="Model", data=results_df, palette="viridis")
plt.title("Сравнение моделей по метрике AUC-ROC")
plt.xlim(0, 1.1)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()