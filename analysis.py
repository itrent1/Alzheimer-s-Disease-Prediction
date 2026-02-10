import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

def visualize_normalized_importance(X, y):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM (Linear)": SVC(probability=True, kernel='linear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            continue

        importances = (importances / np.sum(importances)) * 100

        feat_imp = pd.DataFrame({'Feature': X.columns, 'Importance (%)': importances})
        feat_imp = feat_imp.sort_values(by='Importance (%)', ascending=False)

        sns.barplot(ax=axes[i], x='Importance (%)', y='Feature', data=feat_imp, palette='viridis')
        axes[i].set_title(f'{name}', fontsize=14)
        axes[i].set_xlabel('Вклад признака в предсказание (%)')
        axes[i].set_xlim(0, 100) 
        axes[i].set_ylabel('')

    plt.tight_layout()
    plt.savefig('normalized_importance_100.png')
    plt.show()

try:
    X = pd.read_csv('X_preprocessed.csv')
    y = pd.read_csv('y_labels.csv').values.ravel()
    visualize_normalized_importance(X, y)
except FileNotFoundError:
    print("Ошибка: Сначала выполните предобработку данных.")