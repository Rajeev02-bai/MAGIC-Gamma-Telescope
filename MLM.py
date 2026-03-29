import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

magic_gamma_telescope = fetch_ucirepo(id=159)
X = magic_gamma_telescope.data.features
y = magic_gamma_telescope.data.targets

print("Dataset Shape:", X.shape)
print("\nFeature Names:\n", X.columns.tolist())
print("\nClass Distribution:\n", y.value_counts())

y = y.squeeze()
y = (y == 'g').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred      = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

print("\n── Model Evaluation ──")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_pred_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Hadron (0)', 'Gamma (1)']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Hadron', 'Gamma'],
            yticklabels=['Hadron', 'Gamma'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc_score   = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – MAGIC Gamma Telescope')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

coeff_df = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

print("\nFeature Coefficients:\n", coeff_df.to_string(index=False))
    
plt.figure(figsize=(8, 5))
sns.barplot(data=coeff_df, x='Coefficient', y='Feature',
            palette='coolwarm', hue='Feature', legend=False)
plt.title('Logistic Regression – Feature Coefficients')
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.show()