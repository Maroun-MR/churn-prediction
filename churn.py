import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ── Load ──────────────────────────────────────────
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# ── Clean ─────────────────────────────────────────
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)
df = df.drop('customerID', axis=1)

# Convert binary columns
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# One-hot encoding
df = pd.get_dummies(df, columns=[
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract',
    'PaymentMethod'
])

# Convert all boolean columns to integers
df = df.astype({col: int for col in df.select_dtypes('bool').columns})

# Drop any remaining NaN rows
df = df.dropna()

print("Missing values:", df.isnull().sum().sum())
print("Shape:", df.shape)

# ── Split ─────────────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Models ────────────────────────────────────────
# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# XGBoost
xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# ── Results ───────────────────────────────────────
print("\n--- Model Comparison ---")
print(f"Logistic Regression: {accuracy_score(y_test, lr_pred)*100:.2f}%")
print(f"Random Forest:       {accuracy_score(y_test, rf_pred)*100:.2f}%")
print(f"XGBoost:             {accuracy_score(y_test, xgb_pred)*100:.2f}%")





# ── Visualizations ────────────────────────────────

# 1. Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn', palette='Blues')
plt.title('Churn Distribution')
plt.xticks([0, 1], ['Not Churned', 'Churned'])
plt.tight_layout()
plt.savefig('churn_distribution.png')
plt.show()

# 2. Model Accuracy Comparison
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
accuracies = [
    accuracy_score(y_test, lr_pred)*100,
    accuracy_score(y_test, rf_pred)*100,
    accuracy_score(y_test, xgb_pred)*100
]

plt.figure(figsize=(7, 4))
sns.barplot(x=models, y=accuracies, hue=models, palette='Blues_d', legend=False)

plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(70, 90)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# 3. Feature Importance (Random Forest)
feat_importance = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=False)[:10]

plt.figure(figsize=(8, 5))
feat_importance.plot(kind='bar', color='steelblue')
plt.title('Top 10 Most Important Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()