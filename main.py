# --- Prerequisites ---
# pip install pandas scikit-learn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings


warnings.filterwarnings('ignore')


try:
    
    df = pd.read_csv('cleaned_crop_dataset.csv')
    print(" Cleaned dataset loaded successfully.")
except FileNotFoundError:
    print(" Error: 'cleaned_crop_dataset.csv' not found. Please ensure the file is in the correct folder.")
    exit()


X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(" Data prepared and scaled.")
print("-" * 50)


print(" Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, lr_preds)
print(" Logistic Regression training complete.")


print(" Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(" Random Forest training complete.")
print("-" * 50)


print("\n" + "="*20 + " MODEL COMPARISON " + "="*20)

print(f"\n Accuracy:")
print(f"  - Logistic Regression: {lr_accuracy:.4f} ({lr_accuracy:.2%})")
print(f"  - Random Forest:       {rf_accuracy:.4f} ({rf_accuracy:.2%})")

rf_importances = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)

print("\n Model Interpretability:")
print("\n--- Random Forest Feature Importances ---")
print("(Shows which environmental factors were most predictive overall for choosing a crop)")
print(rf_importances)

print("\n" + "="*58)


print("\n🔬 Analysis:")
if rf_accuracy > lr_accuracy + 0.05:
    print("The Random Forest model is significantly more accurate. This strongly suggests that the relationships between soil nutrients, climate, and the best crop to grow are complex and non-linear. Random Forest excels at finding these patterns, making it the superior choice for this task.")
else:
    print("Both models perform similarly. This indicates the relationships in your data might be more linear than expected. However, given its high performance, Random Forest remains an excellent and robust choice.")