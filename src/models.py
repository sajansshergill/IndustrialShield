from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
logreg_grid = {
    'C': [0.1, 1],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
logreg = LogisticRegression(max_iter=500, random_state=42)
logreg_cv = GridSearchCV(logreg, logreg_grid, cv=3, scoring='accuracy', n_jobs=-1)
logreg_cv.fit(X_scaled_df, y)
print("✅ Best Params - Logistic Regression:", logreg_cv.best_params_)

# Random Forest (simplified)
rf_grid = {
    'n_estimators': [100],
    'max_depth': [10, None],
}
rf = RandomForestClassifier(random_state=42)
rf_cv = GridSearchCV(rf, rf_grid, cv=3, scoring='accuracy', n_jobs=-1)
rf_cv.fit(X_scaled_df, y)
print("✅ Best Params - Random Forest:", rf_cv.best_params_)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Define best params from previous GridSearchCV (replace with yours)
logreg_best_params = {'C': 1, 'penalty': 'l2', 'solver': 'lbfgs'}
rf_best_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2}

# Define tuned models
best_logreg = LogisticRegression(max_iter=1000, random_state=42, **logreg_best_params)
best_rf = RandomForestClassifier(random_state=42, **rf_best_params)

# Old default models
old_models = {
    "Logistic Regression (Default)": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest (Default)": RandomForestClassifier(random_state=42),
    "XGBoost (Default)": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Tuned models
new_models = {
    "Logistic Regression (Tuned)": best_logreg,
    "Random Forest (Tuned)": best_rf
}

# Evaluate models
results = []

for name, model in old_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))

for name, model in new_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))

# Results table
accuracy_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
accuracy_df = accuracy_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
print("\n📊 Accuracy Comparison Table:")
display(accuracy_df)


