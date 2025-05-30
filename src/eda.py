import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# List of Modbus features
features = ['FC1_Read_Input_Register', 'FC2_Read_Discrete_Value',
            'FC3_Read_Holding_Register', 'FC4_Read_Coil']

# Plot histograms for each feature
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(data=df, x=feature, hue='label', bins=50, kde=False, stat="count", element="step")
    plt.title(f'Histogram of {feature} by Label')
    plt.xlabel(feature)
    plt.ylabel("Count")

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(12, 10))

# List of features
features = ['FC1_Read_Input_Register', 'FC2_Read_Discrete_Value',
            'FC3_Read_Holding_Register', 'FC4_Read_Coil']

# Create subplots for box plots
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='label', y=feature, data=df)
    plt.title(f'Box Plot of {feature} by Label')
    plt.xlabel('Label (0 = Normal, 1 = Attack)')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Feature matrix and target
X = df[['FC1_Read_Input_Register', 'FC2_Read_Discrete_Value',
        'FC3_Read_Holding_Register', 'FC4_Read_Coil']]
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit random forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importances = rf.feature_importances_
features = X.columns

# Plot
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=features)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Dictionary of models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Accuracy and classification report
    acc = accuracy_score(y_test, y_pred)
    print(f"🔍 Accuracy ({name}): {acc:.4f}")
    print(f"📋 Classification Report ({name}):\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# If not already a DataFrame
train_test_df = pd.DataFrame(training_results)

# Extract data
models = train_test_df["Model"]
train_acc = train_test_df["Training Accuracy"]
test_acc = train_test_df["Testing Accuracy"]
gap = train_test_df["Gap (Train - Test)"]

# Set bar width and positions
x = np.arange(len(models))
width = 0.35

# Plot: Training vs. Testing Accuracy
plt.figure(figsize=(10, 6))
plt.bar(x - width/2, train_acc, width, label='Train Accuracy')
plt.bar(x + width/2, test_acc, width, label='Test Accuracy')

plt.ylabel('Accuracy')
plt.title('Training vs. Testing Accuracy per Model')
plt.xticks(x, models)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()

# Plot: Accuracy Gap
plt.figure(figsize=(10, 4))
plt.bar(models, gap, color='orange')
plt.title('Gap Between Training and Testing Accuracy (Overfitting Check)')
plt.ylabel('Train - Test Accuracy')
plt.xlabel('Model')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.tight_layout()
plt.show()


# 📦 Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ✅ Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# ✅ Learning Curve Plot Function
def plot_learning_curve(name, estimator, X, y):
    from sklearn.model_selection import learning_curve

    plt.figure(figsize=(8, 5))
    plt.title(f"Learning Curve: {name}")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    train_scores_mean = train_scores.mean(axis=1)
    test_scores_mean = test_scores.mean(axis=1)

    plt.plot(train_sizes, train_scores_mean, 'o-', label="Train Accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', label="Validation Accuracy")
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

# ✅ ROC Curve Function
def plot_roc_curve(models_dict, X_test, y_test):
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(8, 6))
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve and AUC")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ✅ Train models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_scaled_df, y)

# ✅ Plot Learning Curves (skip XGBoost to avoid compatibility error)
for name, model in models.items():
    if name != "XGBoost":  # Skip XGBoost due to sklearn compatibility
        plot_learning_curve(name, model, X_scaled_df, y)

# ✅ Plot ROC + AUC Curve for all
plot_roc_curve(models, X_test, y_test)


