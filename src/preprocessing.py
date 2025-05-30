import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ⚙️ Preprocessing & Model Building
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# Utility
import warnings
warnings.filterwarnings('ignore')


from sklearn.preprocessing import StandardScaler

# Define features and target
X = df[['FC1_Read_Input_Register', 'FC2_Read_Discrete_Value',
        'FC3_Read_Holding_Register', 'FC4_Read_Coil']]
y = df['label']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

# Optional: convert back to DataFrame for readability
import pandas as pd
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Preview the scaled data
X_scaled_df.head()


from sklearn.model_selection import train_test_split

# Use the scaled feature set and the original labels
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
)

# Display the shapes to confirm
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


