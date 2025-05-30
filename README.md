# 🛡️ IndustrialShield - ML-Based Intrusion Detection for Manufacturing IoT

## 🎯 Problem Statement
Industrial networks using protocols like Modbus are highly vulnerable to cyberattacks. This project develops a Machine Learning-based Intrusion Detection System (IDS) to classify and detect attacks from labeled Modbus traffic.

---

## 🧠 Solution Overview

### 📂 Project Structure
industrialshield-ml-ids/
├── data/ # Raw and processed datasets
├── notebooks/ # EDA, feature engineering, modeling
├── src/ # Python scripts for modeling and pipeline
├── dashboard/ # Streamlit dashboard (optional)
├── reports/ # Summary reports
├── requirements.txt # Python dependencies
└── README.md # Project overview


### 🔬 Workflow Steps
1. **Data Preprocessing**: Cleaned and scaled Modbus traffic features.
2. **EDA & Visualization**: Histograms, box plots, and correlations to detect patterns and outliers.
3. **Feature Engineering**: One-hot encoding, normalization, outlier consideration.
4. **Modeling**: Trained and compared Logistic Regression, Random Forest.  
   🔸 *Random Forest outperformed others after tuning.*  
   🔸 *XGBoost was excluded due to compatibility issues.*
5. **Evaluation**: Accuracy, Confusion Matrix, ROC-AUC Curve, and Learning Curves.
6. **Hyperparameter Tuning**: GridSearchCV used to improve accuracy and reduce overfitting.
7. **Conclusion**: Random Forest offered a balance of performance and interpretability.

---

## 🚀 How to Run
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/industrialshield-ml-ids.git
cd industrialshield-ml-ids

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebooks/03_model_training.ipynb

## 🛠 Technologies
Python (Pandas, Sklearn, XGBoost, Matplotlib, Seaborn)

Jupyter Notebook

Streamlit (optional)

Dataset: Train_Test_IoT_Modbus(in).csv

