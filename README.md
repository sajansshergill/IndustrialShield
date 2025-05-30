# IndustrialShield - Machine Learning-Based Intrusion Detection System (IDS) for Manufacturing Environments

# 🎯 Problem Statement:
Manufacturing systems are increasingly targeted by cyberattacks due to their reliance on legacy industrial control 
systems and limited real-time threat detection. The goal of this project is to build a machine learning-based IDS 
to detect intrusions in real-time using labeled network traffic data and apply it to manufacturing IT/OT infrastructure.

---------------------------------------------------------------------------------------------------------------

# Repo Structure:
industrialshield-ml-ids/
│
├── data/
│   ├── raw/                          # Original datasets (e.g., CICIDS2017, NSL-KDD)
│   ├── processed/                    # Cleaned & preprocessed datasets
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # EDA and initial insights
│   ├── 02_feature_engineering.ipynb  # Feature creation and transformation
│   ├── 03_model_training.ipynb       # ML model development and evaluation
│   ├── 04_dashboard_building.ipynb   # Dashboard integration and alerts
│
├── src/
│   ├── data_loader.py                # Data loading and preprocessing scripts
│   ├── model.py                      # Model training and prediction logic
│   ├── utils.py                      # Utility functions (e.g., metrics, logging)
│   ├── intrusion_detector.py         # Pipeline execution and orchestration
│
├── dashboard/
│   ├── app.py                        # Streamlit or Flask dashboard
│   ├── charts.py                     # Visual components
│
├── reports/
│   ├── README.md                     # Project summary and setup
│   ├── final_report.pdf              # Final documented report
│
├── requirements.txt                 # Required Python libraries
└── README.md                        # Main README for the project


---------------------------------------------------------------------------------------------------------------

# 🧠 Solution Approach:
1. Data Collection
- Download and use cybersecurity datasets:
- NSL-KDD or CICIDS2017 (for supervised learning)
- Simulate industrial IoT attack data using TON_IoT dataset if desired.
- Clean and balance the dataset.

2. Exploratory Data Analysis (EDA)
- Identify most frequent attack types.
- Visualize distributions of duration, protocol type, byte flow, etc.
- Correlate features to intrusion labels.

3. Feature Engineering
- One-hot encode categorical features like protocol and service.
- Normalize numerical features (duration, src_bytes).
- Create new features (e.g., bytes per second, session length).

4. Model Building
- Use algorithms like:
  I. Random Forest
  II. XGBoost
  III. Autoencoders (for anomaly detection)
  IV. Train/Test split, evaluate with:
  V. Accuracy
  VI. Precision, Recall, F1-Score
  VII. ROC-AUC curve

5. Dashboard Development
- Streamlit or Flask app to visualize:
  I. Real-time predictions
  II. Feature importance
  III. Intrusion severity score
  IV. Optional: Include log uploader or alert system via email

6. Documentation & Reporting
- Summarize architecture, model results, and business relevance.
- Include diagrams showing ML pipeline & threat detection flow.

---------------------------------------------------------------------------------------------------------------

# 🔧 Tools & Technologies:
- Languages: Python (NumPy, Pandas, Scikit-learn, Matplotlib, Streamlit)
- ML Libraries: XGBoost, LightGBM, Autoencoder (Keras/TensorFlow)
- Visualization: Seaborn, Plotly, Power BI (optional)
- Deployment: Streamlit/Flask (for dashboard)
- Data: NSL-KDD, CICIDS2017, TON_IoT datasets

---------------------------------------------------------------------------------------------------------------

# 🔚 Conclusion
In this project, we developed a machine learning-based Intrusion Detection System (IDS) tailored for Modbus network traffic in industrial IoT environments. Through comprehensive data preprocessing, feature analysis, and the training of multiple classification models—including Logistic Regression, Random Forest, and XGBoost—we evaluated their performance in detecting cyber-attacks. After tuning hyperparameters, Random Forest emerged as the best-performing model, achieving strong accuracy while maintaining interpretability. While XGBoost showed potential, resource constraints limited its deployment in this version.

This project not only reinforces the role of ML in strengthening cybersecurity for manufacturing setups but also highlights the importance of model explainability, evaluation, and ethical treatment of potential outliers. Future work can include model deployment, time-series attack detection, and explainable AI techniques like SHAP to further enhance trust and usability.
