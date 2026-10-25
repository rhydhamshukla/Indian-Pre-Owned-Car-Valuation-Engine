# 🚗 Capstone Major Project: Indian Pre Owned Car Valuation Engine
### **Unlox Academy : Machine Learning August 2026 Batch**
**Student Name:** Rhydham Shukla  
**Registered Email:** rhydhamshukla60@gmail.com  
**Course:** Machine Learning Program  
**Submission Portal:** [Unlox Academy](https://grow.unlox.com/v3/myaccount)  
**Submission Form:** [Google Form](https://forms.gle/cByNbannZ3k3PQtQ8)  
**GitHub Repository:** [https://github.com/rhydhamshukla/Indian-Pre-Owned-Car-Valuation-Engine](https://github.com/rhydhamshukla/Indian-Pre-Owned-Car-Valuation-Engine)

***

## 📌 Executive Summary
This repository contains the complete, production grade Capstone Major Project for Rhydham Shukla.  
The project develops an advanced predictive valuation engine for pre owned automobiles in Indian metropolitan markets based on **53,472 real vehicle sales**.

### Key Achievements and Verified Metrics:
* **Ensemble Architecture:** Stacking Ensemble blending **Random Forest + XGBoost + LightGBM** on log transformed targets.
* **Cross Validation Accuracy:** Empirical **R² = 0.9301** (vs Ridge Baseline R² = 0.5912).
* **Error Metrics:** **RMSE = ₹223,416**, **MAE = ₹108,843**, **MAPE = 11.83%**.
* **Test Set Scoring:** Evaluated all **5,615 unlabelled test instances** (`Cap_Test_Data_2025.csv`), outputting `Cap_Submission_Rhydham_Shukla.csv`.
* **Interactive Web App:** Full Streamlit valuation dashboard with single car prediction (with dynamic fair value confidence bands) and batch CSV scoring.
* **Academic Reports:** 
  1. `Rhydham_Shukla_Major_Project_Summary_1Page.pdf` (Strict 1 page executive summary matching Unlox guidelines)
  2. `Rhydham_Shukla_Major_Project_Report.pdf` (Full 3 page comprehensive technical report)

***

## 📁 Package Contents
* `Rhydham_Shukla_Car_Price_Prediction_Pipeline.ipynb`: Core end to end reproducible Jupyter/Colab notebook.
* `Rhydham_Shukla_Major_Project_Summary_1Page.pdf`: 1 page academic summary PDF.
* `Rhydham_Shukla_Major_Project_Report.pdf`: Full 3 page academic evaluation report with 6 analytical figures.
* `Cap_Submission_Rhydham_Shukla.csv`: 5,615 predicted car prices for the test competition set.
* `app.py`: Interactive Streamlit application code.
* `PROJECT_PRESENTATION_SCRIPT.md`: Timed presentation script for the video demonstration.
* `requirements.txt`: Python package dependencies.
