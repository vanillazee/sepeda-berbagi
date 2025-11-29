# Bike Sharing Demand Prediction – Machine Learning Capstone Project

This project focuses on building a Machine Learning model to predict hourly bike rental demand using the Bike Sharing Dataset (Fanaee-T & Gama, UCI ML Repository). The goal is to help bike-sharing operators allocate bikes more efficiently based on temporal patterns, weather conditions, and environmental factors.

The dataset contains hourly usage logs from a bike-sharing system in Washington, D.C., covering the years 2011–2012. Each record includes information such as temperature, humidity, weather situation, hour of the day, and total rental count (`cnt`). This makes the dataset particularly suitable for regression modeling and behavioral pattern analysis.

## Project Objectives
- Predict hourly total bike rentals (`cnt`) using regression-based machine learning.
- Understand how time, weather, and environmental factors drive bike usage.
- Evaluate multiple models (Baseline, Linear Regression, Random Forest, Gradient Boosting).
- Select and save the best-performing model for deployment.

## Why This Project Matters
Bike-sharing systems rely heavily on real-time and short-term demand forecasting. Accurate predictions help operators:
- prevent bike shortages during peak commuting hours,
- reduce idle bikes during low-demand periods,
- optimize redistribution strategy,
- improve service reliability and user experience.

A reliable model allows operators to plan ahead using expected demand instead of reacting to sudden spikes or drops.

## Workflow Summary
1. **Business Understanding**  
   Define the problem, goals, and evaluation metrics (RMSE, MAE, R²).

2. **Data Understanding**  
   Explore dataset structure, missing values, and descriptive statistics.

3. **Exploratory Data Analysis (EDA)**  
   Analyze hourly usage patterns, weekday-weekend behavior, seasonal trends, and weather impacts.

4. **Preprocessing & Feature Engineering**  
   - Convert date to calendar features  
   - Encode cyclical hour using sin/cos  
   - One-hot encode weather & season  
   - Remove leakage features (`casual`, `registered`)

5. **Modeling & Evaluation**  
   Train baseline → regression → tree-based models.  
   Tune Random Forest using RandomizedSearchCV.  
   Select final model based on RMSE and R².

6. **Model Saving**  
   Save the final tuned Random Forest model (`best_rf`) using pickle/joblib for deployment and reproducibility.

## Final Result
The tuned Random Forest model achieved strong predictive performance, capturing both nonlinear interactions and temporal patterns in the dataset.  
This model is saved in the repository as:

