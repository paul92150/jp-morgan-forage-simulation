# JP Morgan Quantitative Research Simulation – Solutions

This repository contains four end-to-end implementations of tasks from the J.P. Morgan Forage quantitative research simulation. The code is written in Python and structured for clarity, reusability, and alignment with real-world quant research workflows.

## Task 1 – Gas Price Forecasting

Goal: Forecast natural gas prices over the next 18 months using historical data.

- Used linear regression to extract the trend component of gas prices.
- Computed seasonal monthly deviations (residuals) to capture cyclical effects.
- Combined trend and seasonality to produce a monthly forecast.
- Applied Hermite cubic spline interpolation to obtain a smooth, differentiable price function usable in later pricing models.

File: task1_gas_forecasting.py  
Data: data/Nat_Gas.csv

## Task 2 – Gas Storage Contract Pricing

Goal: Compute the net value of a natural gas storage contract under physical and financial constraints.

- Simulates daily injection/withdrawal decisions under rate/capacity limits.
- Uses price forecast from Task 1 to evaluate real-time gas values.
- Applies storage cost and computes daily cash flows and the net present value.

File: task2_gas_storage_pricing.py

## Task 3 – Credit Risk Modeling

Goal: Predict Expected Loss (EL) for borrowers using a custom financial loss metric.

- Trained a logistic regression model with custom Expected Loss objective.
- Tuned hyperparameters using Optuna with financial impact–aligned loss.
- Output: robust EL prediction function usable in production-like workflows.

File: task3_expected_loss_model.py  
Data: data/Task 3 and 4_Loan_Data.csv

## Task 4 – FICO Score Bucketing

Goal: Optimize FICO score buckets to improve default prediction using log-likelihood.

- Implemented two approaches:
  1. Dynamic Programming – Guarantees optimal cutoffs but scales poorly.
  2. Hybrid Optuna-Guided Search – Fast, approximate, and scalable.
- Visualized default rates across buckets.

File: task4_fico_bucketing.py

## Repository Structure

.
├── data/
│   ├── Nat_Gas.csv
│   └── Task 3 and 4_Loan_Data.csv
├── task1_gas_forecasting.py
├── task2_gas_storage_pricing.py
├── task3_expected_loss_model.py
├── task4_fico_bucketing.py
└── README.md

## Requirements

The core libraries used are:

- pandas
- numpy
- matplotlib
- scikit-learn
- optuna

## Author

Paul Lemaire  
Simulation participant | Quantitative research enthusiast | CentraleSupélec

## Notes

This repository is for educational and demonstration purposes only. It showcases structured, production-ready approaches to common quantitative research challenges in trading and risk management.
