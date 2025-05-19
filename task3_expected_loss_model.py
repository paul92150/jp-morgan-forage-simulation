"""
Submission Note:

- I explored a range of models, including Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM.
- I initially used classification-focused metrics (accuracy, AUC) but realized that minimizing classification error alone would not fully capture the business objective: minimizing financial loss.
- Therefore, I defined a custom loss function: the average absolute difference between the predicted Expected Loss (based on Probability of Default * 90% * loan amount) and the realized loss. This metric better aligns with the real financial risk to the bank.
- Using Optuna, I performed automated hyperparameter tuning across models, minimizing this Expected Loss error rather than pure classification accuracy.
- I observed that Logistic Regression, when properly tuned, consistently delivered the best balance between simplicity, interpretability, and performance.
- While exploring Logistic Regression, I tuned the regularization parameter C and observed that increasing C systematically reduced Expected Loss error, indicating that the model benefitted from reduced regularization.
- Based on these observations, I trained a Logistic Regression model without regularization to fully capture the relationships in the data.
- To ensure robustness, I performed multiple 80/20 train-test splits and cross-validation checks, verifying that the model generalizes well and does not overfit.
- Finally, after validating the approach, I retrained the final model on 100% of the data to maximize available information and predictive power.
- The final function provided below uses the intercept, coefficients, and feature standardization parameters learned from the full dataset to predict Expected Loss for any new borrower.

This approach ensures both careful model exploration and selection, rigorous validation, and a focus on financial impact rather than pure classification metrics.
"""

"""
Usage Note:

To predict the expected loss for a new borrower:

1. Prepare a dictionary containing the following features:

- 'credit_lines_outstanding'
- 'loan_amt_outstanding'
- 'total_debt_outstanding'
- 'income'
- 'years_employed'
- 'fico_score'

example_borrower = { ... }

2. Call the function:

loss = expected_loss(example_borrower)

3. The function will return the predicted Expected Loss (in EUR) for that borrower.
"""


import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('data/Task 3 and 4_Loan_Data.csv')
df = df.drop(columns=['customer_id'])
X = df.drop(columns=['default'])
y_true = df['default']
loan_amounts = X['loan_amt_outstanding']

# === MODEL PARAMETERS ===

intercept = -162.25625878207697

coefficients = {
    'credit_lines_outstanding': 109.86621197954673,
    'loan_amt_outstanding': 4.251256854802336,
    'total_debt_outstanding': 46.93871875472031,
    'income': -32.47613302502469,
    'years_employed': -37.81607883883549,
    'fico_score': -14.319640749360962
}

feature_means = {
    'credit_lines_outstanding': 1.4612,
    'loan_amt_outstanding': 4159.677034265904,
    'total_debt_outstanding': 8718.916796726344,
    'income': 70039.90140134419,
    'years_employed': 4.5528,
    'fico_score': 637.5577
}

feature_stds = {
    'credit_lines_outstanding': 1.7437587447809404,
    'loan_amt_outstanding': 1421.3280059071728,
    'total_debt_outstanding': 6626.833395074382,
    'income': 20071.210507366755,
    'years_employed': 1.566784018299906,
    'fico_score': 60.654873429181265
}

# === Final simple function for expected loss prediction ===

def expected_loss(borrower_features: dict) -> float:
    """
    Predict the expected loss (€) for a borrower based on input features.
    
    Input:
        borrower_features (dict): 
            A dictionary with keys:
                - 'credit_lines_outstanding'
                - 'loan_amt_outstanding'
                - 'total_debt_outstanding'
                - 'income'
                - 'years_employed'
                - 'fico_score'
    
    Output:
        expected_loss (float): Expected loss amount in EUR.
    """

    # Standardize the features
    standardized_features = []
    for feature in coefficients.keys():
        standardized_value = (borrower_features[feature] - feature_means[feature]) / feature_stds[feature]
        standardized_features.append(standardized_value)
    
    # Calculate the logit
    logit = intercept + np.dot(list(coefficients.values()), standardized_features)
    
    # Calculate probability of default
    pd_default = 1 / (1 + np.exp(-logit))
    
    # Calculate expected loss
    lgd = 0.9  # Loss Given Default
    loan_amount = borrower_features['loan_amt_outstanding']
    expected_loss_value = pd_default * lgd * loan_amount
    
    return expected_loss_value

# Example usage
borrower = {
    'credit_lines_outstanding': 2,
    'loan_amt_outstanding': 5000,
    'total_debt_outstanding': 12000,
    'income': 30000,
    'years_employed': 3,
    'fico_score': 620
}

expected_loss_value = expected_loss(borrower)
print(f"Expected Loss for the borrower: {expected_loss_value:.2f} EUR")


# === Test the function on the dataset ===

# Manual Logistic Regression prediction function
def manual_logistic_predict(X):
    logits = []
    for idx, row in X.iterrows():
        logit = intercept
        for feature, coef in coefficients.items():
            standardized = (row[feature] - feature_means[feature]) / feature_stds[feature]
            logit += coef * standardized
        logits.append(logit)
    logits = np.array(logits)
    pd_predictions = 1 / (1 + np.exp(-logits))
    return pd_predictions

# Predict PDs
pd_preds = manual_logistic_predict(X)

# Compute confusion matrix (using 0.5 threshold)
y_pred = (pd_preds >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)
cr = classification_report(y_true, y_pred)

print("\n✅ Confusion Matrix:")
print(cm)
print("\n✅ Classification Report:")
print(cr)

# Compute average absolute expected loss
lgd = 0.9  # Loss Given Default
expected_losses = pd_preds * lgd * loan_amounts
real_losses = y_true * lgd * loan_amounts
absolute_errors = np.abs(expected_losses - real_losses)
average_absolute_error = absolute_errors.mean()

print(f"\n✅ Average Absolute Expected Loss: {average_absolute_error:.2f} EUR")

# End of file.
