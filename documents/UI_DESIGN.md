# User Interface Design Guide

## Overview

The Streamlit application will provide an intuitive interface for credit risk assessment with the following main sections:

## Page Structure

### 1. Home / Dashboard

The landing page with key metrics and navigation

```
Navigation: [Dashboard] [Predict] [Batch] [Analytics] [Help]

Quick Stats:
- Total Predictions: 1,234
- Model Accuracy: 94.2%
- High-Risk Cases: 23%
- Latest Prediction: Just Now

Key Metrics:
- Precision: 0.92
- Recall: 0.88
- F1-Score: 0.90
- ROC-AUC: 0.96

Recent Predictions:
- John Doe - HIGH RISK (85%)
- Jane Smith - LOW RISK (15%)
- Bob Johnson - MEDIUM RISK (52%)
- View More...
```

### 2. Single Prediction Interface

For assessing individual loan applicants

```
SINGLE APPLICANT ASSESSMENT

Applicant Information:
- Full Name: [input field]
- Age: [input field]
- Income: [input field]
- Loan Amount: [input field]
- Employment Status: [dropdown]
- Credit History: [dropdown]
- Debt-to-Income Ratio: [input field]

[ASSESS RISK button]

PREDICTION RESULT:
- Risk Level: HIGH RISK (or MEDIUM/LOW)
- Probability: 78%
- Confidence: 92%
- Recommendation: REVIEW

FACTORS INFLUENCING PREDICTION:
- Feature Importance Chart
- Debt-to-Income: ████████████ 42%
- Credit Score: ██████████ 35%
- Income Stability: ███ 18%
- Previous Defaults: █ 5%

[View SHAP Explanation button]
```

### 3. Batch Prediction Interface

For processing multiple applicants

```
BATCH PREDICTION SYSTEM

Upload File:
- Drop CSV file here or click
- Expected columns: age, income, loan_amount, 
  employment_status, credit_history, debt_to_income

File Preview (if uploaded):
- Rows: 150 | Columns: 6
- Sample Data Table:
  | Name | Age | Income  | Status   |
  | John | 35  | $50,000 | Employed |
  | Jane | 28  | $65,000 | Employed |
  | Bob  | 45  | $80,000 | Employed |

[PROCESS BATCH] [CLEAR]

Processing Results:
- Status: Completed
- Results Summary:
  - Total Records: 150
  - High Risk: 34 (23%)
  - Medium Risk: 56 (37%)
  - Low Risk: 60 (40%)

Results Table:
| Name | Risk Level | Probability | Status |
| John | HIGH       | 78%         | OK     |
| Jane | LOW        | 12%         | OK     |
| Bob  | MEDIUM     | 52%         | OK     |

[DOWNLOAD RESULTS] [VIEW CHART]
```

### 4. Analytics & Dashboard

Model performance and trend analysis

```
PERFORMANCE DASHBOARD

Model Performance Metrics:
- Accuracy: 94.2% | Precision: 92%
- Recall: 88% | F1-Score: 90%
- ROC-AUC: 0.96 | Specificity: 85%

Confusion Matrix:              ROC Curve:
- True Negatives: 950         - TPR vs FPR plot
- False Positives: 50
- False Negatives: 80
- True Positives: 920

Feature Importance:
- Debt-to-Income Ratio: ████████████ 42%
- Credit Score: ██████████ 35%
- Annual Income: ███████ 22%
- Employment Status: ████ 12%
- Loan Amount: ██ 8%
- Previous Defaults: █ 3%

Prediction Trends (Last 30 Days):
- Line chart showing prediction counts over time
- Categories: High, Low, Medium

Risk Distribution:
- Low Risk: 40% [████░░░░]
- Medium Risk: 37% [███░░░░░░]
- High Risk: 23% [██░░░░░░░░░]
```

### 5. Model Explainability

SHAP value explanations for individual predictions

```
PREDICTION EXPLANATION (SHAP)

Applicant: John Doe
Risk Level: HIGH (78%)

Feature Contributions (SHAP Values):

Pushing toward LOW RISK:
- Debt-to-Income (45%): ████ +0.35
- Credit Score (580): ████ +0.32
- Age (28): ██ +0.15
- Income ($40k): █ +0.08
- Employment (Freelance): █ +0.06

Pushing toward HIGH RISK:
- Previous Default (Yes): ████ -0.20
- Loan-to-Value (85%): ███ -0.18

Base Value (Average Risk): 50%
Final Prediction: 78% (increase)

Key Insights:
- High debt-to-income ratio is the primary factor
  pushing this applicant to HIGH RISK category
- Low credit score also contributes significantly
- Previous default history is concerning

Recommendation:
- REJECT or REVIEW MANUALLY
```

## UI Features Summary

| Feature | Purpose | Location |
|---------|---------|----------|
| Input Validation | Catch errors early | All forms |
| Real-time Feedback | User guidance | Forms & buttons |
| Visual Indicators | Risk levels at a glance | Badges, colors |
| Charts & Graphs | Data visualization | Analytics pages |
| Download Results | Data export | Batch prediction |
| Help & Documentation | User guidance | Sidebar & tooltips |
| Performance Metrics | Model transparency | Dashboard |
| SHAP Explanations | Model interpretability | Explainability page |

## Color Scheme

```
Low Risk:    #28A745 (Green)
Medium Risk: #FFC107 (Amber)
High Risk:   #DC3545 (Red)
Info:        #0D6EFD (Blue)
Warning:     #FF6B6B (Light Red)
Success:     #10B981 (Teal)
```

## Responsive Design

- Desktop: Full layout with all features
- Tablet: Adjusted spacing and single column for forms
- Mobile: Simplified layout, stacked components

## Accessibility

- Clear button labels
- Color-blind friendly palette
- Keyboard navigation support
- Alt text for all visualizations
- Readable font sizes (minimum 14px)
- High contrast ratios

## Performance Considerations

- Cache model and predictions
- Lazy load visualizations
- Optimize large file uploads
- Progress indicators for long operations
- Pagination for large result tables
