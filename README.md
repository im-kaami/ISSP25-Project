# ISSP25 AI Salary Prediction Project

## Overview
This project analyzes cybersecurity job salary data and builds a machine learning model to predict salaries. It also includes an interactive dashboard for exploring job trends, predictions, and remote work insights.

## Features
- Log-scaled salary distribution plots
- Experience level and remote work visualizations
- Top 10 job titles
- Salary distribution by company size
- Random Forest regression model to predict salaries
- SHAP explainability visualizations
- Interactive dashboard built using Dash and JupyterDash

## Folder Structure
```
ISSP25-Project/
│
├── data/                  # Dataset (if sharable)
├── notebooks/             # Jupyter Notebooks
├── app/                   # Dash app files
├── reports/               # Reports and presentations
├── images/                # Visualizations and plots
├── requirements.txt       # Python dependencies
├── README.md              # Project overview
└── .gitignore             # Ignored files
```

## How to Run the Dashboard
```bash
pip install -r requirements.txt
python app/dashboard.py
```

## Model Evaluation Metrics
- R² Score: 0.3224
- MAE: $36,122
- RMSE: $46,065

## Future Work
- Add skills, education level, and industry to improve model accuracy
- Try other ML models like XGBoost
- Deploy the dashboard as a web app
