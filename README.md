# 💼 ISSP25 AI Salary Prediction Dashboard

## 📊 Overview
This project explores and predicts cybersecurity job salaries using a machine learning model. It includes:
- Exploratory data analysis (EDA)
- Salary prediction using Random Forest
- Interactive dashboard built with **Dash**

---

## ✨ Key Features

- 📈 **Log-scaled Salary Distribution**
- 🧠 **Experience Level & Remote Work Insights**
- 🧑‍💼 **Top 10 Common Job Titles**
- 🏢 **Salary Trends by Company Size**
- 🤖 **ML Model: Random Forest Regressor**
- 🔍 **Model Evaluation (R², MAE, RMSE)**
- 🧬 **SHAP Explainability**
- 🌐 **Interactive Dashboard (Dash)**

---

## 📁 Project Structure

```
ISSP25-Project/
├── data/                  # Dataset (CSV file)
├── notebooks/             # Exploratory and modeling notebooks
├── app/                   # Dash dashboard app
├── reports/               # Reports and presentations
├── images/                # Saved figures and visuals
├── requirements.txt       # Project dependencies
├── README.md              # You're reading this!
└── .gitignore             # Files ignored by Git
```

---

## 🚀 How to Run the Dashboard

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. ✅ IMPORTANT: Update Dataset Path
Before running the app, update the CSV file path in `app/dashboard.py`:
```python
data = pd.read_csv("C:/Users/your_username/path/to/salaries.csv")
```
Make sure the path points to your actual `salaries.csv`.

### 3. Run the dashboard
```bash
python app/dashboard.py
```

Then open your browser and visit:
```
http://127.0.0.1:8052/
```

---

## 📉 Model Evaluation Metrics

| Metric | Value |
|--------|-------|
| **R² Score** | 0.3224 |
| **MAE** | $36,122 |
| **RMSE** | $46,065 |

---

## 🔮 Future Improvements

- ➕ Add features: skills, education level, industry
- 🔁 Test alternative models (e.g., XGBoost, LightGBM)
- ☁️ Deploy dashboard to a cloud platform (e.g., Heroku, Render)

---