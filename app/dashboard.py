import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("https://github.com/im-kaami/ISSP25-Project/blob/main/data/salaries.csv") # make sure this path is correct

# Preprocessing
data.drop(columns=['salary', 'salary_currency'], inplace=True)
Q1 = data['salary_in_usd'].quantile(0.25)
Q3 = data['salary_in_usd'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['salary_in_usd'] >= Q1 - 1.5 * IQR) & (data['salary_in_usd'] <= Q3 + 1.5 * IQR)]

data_encoded = pd.get_dummies(data, columns=['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size'], drop_first=True)

# Model
X = data_encoded.drop(columns=['salary_in_usd'])
y = data_encoded['salary_in_usd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

from dash import Dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd

# Prepare job-level comparison DataFrame
comparison_df = pd.DataFrame({
    'Job Title': data.loc[X_test.index, 'job_title'].values,
    'Actual Salary': y_test.values,
    'Predicted Salary': y_pred
})

# Grouped averages for all jobs
avg_salary_comparison = comparison_df.groupby("Job Title").agg({
    "Actual Salary": "mean",
    "Predicted Salary": "mean"
}).reset_index()

# Remote work ratio data
remote_df = data[["job_title", "salary_in_usd", "remote_ratio"]].copy()

# Create the app
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Salary Model Dashboard", style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Model Evaluation', children=[
            dcc.Graph(
                figure=go.Figure([
                    go.Scatter(
                        x=y_test,
                        y=y_pred,
                        mode='markers',
                        marker=dict(size=6, opacity=0.6, color='blue'),
                        name='Predicted vs Actual'
                    ),
                    go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Perfect Prediction Line'
                    )
                ]).update_layout(
                    title='Model Evaluation: Actual vs Predicted Salary',
                    xaxis_title='Actual Salary (USD)',
                    yaxis_title='Predicted Salary (USD)',
                    legend=dict(x=0.75, y=0.1),
                    annotations=[dict(
                        xref='paper', yref='paper',
                        x=0.02, y=0.98,
                        text=f"R² = {r2:.4f}<br>RMSE = ${rmse:,.0f}<br>MAE = ${mae:,.0f}",
                        showarrow=False,
                        font=dict(size=12),
                        align='left',
                        bordercolor='black',
                        borderwidth=1,
                        borderpad=5,
                        bgcolor='white',
                    )]
                )
            )
        ]),
        dcc.Tab(label='Actual vs Predicted by Job Title', children=[
            html.H4("Grouped View (All Job Titles)"),
            dcc.Graph(
                figure=go.Figure([
                    go.Bar(x=avg_salary_comparison["Job Title"], y=avg_salary_comparison["Actual Salary"],
                           name="Actual", marker_color="steelblue"),
                    go.Bar(x=avg_salary_comparison["Job Title"], y=avg_salary_comparison["Predicted Salary"],
                           name="Predicted", marker_color="orange")
                ]).update_layout(
                    title="Average Actual vs Predicted Salary by Job Title",
                    barmode="group",
                    margin=dict(b=200),
                    height=600,
                    xaxis_title="Job Title",
                    yaxis_title="Salary (USD)"
                )
            ),
            html.H4("Select a Specific Job Title"),
            dcc.Dropdown(
                id='job-dropdown',
                options=[{'label': jt, 'value': jt} for jt in avg_salary_comparison["Job Title"]],
                value=avg_salary_comparison["Job Title"].iloc[0],
                style={'width': '50%'}
            ),
            dcc.Graph(id='single-job-graph')
        ]),
        dcc.Tab(label='Remote Ratio vs Salary', children=[
            html.H4("Select a Job Title"),
            dcc.Dropdown(
                id='remote-job-dropdown',
                options=[{'label': jt, 'value': jt} for jt in sorted(remote_df["job_title"].unique())],
                value=sorted(remote_df["job_title"].unique())[0],
                style={'width': '50%'}
            ),
            dcc.Graph(id='remote-salary-graph')
        ])
    ])
])

@app.callback(
    Output('single-job-graph', 'figure'),
    Input('job-dropdown', 'value')
)
def update_job_salary_graph(job):
    row = avg_salary_comparison[avg_salary_comparison["Job Title"] == job].iloc[0]
    return {
        'data': [
            go.Bar(x=["Actual Salary"], y=[row["Actual Salary"]], name='Actual', marker_color='steelblue'),
            go.Bar(x=["Predicted Salary"], y=[row["Predicted Salary"]], name='Predicted', marker_color='orange')
        ],
        'layout': go.Layout(
            title=f"Actual vs Predicted Salary for {job}",
            yaxis=dict(title='Salary (USD)'),
            barmode='group'
        )
    }

@app.callback(
    Output('remote-salary-graph', 'figure'),
    Input('remote-job-dropdown', 'value')
)
def update_remote_plot(job):
    subset = remote_df[remote_df["job_title"] == job]
    return {
        'data': [
            go.Scatter(
                x=subset["remote_ratio"],
                y=subset["salary_in_usd"],
                mode='markers',
                marker=dict(size=8, opacity=0.7, color='blue')
            )
        ],
        'layout': go.Layout(
            title=f"Remote Ratio vs Salary for {job}",
            xaxis=dict(title='Remote Work (%)'),
            yaxis=dict(title='Salary (USD)')
        )
    }

app.run(debug=True, port=8052)


