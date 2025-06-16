import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from joblib import load

from notebooks.src.config import DADOS_TRATADOS, MODELO_FINAL


@st.cache_data
def load_data():
    return pd.read_parquet('dados/employee_attrition.parquet')


@st.cache_resource
def load_model():
    return load('modelos/logistic_regression.joblib')


df = load_data()
model = load_model()

education_levels_text = {
    1: "Below College",
    2: "College",
    3: "Bachelor",
    4: "Master",
    5: "PhD",
}

satisfaction_levels_text = {
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High",
}

work_life_balance_text = {
    1: "Bad",
    2: "Good",
    3: "Better",
    4: "Best",
}

genders = sorted(df["Gender"].unique())
education_levels = sorted(df["Education"].unique())
education_fields = sorted(df["EducationField"].unique())
departments = sorted(df["Department"].unique())
business_travel = sorted(df["BusinessTravel"].unique())
overtime_options = sorted(df["OverTime"].unique())
job_satisfaction = sorted(df["JobSatisfaction"].unique())
relationship_satisfaction = sorted(df["RelationshipSatisfaction"].unique())
environment_satisfaction = sorted(df["EnvironmentSatisfaction"].unique())
work_life_balance = sorted(df["WorkLifeBalance"].unique())
stock_options = sorted(df["StockOptionLevel"].unique())
job_involvement = sorted(df["JobInvolvement"].unique())

slider_columns = [
    "DistanceFromHome",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

slider_min_max = {
    col: {"min_value": df[col].min(), "max_value": df[col].max()}
    for col in slider_columns
}

ignored_columns = (
    "Age",
    "DailyRate",
    "JobLevel",
    "HourlyRate",
    "MonthlyRate",
    "PerformanceRating",
)

ignored_columns_medians = {
    col: df[col].median() for col in ignored_columns
}

st.title("Employee Attrition Prediction")

with st.container(border=True):
    st.write("### Personal Information")

    gender_widget = st.radio("Gender", genders)

    education_level_widget = st.selectbox(
        "Education Level",
        education_levels,
        format_func=lambda number: education_levels_text[number]
    )

    education_field_widget = st.selectbox("Education Field", education_fields)

    distance_home_widget = st.slider(
        "Distance from Home", **slider_min_max["DistanceFromHome"]
    )

with st.container(border=True):
    st.write("### Work Routine")

    left_col, right_col = st.columns(2)

    with left_col:
        department_widget = st.selectbox("Department", departments)
        business_travel_widget = st.selectbox("Business Travel", business_travel)

    with right_col:
        job_role_widget = st.selectbox(
            "Job Role",
            sorted(df[df["Department"] == department_widget]["JobRole"].unique())
        )

        overtime_widget = st.radio("Overtime", overtime_options)

    monthly_income_widget = st.slider(
        "Monthly Income", **slider_min_max["MonthlyIncome"]
    )

with st.container(border=True):
    st.write("### Professional Experience")

    left_col, right_col = st.columns(2)

    with left_col:
        num_companies_widget = st.slider(
            "Number of Companies Worked", **slider_min_max["NumCompaniesWorked"]
        )
        total_years_widget = st.slider(
            "Total Working Years", **slider_min_max["TotalWorkingYears"]
        )
        years_company_widget = st.slider(
            "Years at Company", **slider_min_max["YearsAtCompany"]
        )

    with right_col:
        years_current_role_widget = st.slider(
            "Years in Current Role", **slider_min_max["YearsInCurrentRole"]
        )
        years_manager_widget = st.slider(
            "Years with Current Manager", **slider_min_max["YearsWithCurrManager"]
        )
        years_last_promotion_widget = st.slider(
            "Years Since Last Promotion", **slider_min_max["YearsSinceLastPromotion"]
        )

with st.container(border=True):
    st.write("### Incentives and Metrics")

    left_col, right_col = st.columns(2)

    with left_col:
        job_satisfaction_widget = st.selectbox(
            "Job Satisfaction",
            job_satisfaction,
            format_func=lambda number: satisfaction_levels_text[number],
        )

        relationship_satisfaction_widget = st.selectbox(
            "Relationship Satisfaction",
            relationship_satisfaction,
            format_func=lambda number: satisfaction_levels_text[number],
        )

        job_involvement_widget = st.selectbox(
            "Job Involvement", job_involvement
        )

    with right_col:
        environment_satisfaction_widget = st.selectbox(
            "Environment Satisfaction",
            environment_satisfaction,
            format_func=lambda number: satisfaction_levels_text[number],
        )

        work_life_balance_widget = st.selectbox(
            "Work-Life Balance",
            work_life_balance,
            format_func=lambda number: work_life_balance_text[number],
        )

        stock_option_widget = st.radio("Stock Option", stock_options)

    salary_hike_widget = st.slider(
        "Salary Hike (%)", **slider_min_max["PercentSalaryHike"]
    )

    training_last_year_widget = st.slider(
        "Trainings Last Year", **slider_min_max["TrainingTimesLastYear"]
    )

model_input = {
    "Age": ignored_columns_medians["Age"],
    "BusinessTravel": business_travel_widget,
    "DailyRate": ignored_columns_medians["DailyRate"],
    "Department": department_widget,
    "DistanceFromHome": distance_home_widget,
    "Education": education_level_widget,
    "EducationField": education_field_widget,
    "EnvironmentSatisfaction": environment_satisfaction_widget,
    "Gender": gender_widget,
    "HourlyRate": ignored_columns_medians["HourlyRate"],
    "JobInvolvement": job_involvement_widget,
    "JobLevel": ignored_columns_medians["JobLevel"],
    "JobRole": job_role_widget,
    "JobSatisfaction": job_satisfaction_widget,
    "MaritalStatus": "Single",
    "MonthlyIncome": monthly_income_widget,
    "MonthlyRate": ignored_columns_medians["MonthlyRate"],
    "NumCompaniesWorked": num_companies_widget,
    "PerformanceRating": ignored_columns_medians["PerformanceRating"],
    "OverTime": overtime_widget,
    "PercentSalaryHike": salary_hike_widget,
    "RelationshipSatisfaction": relationship_satisfaction_widget,
    "StockOptionLevel": stock_option_widget,
    "TotalWorkingYears": total_years_widget,
    "TrainingTimesLastYear": training_last_year_widget,
    "WorkLifeBalance": work_life_balance_widget,
    "YearsAtCompany": years_company_widget,
    "YearsInCurrentRole": years_current_role_widget,
    "YearsSinceLastPromotion": years_last_promotion_widget,
    "YearsWithCurrManager": years_manager_widget,
}

df_model_input = pd.DataFrame([model_input])

predict_button = st.button("Predict Attrition")

if predict_button:
    prediction = model.predict(df_model_input)[0]
    attrition_probability = model.predict_proba(df_model_input)[0][1]

    color = ":red" if prediction == 1 else ":green"

    attrition_text = f"#### Attrition: {color}[{'Yes' if prediction == 1 else 'No'}]"
    probability_text = f"#### Attrition Probability: {color}[{attrition_probability:.1%}]"

    st.markdown(attrition_text)
    st.markdown(probability_text)
