import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Predictive Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To",
    [
        "🏠 Home",
        "📂 Dataset",
        "📊 EDA",
        "⚙ Feature Engineering",
        "🤖 Machine Learning",
        "📈 Forecast",
        "ℹ About"
    ]
)

@st.cache_data
def load_data():
    feature_df = pd.read_csv("feature_engineered_dataset.csv",parse_dates=["Date"])
    model_df = pd.read_csv("Final_Model_Comparison.csv")
    forecast_df = pd.read_csv("future_30_day_forecast.csv")
    return feature_df, model_df, forecast_df

feature_df, model_df, forecast_df = load_data()
if page == "🏠 Home":

    st.title("📊 Predictive Forecasting of Care Load & Placement Demand")
    st.markdown("""This project predicts the future number of children in HHS Care using Machine Learning and Time Series Forecasting models.""")
    st.divider()
    best = model_df.loc[model_df["Accuracy"].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Records",len(feature_df))

    c2.metric(
        "Features",
        len(feature_df.columns)
    )

    c3.metric(
        "Best Model",
        best["Model"]
    )

    c4.metric(
        "Accuracy",
        f"{best['Accuracy']:.2f}%"
    )

    st.divider()

    st.subheader("Project Overview")

    st.write("""
This dashboard analyzes the HHS Unaccompanied Children dataset and predicts
future care load using Random Forest, Gradient Boosting and ARIMA models.

The project includes:

- Data Cleaning
- Feature Engineering
- Model Training
- Model Evaluation
- 30-Day Forecast
- Interactive Dashboard
""")

    st.subheader("Model Performance")

    fig = px.bar(
        model_df,
        x="Model",
        y="Accuracy",
        color="Model",
        text="Accuracy",
        title="Model Accuracy Comparison"
    )

    fig.update_traces(texttemplate="%{text:.2f}")

    st.plotly_chart(
        fig,
        use_container_width=True
    )

elif page == "📂 Dataset":

    st.title("📂 Dataset Explorer")

    st.write("Explore the cleaned and feature engineered dataset.")

    st.subheader("Dataset Preview")

    st.dataframe(feature_df, use_container_width=True)

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", feature_df.shape[0])
        st.metric("Columns", feature_df.shape[1])

    with col2:
        st.metric("Missing Values", int(feature_df.isnull().sum().sum()))
        st.metric("Duplicate Rows", int(feature_df.duplicated().sum()))

    st.subheader("Column Data Types")

    dtype_df = pd.DataFrame({
        "Column": feature_df.columns,
        "Data Type": feature_df.dtypes.astype(str)
    })

    st.dataframe(dtype_df, use_container_width=True)

    st.subheader("Statistical Summary")

    st.dataframe(feature_df.describe(), use_container_width=True)

    st.subheader("Select Column")

    selected_column = st.selectbox(
        "Choose a column",
        feature_df.select_dtypes(include="number").columns
    )

    fig = px.histogram(
        feature_df,
        x=selected_column,
        nbins=30,
        title=f"Distribution of {selected_column}"
    )

    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.box(
        feature_df,
        y=selected_column,
        title=f"Box Plot of {selected_column}"
    )

    st.plotly_chart(fig2, use_container_width=True)

elif page == "📊 EDA":

    st.title("📊 Exploratory Data Analysis")

    st.subheader("Children in HHS Care Over Time")

    fig = px.line(
        feature_df,
        x="Date",
        y="Children in HHS Care",
        title="Children in HHS Care Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("CBP Custody Over Time")

    fig = px.line(
        feature_df,
        x="Date",
        y="Children in CBP custody",
        title="Children in CBP Custody"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Children Discharged from HHS Care")

    fig = px.line(
        feature_df,
        x="Date",
        y="Children discharged from HHS Care",
        title="Children Discharged"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Average Children in HHS Care")

    monthly = feature_df.copy()
    monthly["Month_Name"] = monthly["Date"].dt.strftime("%B")

    monthly_avg = (
        monthly.groupby("Month_Name")["Children in HHS Care"]
        .mean()
        .reset_index()
    )

    month_order = [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ]

    monthly_avg["Month_Name"] = pd.Categorical(
        monthly_avg["Month_Name"],
        categories=month_order,
        ordered=True
    )

    monthly_avg = monthly_avg.sort_values("Month_Name")

    fig = px.bar(
        monthly_avg,
        x="Month_Name",
        y="Children in HHS Care",
        title="Monthly Average"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")

    corr = feature_df.select_dtypes(include="number").corr()

    fig = px.imshow(
     corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Feature Correlation Heatmap",
    aspect="auto"
    )   

    fig.update_layout(
    height=900,
    width=1200,
    font=dict(size=14),
    title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

elif page == "⚙ Feature Engineering":

    st.title("⚙ Feature Engineering")

    st.write(
        "The following features were created to improve forecasting performance."
    )

    explanation = pd.DataFrame({

        "Feature":[
            "Lag_1",
            "Lag_7",
            "Lag_14",
            "Rolling_Mean_7",
            "Rolling_STD_7",
            "Net_Pressure",
            "Day_of_Week",
            "Month",
            "Day"
        ],

        "Description":[
            "Previous day's HHS Care",
            "Previous week's value",
            "Two weeks previous value",
            "7-day Moving Average",
            "7-day Standard Deviation",
            "Transfers - Discharges",
            "Weekday Number",
            "Month Number",
            "Day Number"
        ]

    })

    st.dataframe(explanation, use_container_width=True)

    st.subheader("Feature Engineered Dataset")

    st.dataframe(feature_df, use_container_width=True)

    st.subheader("Feature Distribution")

    feature = st.selectbox(
        "Choose Feature",
        [
            "Lag_1",
            "Lag_7",
            "Lag_14",
            "Rolling_Mean_7",
            "Rolling_STD_7",
            "Net_Pressure"
        ]
    )

    fig = px.histogram(
        feature_df,
        x=feature,
        nbins=30,
        title=feature
    )

    st.plotly_chart(fig, use_container_width=True)

elif page == "🤖 Machine Learning":

    st.title("🤖 Machine Learning Results")

    st.subheader("Model Performance")

    st.dataframe(model_df, use_container_width=True)

    st.subheader("Accuracy Comparison")

    fig = px.bar(
        model_df,
        x="Model",
        y="Accuracy",
        color="Model",
        text="Accuracy"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("MAE Comparison")

    fig = px.bar(
        model_df,
        x="Model",
        y="MAE",
        color="Model",
        text="MAE"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("RMSE Comparison")

    fig = px.bar(
        model_df,
        x="Model",
        y="RMSE",
        color="Model",
        text="RMSE"
    )

    st.plotly_chart(fig, use_container_width=True)

    best = model_df.loc[
        model_df["Accuracy"].idxmax()
    ]

    st.success(
        f"""
Best Model : {best['Model']}

Accuracy : {best['Accuracy']:.2f}%

MAE : {best['MAE']:.2f}

RMSE : {best['RMSE']:.2f}
"""
    )
elif page == "📈 Forecast":

    st.title("📈 30-Day Future Forecast")

    forecast_df = pd.read_csv("future_30_day_forecast.csv")

    if forecast_df.shape[1] == 2:
        forecast_df.columns = ["Day", "Forecast"]
    else:
        forecast_df.columns = ["Forecast"]
        forecast_df["Day"] = range(1, len(forecast_df) + 1)

    st.subheader("Forecast Table")

    st.dataframe(forecast_df, use_container_width=True)

    st.subheader("Forecast Graph")

    fig = px.line(
        forecast_df,
        x="Day",
        y="Forecast",
        markers=True,
        title="Next 30 Days Forecast"
    )

    fig.update_layout(
        xaxis_title="Future Day",
        yaxis_title="Predicted Children in HHS Care"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Minimum Forecast",
        f"{forecast_df['Forecast'].min():.0f}"
    )

    col2.metric(
        "Average Forecast",
        f"{forecast_df['Forecast'].mean():.0f}"
    )

    col3.metric(
        "Maximum Forecast",
        f"{forecast_df['Forecast'].max():.0f}"
    )

    st.subheader("Forecast Trend")

    fig2 = px.area(
        forecast_df,
        x="Day",
        y="Forecast",
        title="30-Day Forecast Trend"
    )

    st.plotly_chart(fig2, use_container_width=True)

    csv = forecast_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Forecast CSV",
        data=csv,
        file_name="future_30_day_forecast.csv",
        mime="text/csv"
    )

elif page == "ℹ About":

    st.title("ℹ Project Information")

    st.header("Project Title")

    st.write(
        "Predictive Forecasting of Care Load & Placement Demand"
    )

    st.header("Problem Statement")

    st.write("""
The project predicts the future number of children in HHS Care
using Machine Learning and Time Series Forecasting techniques.

The objective is to help estimate future care demand,
allowing better resource planning.
""")

    st.header("Dataset")

    st.write("""
Source:
U.S. Department of Health and Human Services (HHS)

Records:
706

Features:
13
""")

    st.header("Algorithms Used")

    algo = pd.DataFrame({

        "Algorithm":[
            "Random Forest",
            "Gradient Boosting",
            "ARIMA"
        ],

        "Purpose":[
            "Machine Learning Regression",
            "Boosting Regression",
            "Time Series Forecasting"
        ]

    })

    st.table(algo)

    st.header("Best Model")

    best = model_df.loc[
        model_df["Accuracy"].idxmax()
    ]

    st.success(
        f"""
Best Model : {best['Model']}

Accuracy : {best['Accuracy']:.2f}%
"""
    )

    st.header("Workflow")

    st.markdown("""
1. Data Collection

2. Data Cleaning

3. Feature Engineering

4. Model Training

5. Model Evaluation

6. Future Forecasting

7. Dashboard Development
""")

    st.header("Developer")

    st.write("""
Developed by:

Yash K. Patil

B.Tech in Robotics & AI
""")

