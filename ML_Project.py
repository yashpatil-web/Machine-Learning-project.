import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA

file_path = r"C:\Users\admin\Desktop\YASH COLLEGE\CODING\machine learning\dataset\HHS_Unaccompanied_Alien_Children_Program.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
df = df.sort_values("Date")
df = df.drop_duplicates(subset="Date")
df.set_index("Date", inplace=True)

for col in df.columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv("cleaned_uac_dataset.csv")

# plt.style.use("ggplot")
# df = pd.read_csv("cleaned_uac_dataset.csv",parse_dates=["Date"],index_col="Date")
# # print(df.head())
# # print(df.info())
# print(df.describe())

# for col in df.columns:
#     plt.figure(figsize=(12,5))
#     plt.plot(df.index, df[col], linewidth=2)
#     plt.title(col, fontsize=14)
#     plt.xlabel("Date")
#     plt.ylabel(col)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# plt.figure(figsize=(10,8))
# sns.heatmap(df.corr(),annot=True,cmap="coolwarm",linewidths=0.5)
# plt.title("Correlation Heatmap")
# plt.tight_layout()
# plt.show()
# for col in df.columns:
#     plt.figure(figsize=(5,6))
#     sns.boxplot(y=df[col])
#     plt.title(f"Boxplot of {col}")
#     plt.tight_layout()
#     plt.show()

# monthly = df.resample("ME").mean()
# plt.figure(figsize=(14,6))
# plt.plot(monthly.index,monthly["Children in HHS Care"],linewidth=3)
# plt.title("Monthly Average Children in HHS Care")
# plt.xlabel("Month")
# plt.ylabel("Children")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# weekly = df.resample("W").mean()
# plt.figure(figsize=(14,6))
# plt.plot(weekly.index,weekly["Children in HHS Care"],linewidth=2)
# plt.title("Weekly Average Children in HHS Care")
# plt.xlabel("Week")
# plt.ylabel("Children")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(14,6))
# plt.plot(df["Children in HHS Care"],label="Original")
# plt.plot(df["Children in HHS Care"].rolling(7).mean(),linewidth=3,label="7-Day Rolling Mean")
# plt.legend()
# plt.title("7-Day Rolling Mean")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# decompose = seasonal_decompose(df["Children in HHS Care"],model="additive",period=7)
# decompose.plot()
# plt.show()

# sns.pairplot(df)
# plt.show()

target = "Children in HHS Care"

df["Lag_1"] = df[target].shift(1)
df["Lag_7"] = df[target].shift(7)
df["Lag_14"] = df[target].shift(14)
df["Rolling_Mean_7"] = df[target].rolling(7).mean()
df["Rolling_STD_7"] = df[target].rolling(7).std()
df["Net_Pressure"] = (df["Children transferred out of CBP custody"] - df["Children discharged from HHS Care"])
df["Day_of_Week"] = df.index.dayofweek
df["Month"] = df.index.month
df["Day"] = df.index.day

df = df.dropna()

df.to_csv("feature_engineered_dataset.csv")

print(df.head())
print(df.shape)

X = df.drop(columns=[target])
y = df[target]

split = int(len(df) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

gb = GradientBoostingRegressor(
    random_state=42
)

rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
gb_pred = gb.predict(X_test)

def evaluate(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    accuracy = 100 - mape
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print(f"MAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")
    print(f"MAPE     : {mape:.2f}%")
    print(f"Accuracy : {accuracy:.2f}%")
    return [name, mae, rmse, mape, accuracy]

rf_result = evaluate("Random Forest", y_test, rf_pred)
gb_result = evaluate("Gradient Boosting", y_test, gb_pred)

model_results = pd.DataFrame([rf_result, gb_result],columns=["Model", "MAE", "RMSE", "MAPE", "Accuracy (%)"])

print(model_results)

model_results.to_csv("model_comparison.csv", index=False)
train = y_train
test = y_test

naive_pred = test.shift(1)
naive_pred.iloc[0] = train.iloc[-1]

moving_avg = train.tail(7).mean()
moving_pred = np.repeat(moving_avg, len(test))

arima = ARIMA(train, order=(5, 1, 0))
arima_fit = arima.fit()

arima_pred = arima_fit.forecast(steps=len(test))
future_forecast = arima_fit.forecast(steps=30)

naive_result = evaluate("Naive Forecast", test, naive_pred)
moving_result = evaluate("Moving Average", test, moving_pred)
arima_result = evaluate("ARIMA", test, arima_pred)

time_series_results = pd.DataFrame([naive_result, moving_result, arima_result],columns=["Model", "MAE", "RMSE", "MAPE", "Accuracy (%)"])
print(time_series_results)

time_series_results.to_csv("time_series_models.csv", index=False)

print("\nNext 30 Days Forecast")
print(future_forecast)

future_forecast.to_csv("future_30_day_forecast.csv")

final_results = pd.DataFrame({
    "Model": ["Random Forest", "Gradient Boosting", "ARIMA"],
    "MAE": [rf_result[1], gb_result[1], arima_result[1]],
    "RMSE": [rf_result[2], gb_result[2], arima_result[2]],
    "MAPE": [rf_result[3], gb_result[3], arima_result[3]],
    "Accuracy": [rf_result[4], gb_result[4], arima_result[4]]
})

print(final_results)

final_results.to_csv("Final_Model_Comparison.csv", index=False)
best = final_results.sort_values("MAE").iloc[0]

print(best)

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, rf_pred, label="Random Forest Prediction")
plt.title("Actual vs Random Forest Prediction")
plt.xlabel("Date")
plt.ylabel("Children in HHS Care")
plt.legend()
plt.grid(True)
plt.savefig("actual_vs_random_forest.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, gb_pred, label="Gradient Boosting Prediction")
plt.title("Actual vs Gradient Boosting")
plt.xlabel("Date")
plt.ylabel("Children in HHS Care")
plt.legend()
plt.grid(True)
plt.savefig("actual_vs_gradient_boosting.png")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, arima_pred, label="ARIMA Prediction")
plt.title("Actual vs ARIMA")
plt.xlabel("Date")
plt.ylabel("Children in HHS Care")
plt.legend()
plt.grid(True)
plt.savefig("actual_vs_arima.png")
plt.show()

future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),periods=30,freq="D")

plt.figure(figsize=(12, 6))
plt.plot(future_dates, future_forecast, marker="o")
plt.title("30-Day Future Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Children in HHS Care")
plt.grid(True)
plt.savefig("future_forecast_30_days.png")
plt.show()
