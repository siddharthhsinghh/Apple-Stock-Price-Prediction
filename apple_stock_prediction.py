import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TICKER = "AAPL"
START_DATE = "2010-01-01"
TRAIN_RATIO = 0.70  # tried 0.6 and 0.8, this split gave the most stable results
TEST_WINDOW = 60
MIN_TRAIN_SIZE = 500
RIDGE_ALPHA = 1.0
LASSO_ALPHA = 0.1  # lower alpha since lasso was zeroing out too many features at 1.0

def calculate_rsi(prices, period=14):
    # RSI: momentum oscillator, 0-100
    delta = prices.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast   = prices.ewm(span=fast, adjust=False).mean()
    ema_slow   = prices.ewm(span=slow, adjust=False).mean()
    macd       = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

def calculate_volatility_features(high, low, close, period=14, bb_period=20, std_dev=2):
    hl = high - low
    hc = np.abs(high - close.shift())
    lc = np.abs(low - close.shift())
    atr = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(window=period).mean()
    sma = close.rolling(window=bb_period).mean()
    std = close.rolling(window=bb_period).std()
    return atr, sma + std * std_dev, sma - std * std_dev

def engineer_features(df):
    data = df.copy()

    for lag in [1, 2, 3, 5, 10, 20, 60]:
        data[f"Close_lag{lag}"] = data["Close"].shift(lag)

    data["Open_lag1"] = data["Open"].shift(1)
    data["High_lag1"] = data["High"].shift(1)
    data["Low_lag1"] = data["Low"].shift(1)
    data["Volume_lag1"] = data["Volume"].shift(1)

    data["SMA_10"] = data["Close"].rolling(window=10).mean()
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()

    data["Price_to_SMA20"] = data["Close"] / data["SMA_20"]
    data["Price_to_SMA50"] = data["Close"] / data["SMA_50"]

    data["RSI"]  = calculate_rsi(data["Close"])
    data["MACD"], data["MACD_Signal"] = calculate_macd(data["Close"])
    data["MACD_Histogram"] = data["MACD"] - data["MACD_Signal"]
    data["ROC_5"] = data["Close"].pct_change(periods=5) * 100
    data["ROC_20"] = data["Close"].pct_change(periods=20) * 100

    data["ATR"], data["BB_Upper"], data["BB_Lower"] = calculate_volatility_features(data["High"], data["Low"], data["Close"])
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / data["SMA_20"]
    data["Volatility_20"] = data["Close"].rolling(window=20).std()

    data["Volume_SMA_20"] = data["Volume"].rolling(window=20).mean()
    data["Volume_Ratio"]  = data["Volume"] / data["Volume_SMA_20"]

    data["Daily_Return"] = data["Close"].pct_change()
    data["High_Low_Spread"]  = (data["High"] - data["Low"]) / data["Close"]
    data["Close_Open_Spread"] = (data["Close"] - data["Open"]) / data["Open"]

    np.random.seed(42)
    data["Random_Control_Feature"] = np.random.normal(loc=0.1, scale=0.5, size=len(data))
    data["Return_Momentum_5d"] = data["Daily_Return"].rolling(window=5).mean()

    return data

def evaluate_model(y_true, y_pred, model_name):
    """Returns a dict of MAE, RMSE, R², and MAPE for a given model."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "Model":    model_name,
        "MAE ($)":  round(mae, 4),
        "RMSE ($)": round(rmse, 4),
        "R² Score": round(r2, 4),
        "MAPE (%)": round(mape, 2),
    }
print(f"Apple stock price prediction — {TICKER}, {START_DATE} to present")

print(f"\nDownloading {TICKER} data from {START_DATE}...")
raw_data = yf.Ticker(TICKER).history(start=START_DATE)
print(f"{len(raw_data)} trading days downloaded")

print("\nEngineering features...")
data = engineer_features(raw_data)
data = data.dropna()
print(f"Features built — {len(data)} clean rows ready")

BASE_FEATURES = [
    "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag5",
    "Close_lag10", "Close_lag20", "Close_lag60",
    "Open_lag1", "High_lag1", "Low_lag1", "Volume_lag1",
    "SMA_10", "SMA_20", "SMA_50", "SMA_200",
    "EMA_12", "EMA_26",
    "Price_to_SMA20", "Price_to_SMA50",
    "RSI", "MACD", "MACD_Signal", "MACD_Histogram",
    "ROC_5", "ROC_20",
    "ATR", "BB_Width", "Volatility_20",
    "Volume_Ratio", "Daily_Return",
    "High_Low_Spread", "Close_Open_Spread",
]

CONTROL_FEATURES = ["Random_Control_Feature", "Return_Momentum_5d"]
ALL_FEATURES = BASE_FEATURES + CONTROL_FEATURES

print("\nRunning walk-forward validation")

initial_train_size = int(len(data) * TRAIN_RATIO)

print(f"Initial training size : {initial_train_size} samples")
print(f"Test window  : {TEST_WINDOW} days per window")
print(f"Total data points : {len(data)}")
print()

all_predictions = {
    "Naive":           [],
    "Ridge_no_control": [],
    "Ridge_with_control": [],
    "Lasso_no_control":     [],
    "ElasticNet":      [],
    "RandomForest":    [],
    "GradientBoosting":[],
}
all_actuals = []
all_dates   = []

train_end  = initial_train_size
window_num = 0

while train_end + TEST_WINDOW <= len(data):
    window_num += 1
    test_start = train_end
    test_end   = test_start + TEST_WINDOW

    print(f"Window {window_num:02d}: Train[0:{train_end}] → Test[{test_start}:{test_end}]")

    train_data = data.iloc[0:train_end]
    test_data  = data.iloc[test_start:test_end]

    X_train_base = train_data[BASE_FEATURES]
    X_train_all  = train_data[ALL_FEATURES]
    y_train      = train_data["Close"]

    X_test_base  = test_data[BASE_FEATURES]
    X_test_all   = test_data[ALL_FEATURES]
    y_test       = test_data["Close"]

    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    X_test_base_scaled  = scaler_base.transform(X_test_base)

    scaler_all = StandardScaler()
    X_train_all_scaled = scaler_all.fit_transform(X_train_all)
    X_test_all_scaled  = scaler_all.transform(X_test_all)

    naive_pred = X_test_base["Close_lag1"].values

    ridge_base = Ridge(alpha=RIDGE_ALPHA, random_state=42)
    ridge_base.fit(X_train_base_scaled, y_train)
    ridge_pred_base = ridge_base.predict(X_test_base_scaled)

    ridge_all = Ridge(alpha=RIDGE_ALPHA, random_state=42)
    ridge_all.fit(X_train_all_scaled, y_train)
    ridge_pred_all = ridge_all.predict(X_test_all_scaled)

    lasso = Lasso(alpha=LASSO_ALPHA, max_iter=10000, random_state=42)
    lasso.fit(X_train_base_scaled, y_train)
    lasso_pred = lasso.predict(X_test_base_scaled)

    enet = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000, random_state=42)
    enet.fit(X_train_all_scaled, y_train)
    enet_pred = enet.predict(X_test_all_scaled)

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=15,
        min_samples_split=10, min_samples_leaf=4,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train_all, y_train)
    rf_pred = rf.predict(X_test_all)

    gb = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.05,
        max_depth=5, min_samples_split=10,
        random_state=42
    )
    gb.fit(X_train_all, y_train)
    gb_pred = gb.predict(X_test_all)

    all_predictions["Naive"].extend(naive_pred)
    all_predictions["Ridge_no_control"].extend(ridge_pred_base)
    all_predictions["Ridge_with_control"].extend(ridge_pred_all)
    all_predictions["Lasso_no_control"].extend(lasso_pred)
    all_predictions["ElasticNet"].extend(enet_pred)
    all_predictions["RandomForest"].extend(rf_pred)
    all_predictions["GradientBoosting"].extend(gb_pred)

    all_actuals.extend(y_test.values)
    all_dates.extend(y_test.index)

    train_end = test_end

print(f"\nCompleted {window_num} walk-forward windows")


print("\nEvaluating models...")

all_actuals = np.array(all_actuals)
all_dates   = pd.DatetimeIndex(all_dates)

results = []
for model_name, preds in all_predictions.items():
    results.append(evaluate_model(all_actuals, np.array(preds), model_name))

results_df = pd.DataFrame(results).sort_values("MAE ($)").reset_index(drop=True)

print("\nModel performance (aggregated across all windows):")
print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\n  Best Model : {best['Model']}")
print(f"  MAE        : ${best['MAE ($)']}")
print(f"  RMSE       : ${best['RMSE ($)']}")
print(f"  R² Score   : {best['R² Score']}")
print(f"  MAPE       : {best['MAPE (%)']}%")


fig, axes = plt.subplots(3, 1, figsize=(20, 16))
fig.suptitle("Apple Stock Price Prediction — Walk-Forward Validation", fontsize=18, fontweight="bold", y=0.98)

ax1 = axes[0]
ax1.plot(all_dates, all_actuals,                          label="Actual Price",      color="black",  linewidth=2.5, alpha=0.9)
ax1.plot(all_dates, all_predictions["Naive"],             label="Naive Baseline",    color="gray",   linestyle=":",  linewidth=1.5, alpha=0.7)
ax1.plot(all_dates, all_predictions["Ridge_no_control"], label="Ridge (no control feature)", color="blue", linestyle="-.", linewidth=1.5, alpha=0.7)
ax1.plot(all_dates, all_predictions["Ridge_with_control"], label="Ridge (with control feature)", color="purple", linestyle="--", linewidth=1.5, alpha=0.7)
ax1.plot(all_dates, all_predictions["GradientBoosting"],  label="Gradient Boosting", color="green",  linestyle="-",  linewidth=1.8, alpha=0.8)
ax1.set_title("Model Comparison: Actual vs Predicted Prices", fontsize=14, fontweight="bold")
ax1.set_ylabel("Price ($)", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
errors_ridge = all_actuals - np.array(all_predictions["Ridge_with_control"])
errors_gb    = all_actuals - np.array(all_predictions["GradientBoosting"])
ax2.plot(all_dates, errors_ridge, label="Ridge Error",color="purple", alpha=0.6, linewidth=1)
ax2.plot(all_dates, errors_gb,    label="Gradient Boosting Error", color="green",  alpha=0.6, linewidth=1)
ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
ax2.fill_between(all_dates, 0, errors_ridge, alpha=0.15, color="purple")
ax2.fill_between(all_dates, 0, errors_gb,    alpha=0.15, color="green")
ax2.set_title("Prediction Errors Over Time", fontsize=14, fontweight="bold")
ax2.set_ylabel("Error ($)", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(True, alpha=0.3)

ax3 = axes[2]
cum_mae_ridge, cum_mae_gb = [], []
actuals_arr   = all_actuals
ridge_arr     = np.array(all_predictions["Ridge_with_control"])
gb_arr        = np.array(all_predictions["GradientBoosting"])

for i in range(1, len(actuals_arr) + 1):
    cum_mae_ridge.append(mean_absolute_error(actuals_arr[:i], ridge_arr[:i]))
    cum_mae_gb.append(mean_absolute_error(actuals_arr[:i], gb_arr[:i]))

ax3.plot(all_dates, cum_mae_ridge, label="Ridge Cumulative MAE", color="purple", linewidth=2, alpha=0.8)
ax3.plot(all_dates, cum_mae_gb,    label="Gradient Boosting Cumulative MAE", color="green",  linewidth=2, alpha=0.8)
ax3.set_title("Cumulative MAE Over Time", fontsize=14, fontweight="bold")
ax3.set_xlabel("Date", fontsize=12)
ax3.set_ylabel("Cumulative MAE ($)", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("apple_stock_prediction_results.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nChart saved as 'apple_stock_prediction_results.png'")
print("\nWalk-Forward Analysis Complete!")
