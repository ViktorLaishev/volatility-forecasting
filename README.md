# Volatility Forecasting with CNN-LSTM

## Overview
This repository showcases a **CNN-LSTM** approach to forecasting daily realized volatility for the S&P 500 (or any chosen asset). It includes:

1. **Data Collection & Preprocessing** via Yahoo Finance, plus custom feature engineering (log returns, rolling volatility, EMAs, etc.).
2. A **Deep Learning Model** (CNN-LSTM) achieving near **R² ~0.80** on short-horizon daily volatility predictions.
3. **Real-Time Usage**:
   - **Auto-retraining** script that periodically pulls new data, retrains the model, and updates `latest_model.h5`.
   - A **Streamlit Web App** that provides an interactive interface to generate forecasts, VaR/ES risk metrics, and sample trading signals.

**Goal**: Combine advanced time-series modeling with easily accessible real-world data and a user-friendly pipeline to produce actionable short-term volatility forecasts.

---

---

## 1. Data & Preprocessing

1. **Data Source**  
   Uses [Yahoo Finance](https://finance.yahoo.com/) to download daily bars (`Open, High, Low, Close, Volume`).

2. **Feature Engineering**  
   - **Log Returns** = `log(Close_t / Close_(t-1))`
   - **Rolling Volatility** = std of log returns over a 10-day window
   - **High-Low Range** = `(High - Low) / Close`
   - **EMA_10, EMA_20** = Exponential moving averages
   - **EMA_Diff** = difference between EMA_10 and EMA_20

3. **Scaling**  
   - Input features → `StandardScaler()`
   - Target volatility → `MinMaxScaler(feature_range=(-1,1))`
   - Saved as `X_scaler.pkl` and `y_scaler.pkl` for consistent inference.

---

## 2. CNN-LSTM Model

1. **Architecture**  
   - **Conv1D** layers to extract local patterns from last 30 days of features.  
   - **LSTM** layers to handle temporal dependencies, finishing with a Dense(1).  
   - **Huber loss** (delta=0.01) for stable learning with small-range data.

2. **Performance**  
   - Achieved **R² ~0.80** for daily realized volatility forecasting on historical data.
   - MSE/MAE remain low, showing strong short-term predictive accuracy.

3. **Saving**  
   - `latest_model.h5` stores the final trained model.
   - If new data arrives and the model is retrained, we overwrite or version this file.

---

## 3. Automated Retraining (`auto_retrain.py`)

1. **Weekly Scheduling**  
   - Uses [schedule](https://github.com/dbader/schedule) or cron to run `retrain_model()` (e.g., every Monday at 00:00).
   - Pulls last 5 years of data, re-trains CNN-LSTM, saves updated model + scalers.

2. **Versioning**  
   - If you wish, store each newly trained model with a timestamp (e.g. `model_YYYYMMDD.h5`) to track performance over time.

---

## 4. Interactive Web App (`app.py`)

1. **Streamlit**  
   - Launch with `streamlit run app.py`.
   - Provides a UI: you pick the ticker, forecast horizon, etc.

2. **Real-Time Forecast**  
   - App loads `latest_model.h5` plus `X_scaler.pkl` and `y_scaler.pkl`.  
   - Fetches new data from Yahoo, applies the same feature engineering, scales it, runs a multi-step naive forecast for next N days.

3. **Outputs**  
   - Historical chart: predicted vs. actual daily volatility.  
   - Quick VaR/ES example.  
   - Simple trading signal suggestion if predicted vol is above/below average.

---

## 5. Installation & Usage

```bash
# 1) Clone the repo
git clone https://github.com/YourUsername/volatility-forecasting.git
cd volatility-forecasting

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Retrain the model
python auto_retrain.py

# 4) Run the Web App
streamlit run app.py



6. Future Enhancements

Ensemble Models
Combine CNN-LSTM with TCN or a Transformer, then average predictions for potentially better coverage of various market regimes.
Extended Data
Factor-based data (Fama-French factors), macro indicators, yield curves to expand the model’s knowledge.
Option Pricing
Compare predicted realized vol with implied vol (e.g., VIX) to see potential under/overpricing in the options market.
Deployment
Containerize (Docker) + a small CI/CD pipeline to keep the entire pipeline in production with daily refreshes.



License: MIT 

Maintainer: Viktor Laishev
