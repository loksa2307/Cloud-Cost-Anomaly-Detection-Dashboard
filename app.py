# anomaly_insights.py
import logging
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Silence noisy libs
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Optional imports
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

try:
    import plotly
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ---------- Utility functions ----------
def load_data(source):
    """
    Load CSV from a file path or uploaded file-like object.
    Expect columns: Date, Cost
    """
    if hasattr(source, "read"):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)
    if 'Date' not in df.columns or 'Cost' not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Cost' columns.")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def aggregate_daily(df):
    return df.groupby('Date', as_index=False)['Cost'].sum()

def detect_anomalies(daily_df, contamination=0.05, random_state=42):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    daily = daily_df.copy()
    daily['anomaly_label'] = model.fit_predict(daily[['Cost']])
    daily['is_anomaly'] = daily['anomaly_label'] == -1
    return daily

def forecast_prophet(daily_df, periods=30):
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet not installed. pip install prophet")
    dfp = daily_df.rename(columns={'Date': 'ds', 'Cost': 'y'})
    m = Prophet()
    m.fit(dfp)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return m, forecast

def plot_anomalies(daily_df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_df['Date'], daily_df['Cost'], label='Daily Cost', linewidth=1)
    anomalies = daily_df[daily_df['is_anomaly']]
    if not anomalies.empty:
        ax.scatter(anomalies['Date'], anomalies['Cost'], color='red', label='Anomaly', s=40, zorder=5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cost')
    ax.set_title('Daily Cost with Detected Anomalies')
    ax.legend()
    plt.tight_layout()
    return fig

def simple_recommendation(cost):
    if cost > 500:
        return "High spend — consider reserved instances or rightsizing."
    if cost > 200:
        return "Moderate spend — review instance types and usage."
    return "Normal — monitor."

# ---------- Streamlit app ----------
def main():
    st.set_page_config(page_title="AI Cloud Cost Optimizer", layout="wide")
    st.title("☁️ AI Cloud Cost Optimizer — Anomaly & Forecast Insights")

    st.markdown("Upload a CSV with columns `Date` and `Cost` or use local `cloud_cost_data.csv`.")

    uploaded = st.file_uploader("CSV file", type=['csv'])
    use_local = st.checkbox("Use local 'cloud_cost_data.csv' if present", value=True)

    df = None
    if uploaded is not None:
        try:
            df = load_data(uploaded)
        except Exception as e:
            st.error(f"Failed to load uploaded CSV: {e}")
            st.stop()
    else:
        if use_local:
            try:
                df = load_data("cloud_cost_data.csv")
            except FileNotFoundError:
                st.info("No uploaded file and local 'cloud_cost_data.csv' not found. Upload a CSV to continue.")
                st.stop()
            except Exception as e:
                st.error(f"Failed to load local CSV: {e}")
                st.stop()
        else:
            st.info("Please upload a CSV to proceed.")
            st.stop()

    st.subheader("Raw data (first 5 rows)")
    st.dataframe(df.head())

    daily = aggregate_daily(df)
    st.subheader("Daily aggregated cost (summary)")
    st.write(daily.describe())

    contamination = st.sidebar.slider("Anomaly fraction", 0.01, 0.2, 0.05, 0.01)
    daily_with_anom = detect_anomalies(daily, contamination=contamination)

    st.subheader("Detected anomalies")
    st.write(f"Number of anomalies: {daily_with_anom['is_anomaly'].sum()}")
    st.dataframe(daily_with_anom[daily_with_anom['is_anomaly']].sort_values('Date').reset_index(drop=True).tail(20))

    st.subheader("Anomaly visualization")
    fig_anom = plot_anomalies(daily_with_anom)
    st.pyplot(fig_anom)

    st.subheader("Forecast (Prophet)")
    if not PROPHET_AVAILABLE:
        st.warning("Prophet not installed. Forecasting disabled. Install with `pip install prophet`.")
    else:
        periods = st.sidebar.number_input("Forecast days", min_value=7, max_value=365, value=30)
        with st.spinner("Training model..."):
            try:
                model, forecast = forecast_prophet(daily_with_anom, periods=periods)
                if PLOTLY_AVAILABLE:
                    try:
                        fig_plotly = model.plot_plotly(forecast)
                        st.plotly_chart(fig_plotly, use_container_width=True)
                    except Exception:
                        fig_forecast = model.plot(forecast)
                        st.pyplot(fig_forecast)
                else:
                    fig_forecast = model.plot(forecast)
                    st.pyplot(fig_forecast)
                st.write(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(10).rename(columns={'ds':'Date'}))
            except Exception as e:
                st.error(f"Forecasting failed: {e}")

    st.subheader("Simple recommendations")
    last_n = st.sidebar.number_input("Show last N days", min_value=5, max_value=90, value=14)
    rec_df = daily_with_anom.copy().sort_values('Date').tail(last_n)
    rec_df['Recommendation'] = rec_df['Cost'].apply(simple_recommendation)
    st.dataframe(rec_df[['Date','Cost','is_anomaly','Recommendation']].reset_index(drop=True))

    csv_bytes = rec_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download recommendations CSV", data=csv_bytes, file_name="recommendations.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("Run with: `streamlit run anomaly_insights.py`. If Prophet logs appear, they are harmless but can be silenced.")

if __name__ == "__main__":
    main()
