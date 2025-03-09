import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_echarts import st_echarts
import time

# Streamlit Page Configuration
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and Styling
st.markdown("<h1 style='text-align: center; color: #FFA500;'>⚠️ Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #FFFFFF;'>🔍 Live 5G Threat Monitoring</h3>", unsafe_allow_html=True)

# Global Variables
log_df = pd.DataFrame(columns=["Packet ID", "Prediction"])

# ------------------- METRICS DISPLAY -------------------
col1, col2, col3 = st.columns(3)
with col1:
    total_anomalies = st.metric("⚠️ Anomalies Detected", 0)
with col2:
    normal_traffic = st.metric("✅ Normal Traffic", 0)
with col3:
    accuracy_display = st.metric("🎯 Model Accuracy", "N/A")

# ------------------- GAUGES -------------------
st.subheader("📊 Threat Level & Current Data Gauges")

col_g1, col_g2 = st.columns(2)

# Persistent Threat Level Gauge (black background, ticks, labels, and split lines)
threat_gauge_options = {
    "backgroundColor": "#0E1117",  # Force black background to avoid white
    "series": [
        {
            "type": "gauge",
            "startAngle": 180,
            "endAngle": 0,
            "min": 0,
            "max": 100,
            "splitNumber": 5,
            "axisLine": {
                "lineStyle": {
                    "width": 10,
                    "color": [[0.3, "#1a9850"], [0.7, "#fdae61"], [1, "#d73027"]],
                },
            },
            "pointer": {"itemStyle": {"color": "auto"}},
            "detail": {"formatter": "{value}%"},
            "data": [{"value": 0, "name": "Threat Level"}],
        }
    ]
}

# Current Data Gauge (black background, ticks, labels, and split lines restored)
current_data_gauge_options = {
    "backgroundColor": "#0E1117",  # Force black background to avoid white
    "series": [
        {
            "type": "gauge",
            "startAngle": 180,
            "endAngle": 0,
            "min": 0,
            "max": 100,
            "splitNumber": 5,
            "axisLine": {
                "lineStyle": {
                    "width": 10,
                    "color": [[0.8, "#1a9850"], [0.9, "#fdae61"], [1, "#d73027"]],
                },
            },
            "pointer": {"itemStyle": {"color": "auto"}},
            "detail": {"formatter": "{value}%"},
            "data": [{"value": 0, "name": "Current Data"}],
        }
    ]
}

with col_g1:
    threat_gauge = st_echarts(threat_gauge_options, height="300px")

with col_g2:
    current_data_gauge = st_echarts(current_data_gauge_options, height="300px")

# ------------------- CONFUSION MATRIX -------------------
st.subheader("🔍 Confusion Matrix")
conf_matrix_placeholder = st.empty()

# ------------------- PACKET CLASSIFICATION LOG -------------------
st.subheader("📜 Classification Log")
log_placeholder = st.empty()

# ------------------- REAL-TIME ANOMALY TRENDS -------------------
st.subheader("📈 Anomaly Trend Over Time")
trend_chart_placeholder = st.empty()
anomaly_trend_data = []  # Stores last 50 anomaly values for trend visualization

# ------------------- UI UPDATE FUNCTION -------------------
def update_ui(anomalies, normal, accuracy, conf_matrix):
    """Update the UI dynamically"""
    
    global log_df  # Ensure global scope

    # Update the metrics
    total_anomalies.metric("⚠️ Anomalies Detected", anomalies)
    normal_traffic.metric("✅ Normal Traffic", normal)
    accuracy_display.metric("🎯 Model Accuracy", f"{accuracy:.2f}%")

    # Update Persistent Threat Gauge
    total_packets = anomalies + normal
    anomaly_percentage = (anomalies / total_packets) * 100 if total_packets > 0 else 0
    threat_gauge_options["series"][0]["data"][0]["value"] = anomaly_percentage
    st_echarts(threat_gauge_options, height="300px")

    # Update Current Data Gauge (Dashes into Red Briefly)
    current_value = anomaly_percentage if anomalies > 0 else np.random.uniform(10, 40)  # Default normal range
    current_data_gauge_options["series"][0]["data"][0]["value"] = current_value
    st_echarts(current_data_gauge_options, height="300px")
    
    if anomalies > 0:
        time.sleep(0.5)  # Briefly stays in red
        current_data_gauge_options["series"][0]["data"][0]["value"] = np.random.uniform(10, 40)  # Resets to normal
        st_echarts(current_data_gauge_options, height="300px")

    # Update Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="g", cmap="coolwarm", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    conf_matrix_placeholder.pyplot(fig)

    # Update Log
    new_entry = pd.DataFrame({"Packet ID": [len(log_df) + 1], "Prediction": ["Anomaly" if anomalies > 0 else "Normal"]})
    log_df = pd.concat([log_df, new_entry], ignore_index=True)
    log_placeholder.dataframe(log_df.tail(10))  # Show last 10 packets

    # Update Trend Chart
    anomaly_trend_data.append(anomaly_percentage)
    if len(anomaly_trend_data) > 50:  # Keep only last 50 entries
        anomaly_trend_data.pop(0)
    
    trend_chart_placeholder.line_chart(anomaly_trend_data)

# ------------------- INTEGRATION NOTES -------------------
st.markdown("""
**Integration Notes:**
- Call `update_ui(anomalies, normal, accuracy, conf_matrix)` to update the dashboard dynamically.
- The **Threat Level Gauge** shows overall anomaly percentage.
- The **Current Data Gauge** spikes into red **briefly** when an anomaly is detected.
- The **Anomaly Trend Graph** updates in real time to showcase anomaly activity.
""")
