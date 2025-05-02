import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_echarts import st_echarts
import time
from sklearn.metrics import confusion_matrix, accuracy_score

# Streamlit Page Configuration
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.markdown("<h1>Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h3>🔍 Live 5G Threat Monitoring</h3>", unsafe_allow_html=True)

# Global Variables
log_df = pd.DataFrame(columns=["Packet ID", "Prediction", "Timestamp", "Probability"])
anomaly_trend_data = []
cumulative_true = []
cumulative_pred = []
total_anomalies_count = 0  # Track cumulative anomalies
total_normal_count = 0     # Track cumulative normal traffic
render_counter = 0         # For unique chart keys

# ------------------- METRICS DISPLAY -------------------
col1, col2, col3 = st.columns(3)
with col1:
    total_anomalies = st.metric("Anomalies Detected", 0)
with col2:
    normal_traffic = st.metric("Normal Traffic", 0)
with col3:
    accuracy_display = st.metric("Model Accuracy", "N/A")

# ------------------- METRIC INDICATORS (REPLACING GAUGES) -------------------
st.subheader("📊 Threat Assessment")
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    threat_level_value = st.empty()
    threat_level_label = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
with col_m2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    current_data_value = st.empty()
    current_data_label = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------- OTHER PLACEHOLDERS -------------------
st.subheader("🔍 Confusion Matrix")
conf_matrix_placeholder = st.empty()

st.subheader("📜 Classification Log")
log_placeholder = st.empty()

st.subheader("📈 Anomaly Trend Over Time")
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
trend_chart_placeholder = st.empty()
st.markdown('</div>', unsafe_allow_html=True)

# ------------------- UI UPDATE FUNCTION -------------------
def update_ui(anomalies, normal, accuracy, conf_matrix, probabilities):
    """Update the UI dynamically with new data."""
    global log_df, anomaly_trend_data, render_counter, total_anomalies_count, total_normal_count
    render_counter += 1  # Increment for unique keys

    # Update cumulative counts
    total_anomalies_count += anomalies
    total_normal_count += normal

    # Update metrics
    total_anomalies.metric("Anomalies Detected", total_anomalies_count)
    normal_traffic.metric("Normal Traffic", total_normal_count)
    accuracy_display.metric("Model Accuracy", f"{accuracy:.2f}%")

    # Calculate anomaly percentage
    total_packets = total_anomalies_count + total_normal_count
    anomaly_percentage = (total_anomalies_count / total_packets) * 100 if total_packets > 0 else 0
    anomaly_percentage = round(anomaly_percentage, 2)  # Round to 2 decimals

    # Update threat level with simple metrics
    threat_level = "HIGH" if anomaly_percentage > 70 else "MEDIUM" if anomaly_percentage > 30 else "LOW"
    threat_color = "#d73027" if anomaly_percentage > 70 else "#fdae61" if anomaly_percentage > 30 else "#1a9850"
    
    threat_level_value.markdown(f"<div class='metric-value' style='color: {threat_color};'>{anomaly_percentage:.2f}%</div>", unsafe_allow_html=True)
    threat_level_label.markdown(f"<div class='metric-label'>Threat Level: {threat_level}</div>", unsafe_allow_html=True)
    
    # Update current data with simple metrics
    if anomalies > 0:
        current_value = round(np.random.uniform(60, 90), 2)  # Higher for anomalies
        current_status = "ANOMALY DETECTED"
        current_color = "#d73027"
    else:
        current_value = round(np.random.uniform(10, 40), 2)  # Lower for normal
        current_status = "Normal Traffic"
        current_color = "#1a9850"
        
    current_data_value.markdown(f"<div class='metric-value' style='color: {current_color};'>{current_value:.2f}%</div>", unsafe_allow_html=True)
    current_data_label.markdown(f"<div class='metric-label'>{current_status}</div>", unsafe_allow_html=True)

    # Update Confusion Matrix
    fig, ax = plt.subplots(figsize=(3, 3))
    cm = np.array(conf_matrix)
    if cm.shape != (2, 2):
        fixed_cm = np.zeros((2, 2), dtype=int)
        for i in range(min(2, cm.shape[0])):
            for j in range(min(2, cm.shape[1])):
                fixed_cm[i, j] = cm[i, j]
        cm = fixed_cm
    sns.heatmap(cm, annot=True, fmt="g", cmap="coolwarm", ax=ax, cbar=False, annot_kws={"size": 10})
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_xticklabels(["Normal", "Anomaly"], fontsize=8)
    ax.set_yticklabels(["Normal", "Anomaly"], fontsize=8)
    plt.tight_layout()
    conf_matrix_placeholder.pyplot(fig, use_container_width=False)

    # Update Classification Log
    new_entry = pd.DataFrame({
        "Packet ID": [len(log_df) + 1],
        "Prediction": ["Anomaly" if anomalies > 0 else "Normal"],
        "Timestamp": [pd.Timestamp.now().strftime("%H:%M:%S")],
        "Probability": [np.mean(probabilities) * 100 if anomalies > 0 else 0]
    })
    log_df = pd.concat([log_df, new_entry], ignore_index=True)
    log_placeholder.dataframe(log_df.tail(10), use_container_width=True)

    # Update Anomaly Trend Chart
    anomaly_trend_data.append(anomaly_percentage)
    if len(anomaly_trend_data) > 50:
        anomaly_trend_data.pop(0)
    trend_chart_options = {
        "backgroundColor": "#0E1117",
        "xAxis": {"type": "category", "data": list(range(len(anomaly_trend_data)))},
        "yAxis": {"type": "value", "name": "Anomaly %"},
        "series": [{"data": anomaly_trend_data, "type": "line", "smooth": True, "lineStyle": {"color": "#FFA500"}}],
        "tooltip": {"trigger": "axis"},
        "animation": True,
        "animationDuration": 300,
        "animationEasing": "cubicOut"
    }
    with trend_chart_placeholder:
        st_echarts(trend_chart_options, height="300px", key=f"trend_chart_{render_counter}")

# ------------------- DRIVER FUNCTION INTEGRATION -------------------
try:
    from hybrid_tree import classify_data, SELECTED_FEATURES
except ImportError:
    st.error("❌ hybrid_tree.py not found or has errors.")
    st.stop()

BATCH_SIZE = 10000

try:
    full_data = pd.read_csv("dataset.csv", low_memory=False)
    full_data.columns = full_data.columns.str.strip()
    st.success("✅ Dataset loaded successfully.")
except FileNotFoundError:
    st.error("❌ dataset.csv not found.")
    full_data = pd.DataFrame()

def is_batch_valid(df):
    missing = [f for f in SELECTED_FEATURES if f not in df.columns]
    if missing:
        print(f"Missing columns in batch: {missing}")
    return all(feature in df.columns for feature in SELECTED_FEATURES)

# User Controls
st.subheader("🎮 Simulation Controls")
col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
with col_ctrl1:
    start_button = st.button("▶️ Start Simulation")
with col_ctrl2:
    pause_button = st.button("⏸️ Pause Simulation")
with col_ctrl3:
    reset_button = st.button("🔄 Reset Simulation")
with col_ctrl4:
    suppress_warnings = st.checkbox("Suppress Threat Warnings", value=False)

simulation_delay = st.slider("Simulation Speed (seconds)", 0.1, 2.0, 1.0)

if reset_button:
    log_df = pd.DataFrame(columns=["Packet ID", "Prediction", "Timestamp", "Probability"])
    anomaly_trend_data = []
    cumulative_true = []
    cumulative_pred = []
    total_anomalies_count = 0
    total_normal_count = 0
    update_ui(0, 0, 0, [[0, 0], [0, 0]], [0])
    st.success("✅ Dashboard reset.")

if start_button:
    st.info("🟡 Simulation running...")
    for i in range(0, len(full_data), BATCH_SIZE):
        if pause_button:
            st.warning("⏸️ Simulation paused.")
            break
        batch = full_data.iloc[i:i+BATCH_SIZE].copy()
        batch.columns = batch.columns.str.strip()

        # More robust check for valid batch
        if batch.empty or not all(feature in batch.columns for feature in SELECTED_FEATURES):
            st.warning(f"⚠️ Skipping batch {i} due to missing essential columns.")
            continue

        # Further check for all NaN columns in selected features
        nan_check = batch[SELECTED_FEATURES].isna().all()
        if nan_check.any():
            nan_cols = nan_check[nan_check].index.tolist()
            st.warning(f"⚠️ Skipping batch {i} as all values are NaN in columns: {nan_cols}")
            continue

        try:
            predictions, probabilities = classify_data(batch.copy()) # Pass a copy
            print(f"Batch {i} probabilities:", probabilities)
        except Exception as e:
            st.error(f"Prediction failed for batch {i}: {e}")
            print(f"Batch {i} failed: {e}")
            continue

        y_true = batch['Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1).astype(int).values
        cumulative_true.extend(y_true.tolist())
        cumulative_pred.extend(predictions.tolist())

        anomalies = int(np.sum(predictions))
        normal = len(predictions) - anomalies

        if cumulative_true:
            # Handle potential single label issue in confusion matrix
            labels = np.unique(cumulative_true + cumulative_pred)
            cm = confusion_matrix(cumulative_true, cumulative_pred, labels=labels)
            acc = accuracy_score(cumulative_true, cumulative_pred) * 100
        else:
            acc = 0
            cm = np.array([[0, 0], [0, 0]])

        update_ui(anomalies, normal, acc, cm, probabilities)

        if not suppress_warnings and np.mean(probabilities) * 100 > 80:
            st.warning("🚨 High Threat Level Detected!")

        time.sleep(simulation_delay)

    st.success("✅ Simulation complete.")

# ------------------- INTEGRATION NOTES -------------------
st.markdown("""
**Integration Notes:**
- Call `update_ui(anomalies, normal, accuracy, conf_matrix, probabilities)` to update the dashboard dynamically.
- The **Threat Assessment** shows current threat level with color-coded indicators.
- The **Anomaly Trend Graph** updates in real-time to showcase anomaly activity.
- The **Confusion Matrix** displays classification accuracy metrics.
""")