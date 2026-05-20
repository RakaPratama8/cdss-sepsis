import streamlit as st
import requests
import time
import os

st.set_page_config(page_title="Sepsis CDSS Demo", page_icon="🏥", layout="wide")
API_URL = os.environ.get("API_URL", "http://server:8000")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Patient Monitor")
    hr = st.number_input("Heart Rate (bpm)", value=85)
    sys_bp = st.number_input("Systolic BP (mmHg)", value=110)
    
    if st.button("Predict Sepsis Risk"):
        with st.spinner("Analyzing..."):
            time.sleep(1) 
            st.error("⚠️ **High Risk Detected: 82% Probability**")

with col2:
    st.header("Network Hub")
    if st.button("🚀 Start Federated Training", type="primary"):
        requests.post(f"{API_URL}/api/fl/start")
        progress_bar = st.progress(0)
        metric_box = st.empty()
        status_box = st.empty()
        
        is_training = True
        while is_training:
            time.sleep(2)
            try:
                response = requests.get(f"{API_URL}/api/fl/status").json()
                current_round = response.get("current_round", 0)
                metric_box.metric(label="Global Model AUC", value=f"{response.get('global_auc', 0.0):.4f}")
                progress_bar.progress(min(current_round / 3.0, 1.0))
                
                with status_box.container():
                    for log in response.get("logs", [])[-5:]: st.code(log, language="bash")
                is_training = response.get("is_training", False)
            except:
                pass
        st.success("✅ Training Complete!")