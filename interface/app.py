import streamlit as st
import requests
import time
import os

st.set_page_config(page_title="CDSS Admin | FL Server", page_icon="⚙️", layout="centered")
API_URL = os.environ.get("API_URL", "http://server:8000")

st.title("⚙️ Central Server Admin")
st.markdown("Manage the Federated Learning orchestration across all hospital nodes.")

if st.button("🚀 Start Federated Training", type="primary", use_container_width=True):
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
                for log in response.get("logs", [])[-5:]: 
                    st.code(log, language="bash")
                    
            is_training = response.get("is_training", False)
        except Exception:
            pass
            
    st.success("✅ Federated Training Complete! Global Model is now distributed to all hospitals.")