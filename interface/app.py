import streamlit as st
import requests
import time
import os

st.set_page_config(page_title="Sepsis CDSS Demo", page_icon="🏥", layout="wide")
API_URL = os.environ.get("API_URL", "http://server:8000")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Patient Inference")
    
    # 1. Hospital Selector (Crucial for loading the correct scaler)
    hospital_id = st.selectbox("Select Hospital Site:", ["hospital_a", "hospital_b", "hospital_c"])
    
    # 2. File Uploader for the 72-hour sequence
    uploaded_file = st.file_uploader("Upload Patient Vitals (CSV/Excel)", type=["csv", "xlsx"])
    
    if st.button("Predict Sepsis Risk"):
        if uploaded_file is not None:
            with st.spinner("Processing 72-hour sequence and running inference..."):
                # Prepare the file and data to send to FastAPI
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"hospital_id": hospital_id}
                
                try:
                    response = requests.post(f"{API_URL}/api/predict", files=files, data=data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if "error" in result:
                            st.warning(f"⚠️ {result['error']}")
                        else:
                            risk_score = result.get("risk_score", 0.0)
                            st.subheader("Results")
                            if risk_score > 0.5:
                                st.error(f"🚨 **High Risk Detected: {risk_score * 100:.1f}% Probability of Sepsis**")
                            else:
                                st.success(f"✅ **Low Risk: {risk_score * 100:.1f}% Probability of Sepsis**")
                    else:
                        st.error(f"Backend Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")
        else:
            st.warning("Please upload a patient .csv or .xlsx file first.")

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