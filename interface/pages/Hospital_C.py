import streamlit as st
import requests
import os

st.set_page_config(page_title="Hospital C | Sepsis CDSS", page_icon="🏥", layout="wide")
API_URL = os.environ.get("API_URL", "http://server:8000")

# CHANGE THIS FOR EACH FILE ("hospital_a", "hospital_b", "hospital_c")
HOSPITAL_ID = "hospital_c"

st.title("🏥 Hospital C: Sepsis Prediction")
st.markdown("Upload a 72-hour patient sequence to evaluate Domain Shift between the Local and Global models.")

uploaded_file = st.file_uploader("Upload Patient Vitals (CSV/Excel)", type=["csv", "xlsx"])

if st.button("Run Dual-Model Inference", type="primary"):
    if uploaded_file is not None:
        with st.spinner("Processing 72-hour sequence and running inference..."):
            # We send the same file twice, requesting different models
            file_data = uploaded_file.getvalue()
            
            try:
                # 1. Fetch Local Prediction
                files_local = {"file": (uploaded_file.name, file_data, uploaded_file.type)}
                res_local = requests.post(f"{API_URL}/api/predict", files=files_local, data={"hospital_id": HOSPITAL_ID, "model_type": "local"})
                
                # 2. Fetch Global Prediction
                files_global = {"file": (uploaded_file.name, file_data, uploaded_file.type)}
                res_global = requests.post(f"{API_URL}/api/predict", files=files_global, data={"hospital_id": HOSPITAL_ID, "model_type": "global"})
                
                if res_local.status_code == 200 and res_global.status_code == 200:
                    data_local = res_local.json()
                    data_global = res_global.json()
                    
                    if "error" in data_local: st.error(f"Local Model Error: {data_local['error']}")
                    if "error" in data_global: st.error(f"Global Model Error: {data_global['error']}")
                    
                    # Display Side-by-Side Comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🏠 Local Model (Siloed)")
                        st.caption("Trained ONLY on this hospital's historical data.")
                        local_score = data_local.get("risk_score", 0.0)
                        if local_score > 0.5:
                            st.error(f"🚨 **Risk: {local_score * 100:.1f}%**")
                        else:
                            st.success(f"✅ **Risk: {local_score * 100:.1f}%**")
                            
                    with col2:
                        st.subheader("🌍 Global Model (Federated)")
                        st.caption("Trained collaboratively across all hospital networks.")
                        global_score = data_global.get("risk_score", 0.0)
                        if global_score > 0.5:
                            st.error(f"🚨 **Risk: {global_score * 100:.1f}%**")
                        else:
                            st.success(f"✅ **Risk: {global_score * 100:.1f}%**")
                else:
                    st.error("Backend failed to respond correctly.")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
    else:
        st.warning("Please upload a patient file first.")