# Add this at the very top of server/main.py
from fastapi import UploadFile, File, Form
from src.preprocessing import SepsisPreprocessor
import os
import numpy as np
from src.model import build_sepsis_gru

# Initialize your pipeline
preprocessor = SepsisPreprocessor()

# Add this at the very bottom of server/main.py
@app.post("/api/predict")
async def predict_sepsis(
    file: UploadFile = File(...), 
    hospital_id: str = Form(...)
):
    try:
        # 1. Route the file through our live inference preprocessing engine
        patient_tensor, feature_names = preprocessor.process_live_inference(file.file, file.filename, hospital_id)
        
        # 2. Check if the Federated Model has actually been trained yet
        weights_path = 'global_model_weights.npz'
        if not os.path.exists(weights_path):
            return {"error": "Global model not found. Please click 'Start Federated Training' first!"}
            
        # 3. Load the Global Architecture and Weights
        time_steps, num_features = patient_tensor.shape[1], patient_tensor.shape[2]
        global_model = build_sepsis_gru((time_steps, num_features))
        
        npzfile = np.load(weights_path)
        global_model.set_weights([npzfile[arr] for arr in npzfile.files])
        
        # 4. Generate the prediction
        risk_prob = global_model.predict(patient_tensor, verbose=0)[0][0]
        
        return {"status": "success", "risk_score": float(risk_prob)}
        
    except Exception as e:
        return {"error": str(e)}