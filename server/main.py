import os
import threading
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import flwr as fl

# Import your custom modules
from src.preprocessing import SepsisPreprocessor
from src.model import build_sepsis_gru

# --- FASTAPI SETUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Initialize your pipeline engine
preprocessor = SepsisPreprocessor()

# Global state to feed live metrics to the Streamlit UI
fl_state = {
    "is_training": False, 
    "current_round": 0, 
    "global_auc": 0.0, 
    "logs": ["Server initialized. Waiting for trigger..."]
}

# --- FLOWER CUSTOM STRATEGY ---
class CDSSStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        if aggregated_metrics:
            _, metrics_dict = aggregated_metrics
            global_auc = metrics_dict.get("auc", 0.0)
            
            # Update API state so Streamlit can poll it live
            fl_state["current_round"] = server_round
            fl_state["global_auc"] = global_auc
            fl_state["logs"].append(f"Round {server_round} complete. Global AUC: {global_auc:.4f}")
        return aggregated_metrics

# --- NEW FLOWER NEXT API ---
def server_fn(context: fl.common.Context) -> fl.server.ServerAppComponents:
    """
    Constructs the Server components using the modern Context API.
    """
    strategy = CDSSStrategy(
        fraction_fit=1.0, 
        fraction_evaluate=1.0, 
        min_fit_clients=3, 
        min_available_clients=3
    )
    config = fl.server.ServerConfig(num_rounds=3)
    return fl.server.ServerAppComponents(strategy=strategy, config=config)

# Define the formal ServerApp as per the Quickstart
server_app = fl.server.ServerApp(server_fn=server_fn)

# --- BACKWARD COMPATIBILITY FOR FASTAPI THREADING ---
def run_flower_server():
    """Runs Flower in a background thread triggered by the Streamlit UI."""
    fl_state["is_training"] = True
    fl_state["logs"].append("Flower Server started. Waiting for 3 hospitals to connect...")
    
    dummy_context = fl.common.Context(node_id=0, node_config={})
    components = server_fn(dummy_context)
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=components.config,
        strategy=components.strategy,
    )
    
    fl_state["is_training"] = False
    fl_state["logs"].append("Federated Training Complete!")

# --- FASTAPI ENDPOINTS ---
@app.post("/api/fl/start")
def start_federated_learning():
    """Triggered by the Admin Dashboard."""
    if not fl_state["is_training"]:
        threading.Thread(target=run_flower_server, daemon=True).start()
    return {"message": "Server started"}

@app.get("/api/fl/status")
def get_status():
    """Polled by the Admin Dashboard every 2 seconds."""
    return fl_state

@app.post("/api/predict")
async def predict_sepsis(
    file: UploadFile = File(...), 
    hospital_id: str = Form(...),
    model_type: str = Form(...) 
):
    """
    Handles live inference for both Local and Global models 
    to demonstrate Domain Shift in the multi-page UI.
    """
    try:
        # 1. Route the file through our live inference preprocessing engine
        patient_tensor, feature_names = preprocessor.process_live_inference(file.file, file.filename, hospital_id)
        
        # 2. Determine which model weights to load
        if model_type == "global":
            weights_path = 'global_model_weights.npz'
        elif model_type == "local":
            # Assumes local models are saved in a volume or root directory accessible to the server
            weights_path = f'local_weights_{hospital_id}.npz' 
        else:
            return {"error": "Invalid model_type specified. Use 'local' or 'global'."}
            
        # 3. Check if the requested model actually exists
        if not os.path.exists(weights_path):
            return {"error": f"Weights not found at '{weights_path}'. Ensure the model is trained!"}
            
        # 4. Load the Architecture and Weights
        time_steps, num_features = patient_tensor.shape[1], patient_tensor.shape[2]
        model = build_sepsis_gru((time_steps, num_features))
        
        npzfile = np.load(weights_path)
        model.set_weights([npzfile[arr] for arr in npzfile.files])
        
        # 5. Generate the prediction
        risk_prob = model.predict(patient_tensor, verbose=0)[0][0]
        
        return {
            "status": "success", 
            "risk_score": float(risk_prob), 
            "model_used": model_type
        }
        
    except Exception as e:
        return {"error": str(e)}