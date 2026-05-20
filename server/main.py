import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import flwr as fl

# --- FASTAPI SETUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

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
    This separates the ML configuration from the execution layer.
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
    
    # 1. Extract components from our modern server_fn
    dummy_context = fl.common.Context(node_id=0, node_config={})
    components = server_fn(dummy_context)
    
    # 2. Run the legacy engine to maintain compatibility with the API wrapper
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
    """Triggered by the Streamlit Dashboard."""
    if not fl_state["is_training"]:
        threading.Thread(target=run_flower_server, daemon=True).start()
    return {"message": "Server started"}

@app.get("/api/fl/status")
def get_status():
    """Polled by the Streamlit Dashboard every 2 seconds."""
    return fl_state