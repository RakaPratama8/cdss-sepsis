import os
import time
import flwr as fl
import numpy as np
from src.model import build_sepsis_gru 

class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id):
        self.hospital_id = hospital_id
        print(f"Initializing Client for {hospital_id.upper()}...")
        
        self.X_train = np.load(f'/data/{hospital_id}/demo_X_{hospital_id}.npy')
        self.y_train = np.load(f'/data/{hospital_id}/demo_y_{hospital_id}.npy')
        
        time_steps, num_features = self.X_train.shape[1], self.X_train.shape[2]
        self.model = build_sepsis_gru((time_steps, num_features))

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=3, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, auc = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        return loss, len(self.X_train), {"auc": auc}

# --- NEW FLOWER NEXT API ---
def client_fn(context: fl.common.Context) -> fl.client.Client:
    """
    Constructs and returns the Client instance. 
    The Context object allows for node-specific configurations in the future.
    """
    hospital_id = os.environ.get("HOSPITAL_ID", "hospital_a")
    return HospitalClient(hospital_id).to_client()

# Define the formal ClientApp as per the Quickstart
app = fl.client.ClientApp(client_fn=client_fn)

# --- BACKWARD COMPATIBILITY FOR DOCKER ---
if __name__ == "__main__":
    hospital_id = os.environ.get("HOSPITAL_ID", "hospital_a")
    print(f"[{hospital_id.upper()}] Standing by for central server...")
    
    while True:
        try:
            # We trigger the client_fn manually to keep your Compose orchestration working
            dummy_context = fl.common.Context(node_id=1, node_config={})
            fl.client.start_client(
                server_address="server:8080", 
                client=client_fn(dummy_context)
            )
            print("Training complete. Disconnecting.")
            break 
        except Exception:
            time.sleep(5)