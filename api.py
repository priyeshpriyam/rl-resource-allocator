import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from stable_baselines3 import PPO

# --- 1. DEFINE DATA MODELS ---
# Pydantic models for request and response data validation
class CloudState(BaseModel):
    normalized_servers: float
    current_load: float

class ActionResponse(BaseModel):
    action_code: int
    action_name: str

# --- 2. INITIALIZE THE APP AND LOAD THE MODEL ---
# Create the FastAPI application
app = FastAPI(title="RL Resource Allocator API")

# Load your trained "champion" model
# This model is loaded only once when the application starts.
model = PPO.load("ppo_tuned_cloud_allocator.zip")
print("Model ppo_tuned_cloud_allocator.zip loaded successfully.")

# Define a mapping from numeric action to a human-readable name
ACTION_MAP = {0: "DO_NOTHING", 1: "ADD_SERVER", 2: "REMOVE_SERVER"}


# --- 3. DEFINE THE PREDICTION ENDPOINT ---
@app.post("/predict", response_model=ActionResponse)
def predict_action(state: CloudState):
    """
    Accepts the current cloud state and returns the optimal action
    predicted by the Reinforcement Learning agent.
    """
    # Convert the incoming data into a NumPy array that the model expects
    observation = np.array(
        [state.normalized_servers, state.current_load], 
        dtype=np.float32
    ).reshape(1, -1) # Reshape for a single prediction

    # Get the prediction from the model
    action, _ = model.predict(observation, deterministic=True)
    
    # Convert the numeric action to an integer and get its name
    action_code = int(action[0])
    action_name = ACTION_MAP.get(action_code, "UNKNOWN_ACTION")
    
    # Return the action code and name
    return {"action_code": action_code, "action_name": action_name}

@app.get("/")
def read_root():
    return {"message": "Welcome to the RL Cloud Resource Allocator API. Go to /docs to see the interactive API documentation."}