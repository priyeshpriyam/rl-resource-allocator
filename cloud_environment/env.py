import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CloudEnv(gym.Env):
    """Custom Environment for Cloud Resource Allocation."""
    
    def __init__(self):
        super(CloudEnv, self).__init__()
        
        self.max_servers = 10
        
        # Define the action space: 0=Do Nothing, 1=Add Server, 2=Remove Server
        self.action_space = spaces.Discrete(3)
        
        # Define the observation space (state): [normalized_server_count, current_load]
        # UPDATED to reflect the new state representation
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([1, np.inf]), # Load can technically exceed 1 before penalty
            dtype=np.float32
        )
        
        # Initialize state variables
        self.active_servers = 1
        self.current_load = 0.0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.active_servers = 1
        self.current_load = 0.0
        self.step_count = 0
        
        # The observation must match the defined space
        normalized_servers = self.active_servers / self.max_servers
        observation = np.array([normalized_servers, self.current_load], dtype=np.float32)
        info = {}
        return observation, info

    def step(self, action):
        # --- THIS IS THE UPDATED LOGIC ---

        # 1. Apply the action from the agent
        if action == 1: # Add a server
            self.active_servers = min(self.max_servers, self.active_servers + 1)
        elif action == 2: # Remove a server
            self.active_servers = max(1, self.active_servers - 1)
        
        # 2. Simulate more realistic, cyclical traffic
        time_of_day_factor = np.sin(2 * np.pi * self.step_count / 24.0) # 24 steps = 1 simulated day
        base_traffic = 5 
        noise = np.random.uniform(0.8, 1.2)
        incoming_requests = base_traffic * (1 + 0.5 * time_of_day_factor) * noise
        
        # 3. Calculate the new system load
        server_capacity = self.active_servers * 1.0
        self.current_load = incoming_requests / server_capacity if server_capacity > 0 else 0
        
        # 4. Calculate the new, smarter reward
        ideal_load = 0.6
        # a. Cost Penalty: For each active server.
        cost_penalty = -self.active_servers
        # b. Load Penalty: A quadratic penalty for deviating from the ideal load.
        load_penalty = -10 * ((self.current_load - ideal_load) ** 2)
        # c. Overload Penalty: A large, sharp penalty for exceeding capacity.
        overload_penalty = -200 if self.current_load > 1.0 else 0
        
        reward = cost_penalty + load_penalty + overload_penalty
        
        # 5. Determine if the episode is done
        self.step_count += 1
        terminated = False
        truncated = self.step_count >= 100
        
        # 6. Form the new observation
        normalized_servers = self.active_servers / self.max_servers
        observation = np.array([normalized_servers, self.current_load], dtype=np.float32)
        
        info = {}
        
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        # A simple printout of the current state
        print(f"Step: {self.step_count}, Servers: {self.active_servers}, Load: {self.current_load:.2f}")