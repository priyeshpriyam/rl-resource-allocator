# Dynamic Cloud Resource Allocation using Deep Reinforcement Learning

This project features an autonomous AI agent that learns to dynamically manage a cluster of cloud servers. The agent's goal is to intelligently add or remove servers to minimize costs while ensuring high performance and reliability, outperforming traditional rule-based systems.

[![Live API Docs](https://img.shields.io/badge/API-Live%20Demo-brightgreen)](https://jethajii-rl-resource-allocator.hf.space/docs)

---

## Project Overview

Traditional cloud auto-scaling relies on simple heuristics (e.g., "if CPU > 80%, add a server"). This project takes a more advanced approach using Deep Reinforcement Learning. An agent is trained in a custom-built simulation environment to learn a sophisticated policy that anticipates traffic patterns and makes proactive scaling decisions.

The final trained agent was deployed as a containerized web API using FastAPI and Docker, and is hosted live on Hugging Face Spaces.

## Key Features

* **Custom RL Environment:** A simulation of a cloud server cluster built from scratch using the Python Gymnasium library.
* **Intelligent Agent:** A Deep RL agent trained with the Proximal Policy Optimization (PPO) algorithm from Stable-Baselines3.
* **Performance Benchmarking:** The agent's performance is quantitatively compared against a traditional rule-based heuristic policy.
* **Hyperparameter Tuning:** Automated hyperparameter optimization was performed using Optuna to find the most effective agent configuration.
* **API & Deployment:** The final agent is served via a REST API built with FastAPI and deployed as a Docker container to a public URL on Hugging Face Spaces.

## Results

The tuned RL agent learned a proactive, "safety-first" strategy. It maintains a higher number of active servers during peak traffic to create a safety buffer, resulting in significantly fewer overload events and a more stable server load compared to the reactive heuristic agent.

![Tuned RL Agent Performance](final_tuned_rl_perf.png)

## Tech Stack

* **Core ML:** Python, PyTorch, Stable-Baselines3, Gymnasium, Optuna
* **API & Deployment:** FastAPI, Uvicorn, Docker
* **Hosting:** Hugging Face Spaces
* **Data Handling:** NumPy

## Local Setup and Usage

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/priyeshpriyam/rl-resource-allocator.git](https://github.com/priyeshpriyam/rl-resource-allocator.git)
    cd rl-resource-allocator
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **To train your own model:**
    ```bash
    python train_final_agent.py
    ```

5.  **To run the local API server:**
    ```bash
    python -m uvicorn api:app --reload
    ```
    You can then access the local API docs at `http://127.0.0.1:8000/docs`.
