import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from simulation_model import SimulationModel
from surrogate_model import SurrogateModel

def validate_model():
    print("--- Day 19: Validation (The Flex) ---")
    
    # Parameters for validation
    c_val = 0.3
    mu_val = 0.1 # Picking a reasonable mu
    print(f"Testing with c={c_val}, mu={mu_val}")
    
    # 1. Run Actual ABM
    print("Running Actual ABM...")
    sim = SimulationModel(n_agents=100, confidence_bound=c_val, mu=mu_val)
    
    # Capture t0
    t0_ops = [sim.G.nodes[n]['opinion'] for n in sim.G.nodes()]
    mean_t0 = np.mean(t0_ops)
    
    sim.run(steps=100)
    
    abm_trajectory = [mean_t0]
    for step_ops in sim.history:
        abm_trajectory.append(np.mean(step_ops))
        
    # abm_trajectory has 101 points (t0..t100). 
    # The RNN predicts t1..t100 (100 points).
    # We should compare t1..t100.
    abm_trajectory_t1_t100 = abm_trajectory[1:]
    
    # 2. Run RNN Surrogate
    print("Running RNN Surrogate...")
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    model_path = os.path.join(output_dir, 'surrogate_model.pth')
    scaler_x_path = os.path.join(output_dir, 'surrogate_scaler_x.pkl')
    scaler_y_path = os.path.join(output_dir, 'surrogate_scaler_y.pkl')
    
    if not os.path.exists(model_path):
        print("Error: Model not found.")
        return
        
    # Load Scalers
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    
    # Prepare Input
    input_params = np.array([[c_val, mu_val]])
    input_scaled = scaler_x.transform(input_params)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    # Load Model
    model = SurrogateModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Predict
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    # Inverse Scale Output
    output_scaled = output_tensor.numpy()
    rnn_trajectory = scaler_y.inverse_transform(output_scaled)[0]
    
    # 3. Plot Comparison
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    
    steps = range(1, 101)
    plt.plot(steps, abm_trajectory_t1_t100, label='Actual ABM', color='blue', linewidth=2)
    plt.plot(steps, rnn_trajectory, label='RNN Prediction', color='red', linestyle='--', linewidth=2)
    
    plt.title(f'ABM vs RNN Surrogate (c={c_val}, mu={mu_val})')
    plt.xlabel('Time Step')
    plt.ylabel('Mean Opinion')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, 'surrogate_validation.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Calculate Error
    mse = np.mean((abm_trajectory_t1_t100 - rnn_trajectory)**2)
    print(f"Mean Squared Error: {mse:.6f}")
    
    if mse < 0.01:
        print("SUCCESS: The lines are close! You have cloned the simulation.")
    else:
        print("WARNING: The lines might be diverging. Check the plot.")

if __name__ == "__main__":
    validate_model()
