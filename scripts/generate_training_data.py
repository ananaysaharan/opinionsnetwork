import sys
import os
import pandas as pd
import numpy as np
import time
import random

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from simulation_model import SimulationModel

def generate_data(n_samples=5000, steps=100):
    print(f"--- Day 15: The Data Factory ---")
    print(f"Generating {n_samples} samples with {steps} steps each...")
    
    data = []
    
    start_time = time.time()
    
    for i in range(n_samples):
        # 1. Sample random parameters
        # c (confidence_bound) typically in [0, 1]
        # mu (convergence parameter) typically in [0, 0.5]
        c = np.random.uniform(0.01, 1.0)
        mu = np.random.uniform(0.01, 0.5)
        
        # 2. Initialize Model
        sim = SimulationModel(n_agents=100, confidence_bound=c, mu=mu)
        
        # 3. Capture t0 state
        # SimulationModel initializes with random opinions
        t0_ops = [sim.G.nodes[n]['opinion'] for n in sim.G.nodes()]
        mean_t0 = np.mean(t0_ops)
        
        # 4. Run Simulation
        sim.run(steps=steps)
        
        # 5. Collect Trajectory
        # sim.history contains lists of opinions at each step t1..t100
        trajectory = [mean_t0]
        for step_ops in sim.history:
            trajectory.append(np.mean(step_ops))
            
        # trajectory now has 101 values (t0 to t100)
        
        # 6. Store Row
        row = {
            "input_c": c,
            "input_mu": mu
        }
        for t, val in enumerate(trajectory):
            row[f"output_opinion_t{t}"] = val
            
        data.append(row)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"Generated {i + 1}/{n_samples} samples... ({elapsed:.2f}s)")
            
    # 7. Save to CSV
    print("Saving to CSV...")
    df = pd.DataFrame(data)
    
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'surrogate_training_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Data saved to {output_path}")
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    generate_data()
