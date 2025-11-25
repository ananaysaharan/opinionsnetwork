import sys
import os
import numpy as np
import random
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from simulation_model import SimulationModel
from abc_utils import calculate_distance

def run_rejection_sampling(n_iterations=1000, epsilon=0.1):
    print(f"--- Day 10: The Rejection Loop (Sequential) ---")
    print(f"Running {n_iterations} iterations with epsilon={epsilon}...")
    
    # Target Observed Statistics (S_obs)
    # Using the values we found in Day 8/9 as our "Real Data"
    S_obs = {
        "mean_opinion": 0.5284,
        "variance": 0.0118,
        "clustering_coefficient": 0.2537
    }
    print(f"Target S_obs: {S_obs}")
    
    accepted_params = []
    start_time = time.time()
    
    for i in range(n_iterations):
        # Step A: Sample random parameters (Priors)
        # c in [0.1, 0.5]
        c = random.uniform(0.1, 0.5)
        # mu in [0.1, 0.9]
        mu = random.uniform(0.1, 0.9)
        
        # Step B: Run Simulation
        # Using n_agents=100, steps=1000 as standard
        sim = SimulationModel(n_agents=100, confidence_bound=c, mu=mu)
        sim.run(steps=1000)
        metrics = sim.get_metrics()
        
        # Extract S_sim
        S_sim = {
            "mean_opinion": metrics["average_opinion"],
            "variance": metrics["opinion_variance"],
            "clustering_coefficient": metrics["clustering_coefficient"]
        }
        
        # Step C: Calculate Distance
        distance = calculate_distance(S_sim, S_obs)
        
        # Step D: Reject or Accept
        if distance < epsilon:
            print(f"[{i+1}/{n_iterations}] ACCEPTED: c={c:.4f}, mu={mu:.4f} (dist={distance:.4f})")
            accepted_params.append({
                "c": c,
                "mu": mu,
                "distance": distance,
                "metrics": S_sim
            })
        else:
            # Optional: Print progress every 100 iterations
            if (i+1) % 100 == 0:
                print(f"[{i+1}/{n_iterations}] Rejected (dist={distance:.4f})")
                
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n--- Results ---")
    print(f"Total Time: {duration:.2f} seconds")
    print(f"Accepted: {len(accepted_params)} / {n_iterations} ({len(accepted_params)/n_iterations*100:.2f}%)")
    
    if accepted_params:
        print("\nTop 5 Best Matches:")
        # Sort by distance
        accepted_params.sort(key=lambda x: x["distance"])
        for p in accepted_params[:5]:
            print(f"  c={p['c']:.4f}, mu={p['mu']:.4f} -> dist={p['distance']:.4f}")
            
    return accepted_params

if __name__ == "__main__":
    # Run a smaller batch for quick verification, or full 1000 if fast enough
    # 1000 iterations might take a bit, let's try 100 for the demo to be quick
    run_rejection_sampling(n_iterations=100, epsilon=0.1)
