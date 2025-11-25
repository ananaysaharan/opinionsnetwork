import sys
import os
import numpy as np
import random
import time
import multiprocessing
from functools import partial

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from simulation_model import SimulationModel
from abc_utils import calculate_distance

# Target Observed Statistics (S_obs)
S_OBS = {
    "mean_opinion": 0.5284,
    "variance": 0.0118,
    "clustering_coefficient": 0.2537
}

def run_single_simulation(params):
    """
    Worker function to run a single simulation.
    Args:
        params (tuple): (c, mu)
    Returns:
        dict: Result containing params, distance, and metrics, or None if rejected (optional optimization).
    """
    c, mu = params
    
    # Run Simulation
    sim = SimulationModel(n_agents=100, confidence_bound=c, mu=mu)
    sim.run(steps=1000)
    metrics = sim.get_metrics()
    
    # Extract S_sim
    S_sim = {
        "mean_opinion": metrics["average_opinion"],
        "variance": metrics["opinion_variance"],
        "clustering_coefficient": metrics["clustering_coefficient"]
    }
    
    # Calculate Distance
    distance = calculate_distance(S_sim, S_OBS)
    
    return {
        "c": c,
        "mu": mu,
        "distance": distance,
        "metrics": S_sim
    }

def run_parallel_abc(n_iterations=10000, epsilon=0.1, n_processes=None):
    print(f"--- Day 11: Parallelization (The Speed Up) ---")
    print(f"Running {n_iterations} iterations with epsilon={epsilon}...")
    
    if n_processes is None:
        n_processes = multiprocessing.cpu_count()
    print(f"Using {n_processes} CPU cores.")
    
    # Step A: Generate all parameters upfront
    print("Generating parameters...")
    params_list = []
    for _ in range(n_iterations):
        c = random.uniform(0.1, 0.5)
        mu = random.uniform(0.1, 0.9)
        params_list.append((c, mu))
        
    start_time = time.time()
    
    # Step B & C: Run simulations in parallel
    print("Starting parallel pool...")
    with multiprocessing.Pool(processes=n_processes) as pool:
        # map returns results in order
        results = pool.map(run_single_simulation, params_list)
        
    # Step D: Filter results
    accepted_params = [r for r in results if r["distance"] < epsilon]
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n--- Results ---")
    print(f"Total Time: {duration:.2f} seconds")
    print(f"Throughput: {n_iterations / duration:.2f} sims/sec")
    print(f"Accepted: {len(accepted_params)} / {n_iterations} ({len(accepted_params)/n_iterations*100:.2f}%)")
    
    if accepted_params:
        print("\nTop 5 Best Matches:")
        accepted_params.sort(key=lambda x: x["distance"])
        for p in accepted_params[:5]:
            print(f"  c={p['c']:.4f}, mu={p['mu']:.4f} -> dist={p['distance']:.4f}")
            
    # Save to CSV (optional but good for "Deliverable")
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "day11_accepted_params.csv")
    with open(output_path, "w") as f:
        f.write("c,mu,distance\n")
        for p in accepted_params:
            f.write(f"{p['c']},{p['mu']},{p['distance']}\n")
    print(f"\nSaved accepted parameters to {output_path}")

if __name__ == "__main__":
    # 10,000 might take a minute or two depending on the machine.
    # Let's stick to the user request of 10,000.
    run_parallel_abc(n_iterations=10000, epsilon=0.1)
