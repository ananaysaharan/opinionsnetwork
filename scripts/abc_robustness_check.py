import sys
import os
import pandas as pd
import numpy as np
import time

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

def run_robustness_check(n_runs=10):
    print("--- Day 13: Robustness Check ---")
    
    # Load accepted parameters
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'abc_accepted_params.csv'))
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
        
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Identify Best Parameters (min distance)
    best_row = df.loc[df['distance'].idxmin()]
    best_c = best_row['c']
    best_mu = best_row['mu']
    min_dist = best_row['distance']
    
    print(f"\nBest Parameters Found:")
    print(f"  c  = {best_c:.4f}")
    print(f"  mu = {best_mu:.4f}")
    print(f"  (Original Distance: {min_dist:.4f})")
    
    print(f"\nRunning {n_runs} simulations to check stability...")
    
    results = []
    start_time = time.time()
    
    for i in range(n_runs):
        sim = SimulationModel(n_agents=100, confidence_bound=best_c, mu=best_mu)
        sim.run(steps=1000)
        metrics = sim.get_metrics()
        
        S_sim = {
            "mean_opinion": metrics["average_opinion"],
            "variance": metrics["opinion_variance"],
            "clustering_coefficient": metrics["clustering_coefficient"]
        }
        
        dist = calculate_distance(S_sim, S_OBS)
        
        results.append({
            "run": i+1,
            "dist": dist,
            "mean_opinion": S_sim["mean_opinion"],
            "variance": S_sim["variance"],
            "clustering_coefficient": S_sim["clustering_coefficient"]
        })
        # print(f"  Run {i+1}: dist={dist:.4f}")
        
    end_time = time.time()
    
    # Convert to DataFrame for easy stats
    res_df = pd.DataFrame(results)
    
    print(f"\n--- Robustness Results ({n_runs} runs) ---")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    print("\nDistance Statistics:")
    print(f"  Mean: {res_df['dist'].mean():.4f}")
    print(f"  Std : {res_df['dist'].std():.4f}")
    print(f"  Min : {res_df['dist'].min():.4f}")
    print(f"  Max : {res_df['dist'].max():.4f}")
    
    print("\nMetric Stability (Mean ± Std):")
    print(f"  Mean Opinion : {res_df['mean_opinion'].mean():.4f} ± {res_df['mean_opinion'].std():.4f} (Target: {S_OBS['mean_opinion']})")
    print(f"  Variance     : {res_df['variance'].mean():.4f} ± {res_df['variance'].std():.4f} (Target: {S_OBS['variance']})")
    print(f"  Clustering   : {res_df['clustering_coefficient'].mean():.4f} ± {res_df['clustering_coefficient'].std():.4f} (Target: {S_OBS['clustering_coefficient']})")
    
    # Conclusion
    if res_df['dist'].mean() < 0.1 and res_df['dist'].std() < 0.05:
        print("\nCONCLUSION: ROBUST. The model consistently reproduces the target behavior.")
    else:
        print("\nCONCLUSION: UNSTABLE. The results vary significantly between runs.")

if __name__ == "__main__":
    run_robustness_check()
