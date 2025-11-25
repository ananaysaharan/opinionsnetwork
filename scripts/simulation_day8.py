import sys
import os
import numpy as np
import networkx as nx

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from simulation_model import SimulationModel

def calculate_summary_statistics(n_agents=100, confidence_bound=0.3, mu=0.5, steps=1000):
    """
    Runs a simulation and calculates the summary statistics (S_obs).
    """
    print(f"Running simulation to generate S_obs (c={confidence_bound}, mu={mu})...")
    
    # Initialize model
    sim = SimulationModel(n_agents=n_agents, confidence_bound=confidence_bound, mu=mu)
    
    # Run simulation
    sim.run(steps=steps)
    
    # Get metrics
    metrics = sim.get_metrics()
    
    # Extract S_obs components
    final_mean_opinion = metrics["average_opinion"]
    opinion_variance = metrics["opinion_variance"]
    clustering_coefficient = metrics["clustering_coefficient"]
    
    print("\n--- Summary Statistics (S_obs) ---")
    print(f"Final Mean Opinion: {final_mean_opinion:.4f}")
    print(f"Opinion Variance: {opinion_variance:.4f}")
    print(f"Clustering Coefficient: {clustering_coefficient:.4f}")
    
    return {
        "mean_opinion": final_mean_opinion,
        "variance": opinion_variance,
        "clustering_coefficient": clustering_coefficient
    }

if __name__ == "__main__":
    # Using a consensus-likely scenario as the "Target" for now
    calculate_summary_statistics(confidence_bound=0.5, mu=0.5)
