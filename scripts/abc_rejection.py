import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from abc_utils import calculate_distance

def main():
    print("--- Day 9: The Judge (Distance Function) ---")
    
    # Define dummy S_obs (Observed Statistics - The Target)
    # Let's assume these are the values we got from Day 8's consensus scenario
    S_obs = {
        "mean_opinion": 0.5284,
        "variance": 0.0118,
        "clustering_coefficient": 0.2537
    }
    print(f"\nObserved Stats (Target): {S_obs}")
    
    # Define dummy S_sim (Simulated Statistics - A guess)
    # Let's assume this is from a simulation that didn't quite match (e.g., lower clustering)
    S_sim = {
        "mean_opinion": 0.55,
        "variance": 0.02,
        "clustering_coefficient": 0.15
    }
    print(f"Simulated Stats (Guess): {S_sim}")
    
    # Calculate Distance
    distance = calculate_distance(S_sim, S_obs)
    
    print(f"\nCalculated Euclidean Distance: {distance:.4f}")
    
    # Manual verification for the user to see
    # d = sqrt((0.55-0.5284)^2 + (0.02-0.0118)^2 + (0.15-0.2537)^2)
    diff_mean = S_sim["mean_opinion"] - S_obs["mean_opinion"]
    diff_var = S_sim["variance"] - S_obs["variance"]
    diff_clust = S_sim["clustering_coefficient"] - S_obs["clustering_coefficient"]
    
    manual_sq_sum = diff_mean**2 + diff_var**2 + diff_clust**2
    manual_dist = np.sqrt(manual_sq_sum)
    
    print(f"Manual Verification: {manual_dist:.4f}")
    
    if abs(distance - manual_dist) < 1e-9:
        print("\nSUCCESS: Function matches manual calculation.")
    else:
        print("\nFAILURE: Function does not match manual calculation.")

if __name__ == "__main__":
    main()
