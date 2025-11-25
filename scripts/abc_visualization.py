import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_posterior():
    print("--- Day 12: Visualizing The Truth ---")
    
    # Load accepted parameters
    input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'abc_accepted_params.csv'))
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please run scripts/simulation_day11.py first.")
        return
        
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} accepted parameter sets.")
    
    # Setup output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Plot 1: Posterior of Confidence Bound (c)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # KDE Plot
    sns.kdeplot(data=df, x='c', fill=True, color='blue', alpha=0.3)
    
    # Add mean line
    mean_c = df['c'].mean()
    plt.axvline(mean_c, color='red', linestyle='--', label=f'Mean c = {mean_c:.4f}')
    
    plt.title("Posterior Distribution of Confidence Bound (c)")
    plt.xlabel("Confidence Bound (c)")
    plt.ylabel("Density")
    plt.legend()
    
    output_path = os.path.join(output_dir, 'abc_posterior_c.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved posterior plot for 'c' to {output_path}")
    
    # Plot 2: Joint Plot (c vs mu) - Optional but very cool
    plt.figure(figsize=(10, 10))
    g = sns.jointplot(data=df, x='c', y='mu', kind="kde", fill=True, cmap="Blues")
    g.fig.suptitle("Joint Posterior Distribution (c vs mu)", y=1.02)
    
    output_path = os.path.join(output_dir, 'abc_posterior_joint.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved joint posterior plot to {output_path}")
    
    print("\nVisualization Complete.")

if __name__ == "__main__":
    plot_posterior()
