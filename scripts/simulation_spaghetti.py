import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import initialize_network, run_simulation

def main():
    print("Initializing Network...")
    n_agents = 100
    G = initialize_network(n_agents)
    
    print("Running Simulation...")
    max_steps = 5000
    confidence_bound = 0.5
    mu = 0.5
    
    # history is now a list of lists (steps x agents)
    history = run_simulation(G, max_steps, confidence_bound, mu)
    
    # Convert to numpy array for easier slicing: shape (steps, agents)
    history_array = np.array(history)
    
    print("Simulation Complete. Plotting spaghetti...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot each agent's trajectory
    # We use a low alpha to handle overlapping lines
    plt.plot(history_array, alpha=0.3, linewidth=0.5)
    
    plt.xlabel('Simulation Step')
    plt.ylabel('Opinion (0.0 - 1.0)')
    plt.title('Opinion Evolution (Spaghetti Plot)')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.join(os.path.dirname(__file__), '..', 'output', 'spaghetti_plot.png')
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
