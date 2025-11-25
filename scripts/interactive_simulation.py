import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import initialize_network, run_adaptive_simulation

def main():
    print("Initializing Network...")
    n_agents = 100
    G = initialize_network(n_agents)
    
    print("Running Adaptive Simulation (Homophily)...")
    max_steps = 10000  # Increased steps to allow for structural changes
    confidence_bound = 0.3 # Tighter bound to encourage breaking up
    mu = 0.5
    
    history = run_adaptive_simulation(G, max_steps, confidence_bound, mu)
    
    # Check for Echo Chambers
    num_components = nx.number_connected_components(G)
    print(f"Number of connected components: {num_components}")
    
    if num_components > 1:
        print("SUCCESS: Echo Chambers created!")
    else:
        print("Result: Network is still connected (or only isolated nodes).")

    # Optional: Plot the final graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    
    # Color nodes by opinion
    opinions = [G.nodes[n]['opinion'] for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=opinions, cmap=plt.cm.coolwarm, node_size=50)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    plt.title(f"Adaptive Network Final State (Components: {num_components})")
    plt.axis('off')
    
    output_file = os.path.join(os.path.dirname(__file__), '..', 'output', 'adaptive_network_plot.png')
    plt.savefig(output_file)
    print(f"Graph plot saved to {output_file}")

if __name__ == "__main__":
    main()
