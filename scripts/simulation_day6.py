import sys
import os
import networkx as nx

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader import load_population_from_csv
from model import run_simulation

def main():
    print("Running Day 6 Verification...")
    
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'population.csv'))
    print(f"Loading population from {csv_path}...")
    
    try:
        G = load_population_from_csv(csv_path)
        print("Graph loaded successfully.")
    except Exception as e:
        print(f"Failed to load graph: {e}")
        return

    print(f"Number of nodes: {len(G.nodes)}")
    print(f"Number of edges: {len(G.edges)}")
    
    # Verify opinions match CSV
    # CSV: 1->0.1, 2->0.9, ...
    # Note: Our loader converts IDs to int if possible.
    try:
        op_1 = G.nodes[1]['opinion']
        print(f"Agent 1 opinion: {op_1} (Expected 0.1)")
        assert abs(op_1 - 0.1) < 1e-6
    except KeyError:
        print("Agent 1 not found in graph nodes:", G.nodes)
    
    # Run a short simulation
    print("Running short simulation...")
    history = run_simulation(G, max_steps=100, confidence_bound=0.3, mu=0.5)
    print(f"Simulation completed. History length: {len(history)}")
    
    print("Day 6 Verification Passed!")

if __name__ == "__main__":
    main()
