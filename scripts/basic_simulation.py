import matplotlib.pyplot as plt
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
    
    history = run_simulation(G, max_steps, confidence_bound, mu)
    
    print("Simulation Complete. Plotting results...")
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_steps), history, label='Average Opinion')
    plt.xlabel('Simulation Step')
    plt.ylabel('Average Opinion')
    plt.title('Opinion Convergence Over Time')
    plt.legend()
    plt.grid(True)
    
    output_file = os.path.join(os.path.dirname(__file__), '..', 'output', 'convergence_plot.png')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    main()
