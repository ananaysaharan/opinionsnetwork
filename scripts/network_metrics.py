import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from simulation_model import SimulationModel

def run_scenario(confidence_bound, scenario_name, steps=1000):
    print(f"Running scenario: {scenario_name} (c={confidence_bound})...")
    
    # Initialize model
    # Using default n_agents=100, mu=0.5
    sim = SimulationModel(n_agents=100, confidence_bound=confidence_bound, mu=0.5)
    
    # Run simulation
    sim.run(steps=steps)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # history is a list of lists (steps x agents)
    # We want to plot each agent's opinion over time
    history = np.array(sim.history)
    
    # Plot each agent's trajectory
    for i in range(sim.n_agents):
        plt.plot(range(steps), history[:, i], alpha=0.3, linewidth=1)
        
    plt.title(f"Opinion Evolution: {scenario_name} (c={confidence_bound})")
    plt.xlabel("Time Step")
    plt.ylabel("Opinion")
    plt.ylim(0, 1)
    
    # Save plot
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, f"day7_{scenario_name}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

def main():
    # Scenario 1: Consensus (High confidence bound)
    # Everyone should eventually agree (or close to it)
    run_scenario(confidence_bound=0.5, scenario_name="consensus")
    
    # Scenario 2: Polarization (Medium confidence bound)
    # Should split into 2 distinct groups usually
    run_scenario(confidence_bound=0.1, scenario_name="polarization")
    
    # Scenario 3: Fragmentation (Low confidence bound)
    # Should split into many small groups
    run_scenario(confidence_bound=0.05, scenario_name="fragmentation")
    
    print("All scenarios completed.")

if __name__ == "__main__":
    main()
