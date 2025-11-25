import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from simulation_model import SimulationModel
import networkx as nx

def test_simulation_model():
    print("Testing SimulationModel...")
    
    # Initialize
    sim = SimulationModel(n_agents=50, confidence_bound=0.3, mu=0.5)
    print("Initialization successful.")
    
    # Check initial graph
    assert len(sim.G.nodes) == 50
    print(f"Graph has {len(sim.G.nodes)} nodes.")
    
    # Run simulation
    print("Running simulation for 500 steps...")
    sim.run(steps=500)
    
    # Check history
    assert len(sim.history) == 500
    print(f"History has {len(sim.history)} entries.")
    
    # Check metrics
    metrics = sim.get_metrics()
    print("Metrics:", metrics)
    assert "number_connected_components" in metrics
    assert "average_opinion" in metrics
    assert "opinion_variance" in metrics
    assert "clustering_coefficient" in metrics
    
    print("Test Passed!")

if __name__ == "__main__":
    test_simulation_model()
