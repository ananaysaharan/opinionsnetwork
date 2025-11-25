import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

print("Anaconda setup is successful.")
print(f"NetworkX version: {nx.__version__}")
print(f"Numpy version: {np.__version__}")

# Quick test of the math engine
G = nx.watts_strogatz_graph(10, 4, 0.1)
print(f"Graph created with {len(G.nodes)} nodes.")