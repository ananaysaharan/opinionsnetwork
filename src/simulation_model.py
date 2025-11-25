import networkx as nx
import numpy as np
import random

class SimulationModel:
    def __init__(self, n_agents=100, confidence_bound=0.3, mu=0.5):
        """
        Initialize the SimulationModel.

        Args:
            n_agents (int): Number of agents in the network.
            confidence_bound (float): Confidence threshold for interaction.
            mu (float): Convergence parameter.
        """
        self.n_agents = n_agents
        self.confidence_bound = confidence_bound
        self.mu = mu
        self.G = self._initialize_network()
        self.history = []

    def _initialize_network(self):
        """
        Creates a Watts-Strogatz small-world graph and assigns random opinions.
        """
        # Create Watts-Strogatz graph
        G = nx.watts_strogatz_graph(n=self.n_agents, k=4, p=0.1)
        
        # Assign random opinions
        for i in G.nodes():
            G.nodes[i]['opinion'] = np.random.random()
            
        return G

    def _interact(self, agent_i_opinion, agent_j_opinion):
        """
        Executes the Deffuant-Weisbuch interaction rule.
        """
        diff = abs(agent_i_opinion - agent_j_opinion)
        
        if diff < self.confidence_bound:
            new_i = agent_i_opinion + self.mu * (agent_j_opinion - agent_i_opinion)
            new_j = agent_j_opinion + self.mu * (agent_i_opinion - agent_j_opinion)
            return new_i, new_j
        else:
            return agent_i_opinion, agent_j_opinion

    def step(self):
        """
        Perform a single step of the adaptive simulation.
        """
        edges = list(self.G.edges())
        
        if not edges:
            return

        # Pick a random edge
        u, v = random.choice(edges)
        
        # Get current opinions
        op_u = self.G.nodes[u]['opinion']
        op_v = self.G.nodes[v]['opinion']
        
        diff = abs(op_u - op_v)
        
        if diff > self.confidence_bound:
            # The Breakup: Remove edge
            self.G.remove_edge(u, v)
            
            # Find new friend for u
            possible_friends = list(self.G.nodes())
            possible_friends.remove(u)
            if possible_friends:
                k = random.choice(possible_friends)
                # Check if edge already exists to avoid overwriting or errors (though add_edge handles it)
                if not self.G.has_edge(u, k):
                    op_k = self.G.nodes[k]['opinion']
                    
                    # If they agree, add edge
                    if abs(op_u - op_k) < self.confidence_bound:
                        self.G.add_edge(u, k)
        else:
            # Interact and update opinions
            new_u, new_v = self._interact(op_u, op_v)
            self.G.nodes[u]['opinion'] = new_u
            self.G.nodes[v]['opinion'] = new_v
        
        # Record history (optional per step, or managed by run)
        # For efficiency, we might not want to append to history in every single step call 
        # if we are running millions of steps, but for now we'll leave it to the run method.

    def run(self, steps=1000):
        """
        Run the simulation for a specified number of steps.
        
        Args:
            steps (int): Number of steps to run.
        """
        for _ in range(steps):
            self.step()
            # Store all opinions
            all_opinions = [self.G.nodes[n]['opinion'] for n in self.G.nodes()]
            self.history.append(all_opinions)

    def get_metrics(self):
        """
        Return current metrics of the simulation.
        """
        return {
            "number_connected_components": nx.number_connected_components(self.G),
            "average_opinion": np.mean([self.G.nodes[n]['opinion'] for n in self.G.nodes()]),
            "opinion_variance": np.var([self.G.nodes[n]['opinion'] for n in self.G.nodes()]),
            "clustering_coefficient": nx.average_clustering(self.G)
        }
