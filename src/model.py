import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def initialize_network(n_agents):
    """
    Creates a Watts-Strogatz small-world graph and assigns random opinions.
    
    Args:
        n_agents (int): Number of agents (nodes) in the network.
        
    Returns:
        nx.Graph: The initialized network with 'opinion' attributes.
    """
    # Create Watts-Strogatz graph
    # n=n_agents, k=4 (each node joins with k nearest neighbors), p=0.1 (probability of rewiring)
    G = nx.watts_strogatz_graph(n=n_agents, k=4, p=0.1)
    
    # Assign random opinions
    for i in G.nodes():
        G.nodes[i]['opinion'] = np.random.random()
        
    return G

def interact(agent_i_opinion, agent_j_opinion, confidence_bound, mu):
    """
    Executes the Deffuant-Weisbuch interaction rule.
    
    Args:
        agent_i_opinion (float): Opinion of agent i.
        agent_j_opinion (float): Opinion of agent j.
        confidence_bound (float): The confidence threshold.
        mu (float): The convergence parameter (0 to 0.5).
        
    Returns:
        tuple: (new_opinion_i, new_opinion_j)
    """
    diff = abs(agent_i_opinion - agent_j_opinion)
    
    if diff < confidence_bound:
        new_i = agent_i_opinion + mu * (agent_j_opinion - agent_i_opinion)
        new_j = agent_j_opinion + mu * (agent_i_opinion - agent_j_opinion)
        return new_i, new_j
    else:
        return agent_i_opinion, agent_j_opinion

def run_simulation(G, max_steps, confidence_bound, mu):
    """
    Runs the social network simulation over time.
    
    Args:
        G (nx.Graph): The initialized network.
        max_steps (int): Number of simulation steps.
        confidence_bound (float): Confidence threshold for interaction.
        mu (float): Convergence parameter.
        
    Returns:
        list: History of average opinions at each step.
    """
    import random
    
    history = []
    
    # Pre-calculate edges list for efficiency if graph is static
    edges = list(G.edges())
    
    for step in range(max_steps):
        # Pick a random edge
        u, v = random.choice(edges)
        
        # Get current opinions
        op_u = G.nodes[u]['opinion']
        op_v = G.nodes[v]['opinion']
        
        # Interact
        new_u, new_v = interact(op_u, op_v, confidence_bound, mu)
        
        # Update opinions
        G.nodes[u]['opinion'] = new_u
        G.nodes[v]['opinion'] = new_v
        
        # Store all opinions for spaghetti plot
        # We store a copy of the list of opinions
        all_opinions = [G.nodes[n]['opinion'] for n in G.nodes()]
        history.append(all_opinions)
        
    return history

def run_adaptive_simulation(G, max_steps, confidence_bound, mu):
    """
    Runs the adaptive social network simulation (homophily).
    
    Args:
        G (nx.Graph): The initialized network.
        max_steps (int): Number of simulation steps.
        confidence_bound (float): Confidence threshold for interaction.
        mu (float): Convergence parameter.
        
    Returns:
        list: History of average opinions at each step.
    """
    import random
    
    history = []
    
    for step in range(max_steps):
        # Edges change dynamically, so we must list them each time
        # This is slower but necessary for adaptive networks
        edges = list(G.edges())
        
        if not edges:
            break
            
        # Pick a random edge
        u, v = random.choice(edges)
        
        # Get current opinions
        op_u = G.nodes[u]['opinion']
        op_v = G.nodes[v]['opinion']
        
        diff = abs(op_u - op_v)
        
        if diff > confidence_bound:
            # The Breakup: Remove edge
            G.remove_edge(u, v)
            
            # Find new friend for u
            # Pick a random agent k from the population (excluding u)
            possible_friends = list(G.nodes())
            possible_friends.remove(u)
            if possible_friends:
                k = random.choice(possible_friends)
                op_k = G.nodes[k]['opinion']
                
                # If they agree, add edge
                if abs(op_u - op_k) < confidence_bound:
                    G.add_edge(u, k)
        else:
            # Interact and update opinions
            new_u, new_v = interact(op_u, op_v, confidence_bound, mu)
            G.nodes[u]['opinion'] = new_u
            G.nodes[v]['opinion'] = new_v
        
        # Store all opinions
        all_opinions = [G.nodes[n]['opinion'] for n in G.nodes()]
        history.append(all_opinions)
        
    return history


if __name__ == "__main__":
    # Test the initialization
    n_agents = 100
    G = initialize_network(n_agents)
    
    # Print a sample agent's opinion
    # We'll pick agent 5 as requested, or just a random one if 5 doesn't exist (it will for n=100)
    agent_id = 5
    if agent_id in G.nodes:
        opinion = G.nodes[agent_id]['opinion']
        print(f"Agent {agent_id} has opinion {opinion:.2f}")
    else:
        print(f"Agent {agent_id} not found in graph.")
