import csv
import networkx as nx

def load_population_from_csv(file_path):
    """
    Loads population data from a CSV file and initializes a graph.
    
    Args:
        file_path (str): Path to the CSV file containing 'user_id' and 'opinion'.
        
    Returns:
        nx.Graph: A graph with nodes corresponding to user_ids and 'opinion' attributes.
                  The graph structure is initialized as a Watts-Strogatz graph
                  mapped to these nodes.
    """
    ids = []
    opinions = {}
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # user_id might be string or int, let's keep it as is (string from CSV)
            # but usually graph nodes are ints or strings. Let's try to convert to int if possible,
            # but for robustness, maybe keep as string if not purely numeric.
            # The mock data has numeric IDs.
            try:
                uid = int(row['user_id'])
            except ValueError:
                uid = row['user_id']
                
            op = float(row['opinion'])
            ids.append(uid)
            opinions[uid] = op
            
    n_agents = len(ids)
    
    # Create a Watts-Strogatz graph for structure
    # This creates nodes 0 to n_agents-1
    G_temp = nx.watts_strogatz_graph(n=n_agents, k=4, p=0.1)
    
    # Create the final graph
    G = nx.Graph()
    
    # Mapping from temporary index to user_id
    mapping = {i: ids[i] for i in range(n_agents)}
    
    # Add nodes with opinions
    for i in range(n_agents):
        uid = ids[i]
        G.add_node(uid, opinion=opinions[uid])
        
    # Add edges from the temporary graph, mapping indices to user_ids
    for u, v in G_temp.edges():
        G.add_edge(mapping[u], mapping[v])
        
    return G
