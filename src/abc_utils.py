import numpy as np

def calculate_distance(sim_stats, obs_stats):
    """
    Calculates the Euclidean distance between simulated and observed statistics.
    
    Args:
        sim_stats (dict or list/array): Simulated statistics.
        obs_stats (dict or list/array): Observed statistics.
        
    Returns:
        float: The Euclidean distance (error score).
    """
    # Convert dictionaries to lists if necessary, ensuring same order of keys
    if isinstance(sim_stats, dict) and isinstance(obs_stats, dict):
        # Sort keys to ensure consistent order
        keys = sorted(sim_stats.keys())
        sim_vec = np.array([sim_stats[k] for k in keys])
        obs_vec = np.array([obs_stats[k] for k in keys])
    else:
        sim_vec = np.array(sim_stats)
        obs_vec = np.array(obs_stats)
        
    # Calculate Euclidean distance
    distance = np.sqrt(np.sum((sim_vec - obs_vec)**2))
    
    return distance
