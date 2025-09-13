"""
Simple specialist detection for AHFAL clustering extension.
Detects clients that specialize in specific classes based on concentration ratios.
"""

import numpy as np

def detect_specialists(client_distributions, global_distribution, num_classes, min_cluster_size=2):
    """
    Detect which clients are specialists for which classes.
    
    Args:
        client_distributions (list): List of dicts {class_id: percentage} for each client
        global_distribution (dict): Global class distribution {class_id: percentage}
        num_classes (int): Total number of classes
        min_cluster_size (int): Minimum clients needed to form a specialist cluster
        
    Returns:
        dict: {client_id: specialist_class_id} or {client_id: 'generalist'}
    """
    print("[Specialist Detection] Starting specialist detection...")
    
    # Calculate concentration scores for all client-class pairs
    concentration_scores = {}
    entropy_scores = {}
    
    for client_id, client_dist in enumerate(client_distributions):
        # Calculate concentration ratios
        client_concentrations = calculate_concentration_ratios(client_dist, global_distribution)
        concentration_scores[client_id] = client_concentrations
        
        # Calculate entropy (lower = more specialized)
        entropy = calculate_entropy(client_dist, num_classes)
        entropy_scores[client_id] = entropy
    
    # Find potential specialists using adaptive thresholds
    potential_specialists = {}
    
    for client_id in range(len(client_distributions)):
        client_concentrations = concentration_scores[client_id]
        client_entropy = entropy_scores[client_id]
        
        # Find class with highest concentration
        max_class = max(client_concentrations.keys(), key=lambda k: client_concentrations[k])
        max_concentration = client_concentrations[max_class]
        
        # Adaptive thresholds based on data distribution
        concentration_threshold = 1.5  # Lowered from 2.0 to detect more specialists
        entropy_threshold = calculate_entropy_threshold(entropy_scores)
        
        # Client is specialist if: high concentration AND low entropy
        if max_concentration >= concentration_threshold and client_entropy <= entropy_threshold:
            if max_class not in potential_specialists:
                potential_specialists[max_class] = []
            potential_specialists[max_class].append(client_id)
    
    # Filter out classes that don't have enough specialists
    valid_specialists = {}
    specialist_mapping = {}
    
    for class_id, client_list in potential_specialists.items():
        if len(client_list) >= min_cluster_size:
            valid_specialists[class_id] = client_list
            for client_id in client_list:
                specialist_mapping[client_id] = class_id
                
    # Assign remaining clients as generalists
    for client_id in range(len(client_distributions)):
        if client_id not in specialist_mapping:
            specialist_mapping[client_id] = 'generalist'
    
    # Print detection results
    print(f"[Specialist Detection] Found specialists for {len(valid_specialists)} classes")
    for class_id, clients in valid_specialists.items():
        print(f"  Class {class_id}: {len(clients)} specialists (clients {clients})")
    
    generalists = [c for c, s in specialist_mapping.items() if s == 'generalist']
    print(f"  Generalists: {len(generalists)} clients {generalists}")
    
    # Print detailed mapping for clarity
    print("\n[Specialist Detection] Final specialist assignments:")
    for client_id, assignment in specialist_mapping.items():
        if assignment == 'generalist':
            print(f"  Client {client_id}: Generalist")
        else:
            print(f"  Client {client_id}: Specialist for Class {assignment}")
    
    return specialist_mapping

def calculate_concentration_ratios(client_dist, global_dist):
    """
    Calculate how concentrated each class is for this client vs. global average.
    
    Args:
        client_dist (dict): {class_id: percentage} for this client
        global_dist (dict): {class_id: percentage} globally
        
    Returns:
        dict: {class_id: concentration_ratio} where ratio > 1 means over-represented
    """
    concentration_ratios = {}
    
    for class_id in client_dist.keys():
        client_percentage = client_dist.get(class_id, 0.0)
        global_percentage = global_dist.get(class_id, 0.01)  # Avoid division by zero
        
        # Concentration ratio: how much more this client has vs. global average
        ratio = client_percentage / global_percentage if global_percentage > 0 else 0.0
        concentration_ratios[class_id] = ratio
        
    return concentration_ratios

def calculate_entropy(client_dist, num_classes):
    """
    Calculate entropy of client's class distribution (lower = more specialized).
    
    Args:
        client_dist (dict): {class_id: percentage} for this client
        num_classes (int): Total number of classes
        
    Returns:
        float: Entropy score (0 = perfectly specialized, higher = more uniform)
    """
    # Convert to probability distribution
    total_samples = sum(client_dist.values()) if client_dist.values() else 1
    probabilities = []
    
    for class_id in range(num_classes):
        prob = client_dist.get(class_id, 0) / total_samples if total_samples > 0 else 0
        probabilities.append(prob)
    
    # Calculate entropy: -sum(p * log(p))
    entropy = 0.0
    for prob in probabilities:
        if prob > 0:
            entropy -= prob * np.log2(prob)
            
    return entropy

def calculate_entropy_threshold(entropy_scores):
    """
    Calculate adaptive entropy threshold based on distribution of entropy scores.
    
    Args:
        entropy_scores (dict): {client_id: entropy_score}
        
    Returns:
        float: Entropy threshold (clients below this are considered specialized)
    """
    if not entropy_scores:
        return 2.0  # Default threshold
    
    entropy_values = list(entropy_scores.values())
    
    # Use median as threshold (bottom 50% are considered more specialized)
    threshold = np.median(entropy_values)
    
    print(f"[Specialist Detection] Entropy threshold: {threshold:.3f}")
    print(f"  Min entropy: {min(entropy_values):.3f}, Max entropy: {max(entropy_values):.3f}")
    
    return threshold

def is_specialist(client_dist, global_dist, client_id, class_id):
    """
    Check if a specific client is a specialist for a specific class.
    
    Args:
        client_dist (dict): Client's class distribution
        global_dist (dict): Global class distribution
        client_id (int): Client ID
        class_id (int): Class ID to check
        
    Returns:
        bool: True if client is specialist for this class
    """
    concentrations = calculate_concentration_ratios(client_dist, global_dist)
    concentration = concentrations.get(class_id, 0.0)
    
    # Simple threshold check
    return concentration >= 2.0
