"""
Simple cluster management for specialist clients.
Groups specialists and handles edge cases.
"""


def form_clusters(specialist_mapping, min_cluster_size=2):
    """
    Form clusters from specialist mapping.
    
    Args:
        specialist_mapping (dict): {client_id: specialist_class_id or 'generalist'}
        min_cluster_size (int): Minimum clients per specialist cluster
        
    Returns:
        dict: {cluster_id: [client_ids]}
              cluster_id is either class_id (int) or 'generalist' (str)
    """
    if not specialist_mapping:
        print("[ClusterManager] No specialist mapping provided")
        return {}
    
    # Group clients by their assignment
    raw_clusters = {}
    for client_id, assignment in specialist_mapping.items():
        if assignment not in raw_clusters:
            raw_clusters[assignment] = []
        raw_clusters[assignment].append(client_id)
    
    print(f"[Cluster Manager] Raw clusters before size filtering:")
    for cluster_id, client_list in raw_clusters.items():
        print(f"  '{cluster_id}': {len(client_list)} clients {client_list}")
    
    # Validate cluster sizes and merge small specialist clusters into generalist
    final_clusters = {}
    generalist_clients = raw_clusters.get('generalist', [])
    
    for cluster_id, client_list in raw_clusters.items():
        if cluster_id == 'generalist':
            continue  # Handle generalists at the end
            
        # If specialist cluster is too small, merge into generalist
        if len(client_list) < min_cluster_size:
            print(f"[ClusterManager] Merging small specialist cluster {cluster_id} into generalist")
            generalist_clients.extend(client_list)
        else:
            final_clusters[cluster_id] = client_list
    
    # Add generalist cluster if we have any generalists
    if generalist_clients:
        final_clusters['generalist'] = generalist_clients
    
    # Log final clusters
    _log_cluster_summary(final_clusters)
    
    return final_clusters


def validate_clusters(clusters, min_size=2):
    """
    Validate that clusters meet minimum size requirements.
    
    Args:
        clusters (dict): {cluster_id: [client_ids]}
        min_size (int): Minimum cluster size
        
    Returns:
        bool: True if all clusters are valid
    """
    if not clusters:
        return False
    
    for cluster_id, client_list in clusters.items():
        # Generalist cluster can be any size (including 0)
        if cluster_id == 'generalist':
            continue
            
        # Specialist clusters must meet minimum size
        if len(client_list) < min_size:
            print(f"[ClusterManager] Invalid cluster {cluster_id}: size {len(client_list)} < {min_size}")
            return False
    
    return True


def get_cluster_info(clusters):
    """
    Get summary information about clusters.
    
    Args:
        clusters (dict): {cluster_id: [client_ids]}
        
    Returns:
        dict: Summary information about clusters
    """
    info = {
        'num_clusters': len(clusters),
        'num_specialist_clusters': len([c for c in clusters.keys() if c != 'generalist']),
        'total_clients': sum(len(clients) for clients in clusters.values()),
        'cluster_sizes': {cluster_id: len(clients) for cluster_id, clients in clusters.items()}
    }
    
    return info


def _log_cluster_summary(clusters):
    """Helper function to log cluster formation results."""
    if not clusters:
        print("[ClusterManager] No clusters formed")
        return
    
    specialist_clusters = {k: v for k, v in clusters.items() if k != 'generalist'}
    generalist_clients = clusters.get('generalist', [])
    
    print(f"[ClusterManager] Formed {len(specialist_clusters)} specialist clusters:")
    for cluster_id, clients in specialist_clusters.items():
        print(f"  Class {cluster_id}: {len(clients)} clients {clients}")
    
    if generalist_clients:
        print(f"[ClusterManager] Generalist cluster: {len(generalist_clients)} clients {generalist_clients}")
    
    total_clients = sum(len(clients) for clients in clusters.values())
    print(f"[ClusterManager] Total clients: {total_clients}")
