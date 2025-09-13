#!/usr/bin/env python3
"""
Simple test script for Phase 1 specialist clustering implementation.
Tests the basic specialist detection and clustering functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.specialist_detection import detect_specialists, calculate_concentration_ratios
from training.cluster_manager import form_clusters, validate_clusters
import config

def test_specialist_detection():
    """Test specialist detection with synthetic data."""
    print("=== Testing Specialist Detection ===")
    
    # Create synthetic client distributions (5 clients, 10 classes like CIFAR-10)
    # Client 0: Specialist in class 0 (planes)
    client_0 = {0: 0.6, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.05, 7: 0.05, 8: 0.05, 9: 0.05}
    # Client 1: Specialist in class 1 (cars)  
    client_1 = {0: 0.05, 1: 0.55, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.05, 7: 0.05, 8: 0.05, 9: 0.05}
    # Client 2: Another plane specialist
    client_2 = {0: 0.5, 1: 0.1, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.05, 6: 0.05, 7: 0.05, 8: 0.05, 9: 0.05}
    # Client 3: Generalist (uniform-ish distribution)
    client_3 = {0: 0.12, 1: 0.11, 2: 0.1, 3: 0.09, 4: 0.11, 5: 0.1, 6: 0.12, 7: 0.08, 8: 0.09, 9: 0.08}
    # Client 4: Another generalist
    client_4 = {0: 0.1, 1: 0.1, 2: 0.12, 3: 0.11, 4: 0.1, 5: 0.09, 6: 0.1, 7: 0.12, 8: 0.08, 9: 0.08}
    
    client_distributions = [client_0, client_1, client_2, client_3, client_4]
    
    # Global distribution (average)
    global_distribution = {}
    for class_id in range(10):
        global_distribution[class_id] = sum(client[class_id] for client in client_distributions) / len(client_distributions)
    
    print("Global distribution:", {k: f"{v:.3f}" for k, v in global_distribution.items()})
    
    # Test specialist detection
    specialist_mapping = detect_specialists(
        client_distributions, 
        global_distribution, 
        num_classes=10,
        min_cluster_size=config.MIN_CLUSTER_SIZE
    )
    
    print("\\nSpecialist mapping:", specialist_mapping)
    
    # Test cluster formation
    clusters = form_clusters(specialist_mapping, min_cluster_size=config.MIN_CLUSTER_SIZE)
    print("\\nClusters formed:", clusters)
    
    # Test validation
    valid = validate_clusters(clusters, min_size=config.MIN_CLUSTER_SIZE)
    print(f"\\nClusters valid: {valid}")
    
    return clusters, valid

def test_edge_cases():
    """Test edge cases like no specialists, all specialists, etc."""
    print("\\n=== Testing Edge Cases ===")
    
    # Case 1: All generalists (uniform distributions)
    print("\\nCase 1: All generalists")
    uniform_clients = []
    for i in range(5):
        uniform = {class_id: 0.1 for class_id in range(10)}
        uniform_clients.append(uniform)
    
    global_uniform = {class_id: 0.1 for class_id in range(10)}
    
    specialists = detect_specialists(uniform_clients, global_uniform, 10, min_cluster_size=2)
    clusters = form_clusters(specialists, min_cluster_size=2)
    print("All generalists - Specialists:", specialists)
    print("All generalists - Clusters:", clusters)
    
    # Case 2: Single client specialists (should merge to generalist)
    print("\\nCase 2: Single client specialists")
    single_specialists = [
        {0: 0.7, 1: 0.03, 2: 0.03, 3: 0.03, 4: 0.03, 5: 0.03, 6: 0.03, 7: 0.03, 8: 0.03, 9: 0.03},  # Class 0 specialist
        {0: 0.03, 1: 0.7, 2: 0.03, 3: 0.03, 4: 0.03, 5: 0.03, 6: 0.03, 7: 0.03, 8: 0.03, 9: 0.03},  # Class 1 specialist
        {0: 0.03, 1: 0.03, 2: 0.7, 3: 0.03, 4: 0.03, 5: 0.03, 6: 0.03, 7: 0.03, 8: 0.03, 9: 0.03},  # Class 2 specialist
        {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1},  # Generalist
        {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1, 7: 0.1, 8: 0.1, 9: 0.1},  # Generalist
    ]
    
    global_single = {class_id: sum(client[class_id] for client in single_specialists) / len(single_specialists) for class_id in range(10)}
    
    specialists_single = detect_specialists(single_specialists, global_single, 10, min_cluster_size=2)
    clusters_single = form_clusters(specialists_single, min_cluster_size=2)
    print("Single specialists - Specialists:", specialists_single)
    print("Single specialists - Clusters:", clusters_single)

if __name__ == "__main__":
    print("Phase 1 Specialist Clustering Test")
    print("=" * 40)
    
    # Test normal functionality
    clusters, valid = test_specialist_detection()
    
    # Test edge cases
    test_edge_cases()
    
    # Summary
    print("\\n=== Test Summary ===")
    if valid and len(clusters) > 0:
        print("✅ Phase 1 implementation appears to be working correctly!")
        print(f"✅ Successfully formed {len(clusters)} clusters")
        print("✅ All basic functionality tested")
    else:
        print("❌ Phase 1 implementation has issues")
        print("❌ Check the specialist detection or clustering logic")
    
    print("\\nReady to run with main.py!")
