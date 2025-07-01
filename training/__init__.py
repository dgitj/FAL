"""
Training module for federated active learning.

This module provides classes and functions for training federated learning
models with various active learning sampling strategies.
"""

from .trainer import FederatedTrainer
from .evaluation import (
    evaluate_model,
    evaluate_per_class_accuracy,
    evaluate_gradient_alignment,
    evaluate_knowledge_gap,
    create_confusion_matrix
)
from .utils import (
    set_all_seeds,
    get_seed_worker,
    read_data,
    create_data_loaders,
    get_device,
    calculate_model_distance,
    create_experiment_name,
    log_config,
    create_results_dir,
    calculate_class_distribution,
    Timer,
    requires_grad,
    count_parameters
)

__all__ = [
    # Trainer
    'FederatedTrainer',
    
    # Evaluation
    'evaluate_model',
    'evaluate_per_class_accuracy',
    'evaluate_gradient_alignment',
    'evaluate_knowledge_gap',
    'create_confusion_matrix',
    
    # Utilities
    'set_all_seeds',
    'get_seed_worker',
    'read_data',
    'create_data_loaders',
    'get_device',
    'calculate_model_distance',
    'create_experiment_name',
    'log_config',
    'create_results_dir',
    'calculate_class_distribution',
    'Timer',
    'requires_grad',
    'count_parameters'
]