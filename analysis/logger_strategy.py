import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import config

class FederatedALLogger:
    def __init__(self, strategy_name, num_clients, num_classes, log_dir="analysis_logs"):
        self.strategy_name = strategy_name
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize logging structures
        self.selected_samples = {}
        self.sample_classes = {}
        self.global_accuracy = {}
        self.class_accuracies = {}
        self.model_distances = {}
        self.round_history = []
        
        # Print config variables
        print("\n===== Experiment Configuration =====")
        for key, value in vars(config).items():
            if not key.startswith('__') and key.isupper():
                print(f"{key}: {value}")
        print("===================================\n")
        
    def log_selected_samples(self, cycle, client_samples, client_id):
        if cycle not in self.selected_samples:
            self.selected_samples[cycle] = {}
        self.selected_samples[cycle][client_id] = client_samples.copy()
        if cycle not in self.round_history:
            self.round_history.append(cycle)
    
    def log_sample_classes(self, cycle, client_classes, client_id):
        if cycle not in self.sample_classes:
            self.sample_classes[cycle] = {}
        self.sample_classes[cycle][client_id] = client_classes.copy()
    
    def log_global_accuracy(self, cycle, accuracy):
        self.global_accuracy[cycle] = float(accuracy)
    
    def log_class_accuracies(self, cycle, class_accuracies):
        self.class_accuracies[cycle] = {str(k): float(v) for k, v in class_accuracies.items()}
    
    def log_model_distances(self, cycle, model_distances):
        self.model_distances[cycle] = {str(k): float(v) for k, v in model_distances.items()}

    def calculate_model_distance(self, local_model, global_model):
        distance = 0.0
        local_params = dict(local_model.named_parameters())
        global_params = dict(global_model.named_parameters())
        
        for name, param in global_params.items():
            if name in local_params:
                distance += torch.norm(param - local_params[name]).item() ** 2
                
        return np.sqrt(distance)
    
    def save_data(self):
        # Convert numpy types to Python types
        def convert_for_json(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [convert_for_json(i) for i in obj]
            elif isinstance(obj, dict):
                return {str(k): convert_for_json(v) for k, v in obj.items()}
            else:
                return obj
        
        # Get config variables
        config_vars = {k: convert_for_json(v) for k, v in vars(config).items() 
                      if not k.startswith('__') and k.isupper()}
        
        # Prepare data for JSON
        data = {
            'strategy_name': self.strategy_name,
            'num_clients': self.num_clients,
            'num_classes': self.num_classes,
            'config': config_vars,
            'selected_samples': convert_for_json(self.selected_samples),
            'sample_classes': convert_for_json(self.sample_classes),
            'global_accuracy': convert_for_json(self.global_accuracy),
            'class_accuracies': convert_for_json(self.class_accuracies),
            'model_distances': convert_for_json(self.model_distances)
        }
        
        # Save JSON file
        json_filename = f"{self.strategy_name}_c{self.num_clients}_results.json"
        json_path = os.path.join(self.log_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved experiment data to: {json_path}")