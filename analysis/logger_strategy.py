import os
import numpy as np
import json
import config

class FederatedALLogger:
    def __init__(self, strategy_name, num_clients, num_classes, trial_id, log_dir="analysis_logs"):
        self.strategy_name = strategy_name
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.trial_id = trial_id
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize logging structures
        self.global_accuracy = {}
        self.class_accuracies = {}
        self.round_history = []
        self.communication_costs = {}  # NEW: Track communication costs
        self.cycle_times = {}  # NEW: Track wall-clock time per cycle
        
        # Print config variables
        print("\n===== Experiment Configuration =====")
        for key, value in vars(config).items():
            if not key.startswith('__') and key.isupper():
                print(f"{key}: {value}")
        print("===================================\n")
        
    def log_global_accuracy(self, cycle, accuracy):
        self.global_accuracy[cycle] = float(accuracy)
    
    def log_class_accuracies(self, cycle, class_accuracies):
        self.class_accuracies[cycle] = {str(k): float(v) for k, v in class_accuracies.items()}
    
    def log_cycle_time(self, cycle, time_seconds):
        """
        Log wall-clock time for a cycle.
        
        Args:
            cycle: AL cycle number
            time_seconds: Time in seconds
        """
        self.cycle_times[cycle] = float(time_seconds)
    
    def log_communication_costs(self, cycle, model_params, model_bytes, num_clients, 
                                 client_class_distributions, global_distribution=None):
        """
        Log communication costs for a cycle.
        
        Args:
            cycle: AL cycle number
            model_params: Number of model parameters
            model_bytes: Model size in bytes
            num_clients: Number of clients
            client_class_distributions: Dict {client_id: class_distribution_vector}
            global_distribution: Optional global class distribution dict
        """
        # Class distribution size per client (C classes * 4 bytes per float32)
        class_dist_bytes_per_client = self.num_classes * 4
        total_extra_bytes = class_dist_bytes_per_client * num_clients
        total_model_bytes = model_bytes * num_clients
        
        # Calculate overhead
        overhead_pct = (total_extra_bytes / total_model_bytes * 100) if total_model_bytes > 0 else 0
        
        self.communication_costs[cycle] = {
            'model_params': int(model_params),
            'model_bytes': int(model_bytes),
            'num_clients': int(num_clients),
            'extra_bytes_per_client': int(class_dist_bytes_per_client),
            'total_model_bytes': int(total_model_bytes),
            'total_extra_bytes': int(total_extra_bytes),
            'overhead_percentage': float(overhead_pct),
            'client_distributions': {str(cid): vec.tolist() if hasattr(vec, 'tolist') else vec 
                                     for cid, vec in client_class_distributions.items()}
        }
        
        if global_distribution is not None:
            self.communication_costs[cycle]['global_distribution'] = {
                str(k): float(v) for k, v in global_distribution.items()
            }
    
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
            'global_accuracy': convert_for_json(self.global_accuracy),
            'class_accuracies': convert_for_json(self.class_accuracies),
            'communication_costs': convert_for_json(self.communication_costs),
            'cycle_times': convert_for_json(self.cycle_times)
        }
        
        # Save JSON file
        json_filename = f"{self.strategy_name}_c{self.num_clients}_trial{self.trial_id}.json"
        json_path = os.path.join(self.log_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved experiment data to: {json_path}")
