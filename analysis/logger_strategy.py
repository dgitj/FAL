import os
import torch
import numpy as np
import torch.nn.functional as F
import json
import config
import copy

class FederatedALLogger:
    def __init__(self, strategy_name, num_clients, num_classes, trial_id, log_dir="analysis_logs"):
        self.strategy_name = strategy_name
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.trial_id = trial_id
        self.log_dir = log_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradient_alignments = {}
        self.knowledge_gaps = {}
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize logging structures
        self.selected_samples = {}
        self.sample_classes = {}
        self.global_accuracy = {}
        self.class_accuracies = {}
        self.model_distances = {}
        self.round_history = []
        self.gradient_alignments = {} 
        self.knowledge_gaps = {}  
        
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
    
    def log_gradient_alignment(self, cycle, model_local, model_global, dataloader, client_id):
        """Log alignment between local and global model gradients."""
        criterion = torch.nn.CrossEntropyLoss()
        
        # Create copies to avoid modifying original models
        local_copy = copy.deepcopy(model_local).to(self.device)
        global_copy = copy.deepcopy(model_global).to(self.device)
        
        alignments = []
        conflict_ratios = []
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Get local gradients
            local_copy.zero_grad()
            local_outputs, _ = local_copy(inputs)
            local_loss = criterion(local_outputs, targets)
            local_loss.backward()
            local_grads = []
            for param in local_copy.parameters():
                if param.grad is not None:
                    local_grads.append(param.grad.view(-1))
            local_grads = torch.cat(local_grads)
            
            # Get global gradients
            global_copy.zero_grad()
            global_outputs, _ = global_copy(inputs)
            global_loss = criterion(global_outputs, targets)
            global_loss.backward()
            global_grads = []
            for param in global_copy.parameters():
                if param.grad is not None:
                    global_grads.append(param.grad.view(-1))
            global_grads = torch.cat(global_grads)
            
            # Calculate alignment
            cos_sim = F.cosine_similarity(local_grads.unsqueeze(0), global_grads.unsqueeze(0))[0]
            alignments.append(cos_sim.item())
            
            # Calculate conflict ratio
            conflicts = ((local_grads * global_grads) < 0).float().mean().item()
            conflict_ratios.append(conflicts)
        
        # Store metrics
        if cycle not in self.gradient_alignments:
            self.gradient_alignments[cycle] = {}
        self.gradient_alignments[cycle][client_id] = {
            'alignment': np.mean(alignments) if alignments else 0,
            'conflict_ratio': np.mean(conflict_ratios) if conflict_ratios else 0
        }

    def log_knowledge_gap(self, cycle, model_local, model_global, dataloader, client_id):
        """Measure knowledge gap between local and global models."""
        local_predictions = {}
        global_predictions = {}
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Get local and global predictions
                local_outputs, _ = model_local(inputs)
                global_outputs, _ = model_global(inputs)
                
                local_probs = F.softmax(local_outputs, dim=1)
                global_probs = F.softmax(global_outputs, dim=1)
                
                # Calculate entropy (uncertainty)
                local_entropy = -torch.sum(local_probs * torch.log(local_probs + 1e-10), dim=1)
                global_entropy = -torch.sum(global_probs * torch.log(global_probs + 1e-10), dim=1)
                
                # Store results by class
                for i, target in enumerate(targets):
                    class_id = target.item()
                    
                    if class_id not in local_predictions:
                        local_predictions[class_id] = {
                            'correct': 0, 'total': 0, 'entropy': []
                        }
                        global_predictions[class_id] = {
                            'correct': 0, 'total': 0, 'entropy': []
                        }
                    
                    # Local model results
                    local_pred = torch.argmax(local_outputs[i]).item()
                    local_predictions[class_id]['correct'] += int(local_pred == class_id)
                    local_predictions[class_id]['total'] += 1
                    local_predictions[class_id]['entropy'].append(local_entropy[i].item())
                    
                    # Global model results
                    global_pred = torch.argmax(global_outputs[i]).item()
                    global_predictions[class_id]['correct'] += int(global_pred == class_id)
                    global_predictions[class_id]['total'] += 1
                    global_predictions[class_id]['entropy'].append(global_entropy[i].item())
        
        # Calculate metrics for each class
        knowledge_gaps = {}
        for class_id in set(local_predictions.keys()).union(global_predictions.keys()):
            local_acc = 0
            global_acc = 0
            
            if class_id in local_predictions and local_predictions[class_id]['total'] > 0:
                local_acc = local_predictions[class_id]['correct'] / local_predictions[class_id]['total'] * 100
            
            if class_id in global_predictions and global_predictions[class_id]['total'] > 0:
                global_acc = global_predictions[class_id]['correct'] / global_predictions[class_id]['total'] * 100
            
            # Calculate knowledge gap metrics
            knowledge_gaps[class_id] = {
                'local_acc': local_acc,
                'global_acc': global_acc,
                'gap': global_acc - local_acc,
                'local_entropy': np.mean(local_predictions.get(class_id, {'entropy': [0]})['entropy']),
                'global_entropy': np.mean(global_predictions.get(class_id, {'entropy': [0]})['entropy'])
            }
        
        # Store metrics
        if cycle not in self.knowledge_gaps:
            self.knowledge_gaps[cycle] = {}
        self.knowledge_gaps[cycle][client_id] = knowledge_gaps
        
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
            'model_distances': convert_for_json(self.model_distances),
            'gradient_alignments': convert_for_json(self.gradient_alignments),
            'knowledge_gaps': convert_for_json(self.knowledge_gaps)
        }
        
        # Save JSON file
        json_filename = f"{self.strategy_name}_c{self.num_clients}_trial{self.trial_id}.json"
        json_path = os.path.join(self.log_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved experiment data to: {json_path}")