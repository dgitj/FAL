"""
Evaluation module for federated active learning models.
Provides functions for testing model performance and analyzing per-class metrics.
"""

import torch
import numpy as np
import config  # Import config module for NUM_CLASSES


def evaluate_model(model, dataloader, device, mode='test'):
    """
    Evaluate model accuracy on a dataset.
    
    Args:
        model (torch.nn.Module): The model to evaluate
        dataloader (DataLoader): DataLoader containing evaluation data
        device (torch.device): Device to run evaluation on (cuda/cpu)
        mode (str): Evaluation mode ('test' or 'val')
        
    Returns:
        float: Accuracy percentage
    """
    model.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Handle different model output formats
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                scores = outputs[0]
            else:
                scores = outputs
                
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total


def evaluate_per_class_accuracy(model, test_loader, device, num_classes=None):
    """
    Evaluate accuracy for each class separately.
    
    Args:
        model (torch.nn.Module): The model to evaluate
        test_loader (DataLoader): DataLoader containing test data
        device (torch.device): Device to run evaluation on
        num_classes (int, optional): Number of classes in the dataset. If None, uses config.NUM_CLASSES
        
    Returns:
        dict: Dictionary mapping class indices to their accuracy percentages
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES  # Use config value if not provided
        
    model.eval()
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Handle different model output formats
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                scores = outputs[0]
            else:
                scores = outputs
                
            _, preds = torch.max(scores.data, 1)
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (preds[i] == labels[i]).item()
                class_total[label] += 1
    
    # Calculate accuracy for each class
    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracies[i] = class_correct[i] / class_total[i] * 100
        else:
            class_accuracies[i] = 0.0
    
    return class_accuracies


def evaluate_gradient_alignment(local_model, global_model, dataloader, device):
    """
    Measure alignment between local and global model gradients.
    
    Args:
        local_model (torch.nn.Module): Local client model
        global_model (torch.nn.Module): Global server model
        dataloader (DataLoader): DataLoader for evaluation
        device (torch.device): Device to run evaluation on
        
    Returns:
        tuple: (alignment_score, conflict_ratio)
            - alignment_score: Cosine similarity between gradients (-1 to 1)
            - conflict_ratio: Percentage of parameters with opposing gradients
    """
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create copies to avoid modifying original models
    local_copy = local_model.to(device)
    global_copy = global_model.to(device)
    
    alignments = []
    conflict_ratios = []
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Get local gradients
        local_copy.zero_grad()
        local_outputs = local_copy(inputs)
        if isinstance(local_outputs, tuple):
            local_outputs = local_outputs[0]
        local_loss = criterion(local_outputs, targets)
        local_loss.backward()
        local_grads = []
        for param in local_copy.parameters():
            if param.grad is not None:
                local_grads.append(param.grad.view(-1))
        local_grads = torch.cat(local_grads)
        
        # Get global gradients
        global_copy.zero_grad()
        global_outputs = global_copy(inputs)
        if isinstance(global_outputs, tuple):
            global_outputs = global_outputs[0]
        global_loss = criterion(global_outputs, targets)
        global_loss.backward()
        global_grads = []
        for param in global_copy.parameters():
            if param.grad is not None:
                global_grads.append(param.grad.view(-1))
        global_grads = torch.cat(global_grads)
        
        # Calculate alignment (cosine similarity)
        cos_sim = torch.nn.functional.cosine_similarity(
            local_grads.unsqueeze(0), global_grads.unsqueeze(0))[0]
        alignments.append(cos_sim.item())
        
        # Calculate conflict ratio (percentage of parameters with opposing signs)
        conflicts = ((local_grads * global_grads) < 0).float().mean().item()
        conflict_ratios.append(conflicts)
    
    # Average metrics across batches
    mean_alignment = np.mean(alignments) if alignments else 0
    mean_conflict = np.mean(conflict_ratios) if conflict_ratios else 0
    
    return mean_alignment, mean_conflict


def evaluate_knowledge_gap(local_model, global_model, dataloader, device, num_classes=None):
    """
    Measure knowledge gap between local and global models.
    Compares performance and uncertainty metrics between models.
    
    Args:
        local_model (torch.nn.Module): Local client model
        global_model (torch.nn.Module): Global server model
        dataloader (DataLoader): DataLoader for evaluation
        device (torch.device): Device to run evaluation on
        num_classes (int, optional): Number of classes in the dataset. If None, uses config.NUM_CLASSES
        
    Returns:
        dict: Dictionary with class-wise metrics comparing local and global models
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES  # Use config value if not provided
        
    local_model.eval()
    global_model.eval()
    
    # Initialize metrics storage
    class_metrics = {i: {
        'local_correct': 0, 'global_correct': 0, 'total': 0,
        'local_entropy': [], 'global_entropy': []
    } for i in range(num_classes)}
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get local and global predictions
            local_outputs = local_model(inputs)
            global_outputs = global_model(inputs)
            
            # Handle different output formats
            if isinstance(local_outputs, tuple):
                local_outputs = local_outputs[0]
            if isinstance(global_outputs, tuple):
                global_outputs = global_outputs[0]
            
            local_probs = torch.nn.functional.softmax(local_outputs, dim=1)
            global_probs = torch.nn.functional.softmax(global_outputs, dim=1)
            
            # Calculate entropy (uncertainty)
            local_entropy = -torch.sum(local_probs * torch.log(local_probs + 1e-10), dim=1)
            global_entropy = -torch.sum(global_probs * torch.log(global_probs + 1e-10), dim=1)
            
            # Get predictions
            local_preds = torch.argmax(local_outputs, dim=1)
            global_preds = torch.argmax(global_outputs, dim=1)
            
            # Update metrics for each class
            for i, label in enumerate(labels):
                class_id = label.item()
                
                # Update counts
                class_metrics[class_id]['total'] += 1
                class_metrics[class_id]['local_correct'] += int(local_preds[i] == class_id)
                class_metrics[class_id]['global_correct'] += int(global_preds[i] == class_id)
                
                # Store entropy values
                class_metrics[class_id]['local_entropy'].append(local_entropy[i].item())
                class_metrics[class_id]['global_entropy'].append(global_entropy[i].item())
    
    # Compute final metrics for each class
    results = {}
    for class_id, metrics in class_metrics.items():
        if metrics['total'] > 0:
            local_acc = metrics['local_correct'] / metrics['total'] * 100
            global_acc = metrics['global_correct'] / metrics['total'] * 100
            
            results[class_id] = {
                'local_acc': local_acc,
                'global_acc': global_acc,
                'gap': global_acc - local_acc,
                'local_entropy': np.mean(metrics['local_entropy']) if metrics['local_entropy'] else 0,
                'global_entropy': np.mean(metrics['global_entropy']) if metrics['global_entropy'] else 0
            }
        else:
            results[class_id] = {
                'local_acc': 0, 'global_acc': 0, 'gap': 0,
                'local_entropy': 0, 'global_entropy': 0
            }
    
    return results


def create_confusion_matrix(model, dataloader, device, num_classes=None):
    """
    Generate confusion matrix for model predictions.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (DataLoader): DataLoader for evaluation
        device (torch.device): Device to run evaluation on
        num_classes (int, optional): Number of classes. If None, uses config.NUM_CLASSES
        
    Returns:
        numpy.ndarray: Confusion matrix where rows are true labels and columns are predicted labels
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES  # Use config value if not provided
        
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get predictions
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            _, preds = torch.max(outputs, 1)
            
            # Update confusion matrix
            for true, pred in zip(labels, preds):
                confusion[true.item()][pred.item()] += 1
                
    return confusion