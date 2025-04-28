"""
Utility functions for model checkpoint loading and conversion
"""

import torch
from collections import OrderedDict

def load_checkpoint_flexible(model, checkpoint, strict=False):
    """
    Tries to load a checkpoint into a model with flexible key matching
    
    Args:
        model: Model to load weights into
        checkpoint: Checkpoint dict or state_dict
        strict: Whether to use strict loading (default: False)
        
    Returns:
        success: Boolean indicating if loading was successful
    """
    # Extract state dict if it's nested in a checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Try direct loading first
    try:
        model.load_state_dict(state_dict, strict=strict)
        return True
    except Exception as e:
        if strict:
            print(f"Strict loading failed: {str(e)}")
            return False
        
        print("Attempting flexible loading...")
        
    # Get model state dict
    model_dict = model.state_dict()
    
    # Create mapping for similar keys
    similar_keys = {}
    for model_key in model_dict.keys():
        # Get the deepest part of the key (last segment)
        deep_key = model_key.split('.')[-1]
        
        # Find checkpoint keys containing this segment
        matching_keys = [k for k in state_dict.keys() if deep_key in k]
        if matching_keys:
            similar_keys[model_key] = matching_keys[0]  # Choose the first match
    
    # Collect the loadable weights
    loadable = {
        model_key: state_dict[ckpt_key] 
        for model_key, ckpt_key in similar_keys.items()
        if model_dict[model_key].shape == state_dict[ckpt_key].shape
    }
    
    # Create a simpler mapping avoiding nested encoder prefixes if needed
    if len(loadable) < 10:  # If few keys matched, try another approach
        loadable = {}
        for model_key in model_dict.keys():
            # Prepare variants of the key
            key_variants = [
                model_key,
                model_key.replace('encoder.encoder.', 'encoder.'),
                model_key.replace('encoder.', 'encoder.encoder.')
            ]
            
            # Try each variant
            for key_var in key_variants:
                if key_var in state_dict and model_dict[model_key].shape == state_dict[key_var].shape:
                    loadable[model_key] = state_dict[key_var]
                    break
    
    # Update model with loadable weights
    if loadable:
        model_dict.update(loadable)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(loadable)}/{len(model_dict)} layers with flexible matching")
        return len(loadable) > len(model_dict) * 0.5  # Success if at least 50% of weights loaded
    else:
        print("Flexible loading failed: No matching keys with compatible shapes found")
        return False
