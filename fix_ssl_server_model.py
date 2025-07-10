"""
Patch file to fix the SSL pretraining issue in main.py
Apply this patch to fix the server model initialization when using SSL pretraining.
"""

def apply_patch():
    import os
    
    # Read the original file
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Find the problematic section
    old_code = """        # Active learning cycles
        for cycle in range(config.CYCLES):
            # Create server model
            if config.DATASET == "MNIST":
                server = resnet_mnist.preact_resnet8_mnist(num_classes=num_classes).to(device)
            else:
                server = resnet.preact_resnet8_cifar(num_classes=num_classes).to(device)
            models = {'clients': client_models, 'server': server}"""
    
    new_code = """        # Active learning cycles
        for cycle in range(config.CYCLES):
            # Create server model - IMPORTANT: Use the same base_model for consistency!
            if config.USE_SSL_PRETRAIN:
                # For SSL pretrained models, use a copy of the base_model
                server = copy.deepcopy(base_model).to(device)
            else:
                # For non-SSL models, create new model as before
                if config.DATASET == "MNIST":
                    server = resnet_mnist.preact_resnet8_mnist(num_classes=num_classes).to(device)
                else:
                    server = resnet.preact_resnet8_cifar(num_classes=num_classes).to(device)
            models = {'clients': client_models, 'server': server}"""
    
    # Replace the code
    if old_code in content:
        content = content.replace(old_code, new_code)
        
        # Write back
        with open('main.py', 'w') as f:
            f.write(content)
        
        print("✅ Patch applied successfully!")
        print("The server model will now use the SSL pretrained encoder when SSL pretraining is enabled.")
        return True
    else:
        print("❌ Could not find the code section to patch.")
        print("The file may have already been modified or has a different format.")
        return False

if __name__ == "__main__":
    apply_patch()
