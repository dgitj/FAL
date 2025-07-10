"""
Direct fix for SSL server model initialization issue
"""

import sys
import os

# Add the project path
sys.path.append('C:\\Users\\wu0175\\projects\\fal\\FAL')

# Read the main.py file
with open('main.py', 'r') as f:
    lines = f.readlines()

# Find the line where server model is created
found = False
for i, line in enumerate(lines):
    if "# Create server model" in line and i < len(lines) - 5:
        # Check if this is already fixed
        if "config.USE_SSL_PRETRAIN" in lines[i+1]:
            print("File already fixed!")
            sys.exit(0)
        
        # Replace the next few lines
        print(f"Found target at line {i+1}")
        
        # Create the new code block
        new_lines = [
            "            # Create server model - IMPORTANT: Use the same base_model for consistency!\n",
            "            if config.USE_SSL_PRETRAIN:\n",
            "                # For SSL pretrained models, use a copy of the base_model\n", 
            "                server = copy.deepcopy(base_model).to(device)\n",
            "            else:\n",
            "                # For non-SSL models, create new model as before\n",
            "                if config.DATASET == \"MNIST\":\n",
            "                    server = resnet_mnist.preact_resnet8_mnist(num_classes=num_classes).to(device)\n",
            "                else:\n",
            "                    server = resnet.preact_resnet8_cifar(num_classes=num_classes).to(device)\n"
        ]
        
        # Find where the models dict is created
        j = i + 1
        while j < len(lines) and "models = {" not in lines[j]:
            j += 1
        
        if j < len(lines):
            # Replace lines from i+1 to j (exclusive)
            lines[i+1:j] = new_lines
            found = True
            break

if found:
    # Write back the file
    with open('main.py', 'w') as f:
        f.writelines(lines)
    print("✅ Successfully fixed the SSL server model initialization!")
    print("The server model will now use the pretrained encoder when SSL is enabled.")
else:
    print("❌ Could not find the target code section.")
    print("Please check the file manually.")
