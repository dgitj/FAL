"""
Test script for the HybridEntropyKAFALEntropyFirst strategy.
This script runs a quick test with fewer cycles, clients, and samples
to validate that the entropy-first approach works correctly.
"""

import os
import sys
import subprocess

def run_test():
    """Run a test of the HybridEntropyKAFALEntropyFirst strategy."""
    print("Starting HybridEntropyKAFALEntropyFirst strategy test...")
    
    # Set up command with appropriate parameters for a quick test
    cmd = [
        "python", "main.py",
        "--strategy", "HybridEntropyKAFALEntropyFirst",
        "--cycles", "1",           # Just 1 cycle for quick testing
        "--clients", "3",          # Fewer clients for faster execution
        "--base", "500",           # Smaller initial labeled set
        "--budget", "200",         # Smaller selection budget
        "--alpha", "0.3",          # Moderate non-IID level
        "--dataset", "CIFAR10"     # Using CIFAR10 dataset
    ]
    
    # Run the command
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(result.stdout)
        
        if result.returncode == 0:
            print("Test completed successfully!")
        else:
            print(f"Test failed with return code {result.returncode}")
    except Exception as e:
        print(f"Error running test: {str(e)}")

if __name__ == "__main__":
    run_test()
