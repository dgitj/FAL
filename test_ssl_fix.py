"""
Test script to verify SSL model compatibility fix
"""

import torch
import sys
sys.path.append('C:\\Users\\wu0175\\projects\\fal\\FAL')

from models.ssl_models import test_ssl_compatibility

if __name__ == "__main__":
    print("Testing SSL model compatibility...")
    success = test_ssl_compatibility()
    
    if success:
        print("\n✅ All tests passed! The SSL model should now work correctly.")
    else:
        print("\n❌ Tests failed! There are still compatibility issues.")
