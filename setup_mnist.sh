#!/bin/bash

# Make required directories
mkdir -p data/mnist

# Download MNIST dataset via Python
python -c "
import torch
import torchvision.datasets as datasets

# Download MNIST datasets
print('Downloading MNIST training set...')
train_set = datasets.MNIST(root='data/mnist', train=True, download=True)
print('Downloading MNIST test set...')
test_set = datasets.MNIST(root='data/mnist', train=False, download=True)

print('MNIST dataset successfully downloaded!')
print(f'Training samples: {len(train_set)}')
print(f'Test samples: {len(test_set)}')
"

echo "MNIST setup complete."
