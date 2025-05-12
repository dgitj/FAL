# MNIST Dataset for Federated Active Learning

## Overview

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). It contains:
- 60,000 training images
- 10,000 test images

## Setup

To set up the MNIST dataset for this project, run:

```bash
bash setup_mnist.sh
```

This script will:
1. Create the necessary directory structure
2. Download the MNIST dataset using PyTorch's datasets module

## Running Experiments with MNIST

To run federated active learning experiments with MNIST, use:

```bash
python main.py --dataset MNIST --strategy PseudoEntropy
```

You can customize other parameters as needed:

```bash
python main.py --dataset MNIST --strategy PseudoEntropy --clients 10 --alpha 0.1 --cycles 3 --budget 2500 --base 5000
```

## Model Architecture Notes

The model architecture used for MNIST is the same PreactResNet8 used for CIFAR-10/100 and SVHN. While not specifically optimized for MNIST (which typically uses simpler architectures), the network has enough capacity to achieve good performance.

## Expected Performance

With the default federated active learning setup, you can expect:
- Initial accuracy: ~85-90%
- Final accuracy after active learning cycles: ~95-98%

Performance may vary based on the active learning strategy used, the number of clients, and the data distribution across clients.
