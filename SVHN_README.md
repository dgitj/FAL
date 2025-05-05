# Using SVHN Dataset in FAL

The FAL (Federated Active Learning) framework now supports the Street View House Numbers (SVHN) dataset in addition to CIFAR-10.

## Running Experiments with SVHN

To run experiments using SVHN, follow these steps:

1. First, run the setup script to create the necessary directory:
   ```bash
   # Make the script executable
   chmod +x setup_svhn.sh
   
   # Run the script
   ./setup_svhn.sh
   ```

2. Run an experiment with SVHN:
   ```bash
   python main.py --dataset SVHN --strategy Entropy --cycles 2 --clients 10
   ```

## Dataset Details

- SVHN contains 73,257 digits for training and 26,032 digits for testing
- Images are 32x32 color digits similar to MNIST but with more challenging real-world settings
- Class distribution is not balanced in SVHN (unlike CIFAR-10)
- Default augmentations include random cropping (no horizontal flips, as they would distort digits)

## Implementation Notes

- The SVHN dataset uses the `split` parameter ('train'/'test') instead of the `train` parameter (True/False) used by CIFAR-10
- SVHN labels are accessed via the `labels` attribute rather than through the dataset's `__getitem__` method
- Different normalization values are used for SVHN: mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]

## Performance Considerations

- When using SVHN with the Dirichlet partitioner, note that SVHN's natural class distribution is already imbalanced (unlike CIFAR-10)
- You may need to adjust training parameters for optimal performance with SVHN

## Comparison with CIFAR-10

Running experiments on both datasets can provide interesting insights into how active learning strategies perform across different image classification tasks.
