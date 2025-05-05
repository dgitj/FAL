# Using CIFAR-100 in FAL

The FAL (Federated Active Learning) framework now supports the CIFAR-100 dataset alongside CIFAR-10 and SVHN.

## Running Experiments with CIFAR-100

To run experiments using CIFAR-100, follow these steps:

1. First, run the setup script to create the necessary directory:
   ```bash
   # Make the script executable
   chmod +x setup_cifar100.sh
   
   # Run the script
   ./setup_cifar100.sh
   ```

2. Run an experiment with CIFAR-100:
   ```bash
   python main.py --dataset CIFAR100 --strategy Entropy --cycles 2 --clients 10
   ```

## Dataset Details

- CIFAR-100 contains 60,000 color images of size 32x32 pixels
- There are 100 fine-grained classes and 20 superclasses (coarse labels)
- Each fine class has 600 images: 500 training images and 100 testing images
- Images have the same format as CIFAR-10, but with more classes and fewer samples per class

## Implementation Notes

- The default implementation uses the fine labels (100 classes)
- Same training interface as CIFAR-10, using train=True/False parameters
- Different normalization values are used for CIFAR-100: mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]

## Performance Considerations

- CIFAR-100 is more challenging than CIFAR-10 due to more classes and fewer examples per class
- You may need to adjust training parameters for optimal performance
- Consider increasing the number of communication rounds or epochs
- Dirichlet partitioning with 100 classes may create more severe non-IID settings

## Comparison with Other Datasets

Running experiments across CIFAR-10, SVHN, and CIFAR-100 allows you to explore how different active learning strategies perform across tasks of varying complexity and class distribution characteristics.
