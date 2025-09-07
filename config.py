''' Configuration File.
'''

# Dataset selection
DATASET = "CIFAR10"  # Options are "CIFAR10", "SVHN", "CIFAR100", and "MNIST"

# directory paths for datasets and number of classes
if DATASET == "MNIST":
    DATA_ROOT = 'data/cifar-10-batches-py'
    NUM_CLASSES = 10
elif DATASET == "SVHN":
    DATA_ROOT = 'data/svhn'
    NUM_CLASSES = 10
elif DATASET == "CIFAR100":
    DATA_ROOT = 'data/cifar-100-python'
    NUM_CLASSES = 100
elif DATASET == "MNIST":
    DATA_ROOT = 'data/mnist'
    NUM_CLASSES = 10
else:
    DATA_ROOT = 'data/cifar-10-batches-py'  # Default fallback
    NUM_CLASSES = 10

MODEL_ARCHITECTURE = "resnet8"  # Options are "resnet8" and "mobilenet_v"

ACTIVE_LEARNING_STRATEGY = "PseudoEntropyVariance"  # Options are "KAFAL", "Entropy", "BADGE", "Random", "Noise", "FEAL", "LOGO", "CoreSet", "PseudoEntropyVariance", "AblationClassUncertainty"

# random seed
SEED = 44
TRIAL_SEED_OFFSET = 2000000

# dirichlet partition non-iid level
ALPHA = 0.1

# setting
BUDGET  = 2500
BASE = 5000
EPOCH=1
COMMUNICATION=3
CYCLES=2
RATIO=1.0
CLIENTS=5
TRIALS=1
LOCAL_MODEL_UPDATE = "Vanilla" # Options are "Vanilla" and "KFCU"  

# training
LR = 0.1
MILESTONES =[260]
NUM_TRAIN = 50000
WDECAY = 5e-4
BATCH = 128
MOMENTUM = 0.9
BETA = [2,2]