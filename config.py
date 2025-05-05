''' Configuration File.
'''

# Dataset selection
DATASET = "CIFAR100"  # Options are "CIFAR10", "SVHN", and "CIFAR100"

# directory paths for datasets and number of classes
if DATASET == "CIFAR10":
    DATA_ROOT = 'data/cifar-10-batches-py'
    NUM_CLASSES = 10
elif DATASET == "SVHN":
    DATA_ROOT = 'data/svhn'
    NUM_CLASSES = 10
elif DATASET == "CIFAR100":
    DATA_ROOT = 'data/cifar-100-python'
    NUM_CLASSES = 100
else:
    DATA_ROOT = 'data/cifar-10-batches-py'  # Default fallback
    NUM_CLASSES = 10

# Active Learning Strategy Options:
# - "KAFAL"
# - "Entropy"
# - "BADGE"
# - "Random"
# - "Noise"
# - "FEAL"
# - "LOGO"
# - "GlobalOptimal"
# - "CoreSet"
# - "CoreSetGlobalOptimal"
# - "PseudoEntropy"
# - "PseudoConfidence"

ACTIVE_LEARNING_STRATEGY = "PseudoConfidence"

# random seed
SEED = 44
TRIAL_SEED_OFFSET = 2000000

# dirichlet partition non-iid level
ALPHA = 0.1

# setting
BUDGET  = 2500
BASE = 5000
EPOCH=2
COMMUNICATION=2
CYCLES=2
RATIO=0.8
CLIENTS=10
TRIALS=1
LOCAL_MODEL_UPDATE = "DebiasedContrastive" # Options are "Vanilla", "SimpleContrastive", "DebiasedContrastive", and "KFCU"  
# This is now set at the top of the file
# DATASET = "CIFAR10" # Options are "CIFAR10" and "SVHN"

# Simple Contrastive Loss settings
CONTRASTIVE_TEMPERATURE = 0.5
CONTRASTIVE_WEIGHT = 1.75

# Debiased Contrastive Loss settings
DCL_TEMPERATURE = 0.5     # Temperature parameter for DCL
DCL_BETA = 0.9           # Beta parameter for negative sample re-weighting
DCL_LAMBDA = 1.0         # Weight for DCL when combined with cross-entropy


# training
LR = 0.1
MILESTONES =[260]
NUM_TRAIN = 50000
WDECAY = 5e-4
BATCH = 128
MOMENTUM = 0.9
BETA = [2,2]