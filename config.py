''' Configuration File.
'''

# Dataset selection
DATASET = "CIFAR10"  # Options are "CIFAR10", "SVHN", "CIFAR100", and "MNIST"

if DATASET == "CIFAR10":
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

# Active Learning Strategy Options:
# - "KAFAL"
# - "Entropy"
# - "BADGE"
# - "Random"
# - "Noise"
# - "FEAL"
# - "LOGO"
# - "CoreSet"
# - "AHFAL"


ACTIVE_LEARNING_STRATEGY = "AHFAL"  

SEED = 44
TRIAL_SEED_OFFSET = 2000000
ALPHA = 0.1
BUDGET  = 2500
BASE = 5000
EPOCH=5
COMMUNICATION=100
CYCLES=6
RATIO=1.0
CLIENTS=10
TRIALS=3
LOCAL_MODEL_UPDATE = "Vanilla" # Options are "Vanilla" or "KFCU"  

# training
LR = 0.1
MILESTONES =[260]
NUM_TRAIN = 50000
WDECAY = 5e-4
BATCH = 128
MOMENTUM = 0.9
BETA = [2,2]

# SSL Pre-training Configuration
USE_SSL_PRETRAIN = True  # Set to True to enable SSL pre-training
SSL_METHOD = "SimCLR"  # Options: "SimCLR", "Rotation" (more methods can be added)
SSL_ROUNDS = 50  # Number of federated rounds for SSL pre-training
SSL_LOCAL_EPOCHS = 5  # Number of local epochs per round during SSL
SSL_BATCH_SIZE = 256  # Batch size for SSL (larger is better for contrastive learning)
SSL_LEARNING_RATE = 0.3  # Learning rate for SSL pre-training
SSL_TEMPERATURE = 0.5  # Temperature parameter for SimCLR loss
SSL_PROJECTION_DIM = 128  # Dimension of projection head output
FREEZE_ENCODER = False  # Whether to freeze encoder during active learning cycles