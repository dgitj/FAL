''' Configuration File.
'''

# directory
DATA_ROOT = 'data/cifar-10-batches-py'

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
# - "SSLEntropy"
# - "PseudoEntropy"

ACTIVE_LEARNING_STRATEGY = "PseudoEntropy"

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
LOCAL_MODEL_UPDATE = "SimpleContrastive" # Options are "Vanilla", "ContrastiveEntropy", "SimpleContrastive", and "KFCU"  
DATATSET = "CIFAR10" # Options are "CIFAR10" and "SVHN"

# TCL settings
TCL_TEMPERATURE = 0.5
TCL_LAMBDA = 0.5  # Reduced from 1.0 to a much smaller value
TCL_HARD_MINING_RATIO = 0.5
TCL_ADAPTIVE_TEMP = True

# Simple Contrastive Loss settings
CONTRASTIVE_TEMPERATURE = 0.5
CONTRASTIVE_WEIGHT = 0.5

# SSL settings
# These parameters control the federated SSL autoencoder step
USE_GLOBAL_SSL = True           # Whether to use SSL autoencoder
SSL_FEDERATED = True           # Whether to train the autoencoder in a federated manner
SSL_USE_CONTRASTIVE = True     # ADDED: Whether to use contrastive learning for SSL
SSL_TEMPERATURE = 0.5          # ADDED: Temperature parameter for contrastive loss
SSL_PROXIMAL_MU = 0.01         # ADDED: Proximal term weight for FedProx-like regularization
SSL_LOCAL_EPOCHS = 1           # Number of epochs for local training on each client
SSL_FEDERATED_ROUNDS = 3       # Number of federated rounds for autoencoder training
SSL_BATCH_SIZE = 128           # Batch size for SSL training
SSL_LATENT_DIM = 128           # Dimension of the latent space in the autoencoder
SSL_LEARNING_RATE = 1e-3       # Learning rate for SSL training


# training
LR = 0.1
MILESTONES =[260]
NUM_TRAIN = 50000
WDECAY = 5e-4
BATCH = 128
MOMENTUM = 0.9
BETA = [2,2]