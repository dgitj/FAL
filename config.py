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

ACTIVE_LEARNING_STRATEGY = "KAFAL"

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
LOCAL_MODEL_UPDATE = "Vanilla" # Options are "Vanilla" and "KFCU"  
DATATSET = "CIFAR10" # Options are "CIFAR10" and "SVHN"


# training
LR = 0.1
MILESTONES =[260]
NUM_TRAIN = 50000
WDECAY = 5e-4
BATCH = 128
MOMENTUM = 0.9
BETA = [2,2]