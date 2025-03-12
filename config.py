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


ACTIVE_LEARNING_STRATEGY = "Entropy"

# same random sampling for base across experiments
BASE_RANDOM_SEED = 42

# setting
BUDGET  = 2500
BASE = 5000
EPOCH=40
COMMUNICATION=50
CYCLES =6
RATIO=0.8
CLIENTS=10
TRIALS=1

# training
LR = 0.1
MILESTONES =[260]
NUM_TRAIN = 50000
WDECAY = 5e-4
BATCH = 128
MOMENTUM = 0.9
BETA = [2,2]