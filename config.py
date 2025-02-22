''' Configuration File.
'''

# directory
DATA_ROOT = 'data/cifar-10-batches-py'

# Active Learning Strategy Options:
# - "KAFAL"
# - "Entropy"
# - "BADGE"
# - "Random"

ACTIVE_LEARNING_STRATEGY = "Random"


# setting
BUDGET  = 2500
BASE = 5000
EPOCH=2
COMMUNICATION=2
CYCLES =2
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