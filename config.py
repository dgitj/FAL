''' Configuration File.
'''

# directory
DATA_ROOT = 'data/cifar-10-batches-py'

# setting
BUDGET=300
BASE=1000
EPOCH=3
COMMUNICATION=2
CYCLES=1
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
