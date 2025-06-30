# AHFAL: Adaptive Hierarchical Federated Active Learning


### Installation
```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### Default Settings
```python
CLIENTS = 10        # Federated clients
ALPHA = 0.1         # Dirichlet non-IID parameter  
CYCLES = 6          # Active learning cycles
BUDGET = 2500       # Labels per cycle
BASE = 5000         # Initial labeled samples
EPOCHS = 5          # Local training epochs
COMMUNICATION = 100   # FL communication rounds
```

### Key Parameters
```bash
--strategy {AHFAL,KAFAL,Entropy,BADGE,Random}  # Active learning method
--dataset {CIFAR10,CIFAR100,SVHN,MNIST}        # Dataset choice
--clients N          # Number of federated clients (default: 10)
--alpha FLOAT        # Non-IID level (default: 0.1, lower = more non-IID)
--cycles N           # Active learning cycles (default: 3)
--budget N           # Samples to label per cycle (default: 2500)
--seed N             # Random seed for reproducibility
--save-checkpoints   # Enable model checkpointing (optional)
```


## 📁 Repository Structure

```
FAL/
├── 📄 main.py                    # Main experiment entry point
├── ⚙️ config.py                  # Global configuration
├── 📋 requirements.txt           # Dependencies
├── 🧠 query_strategies/          # Active learning methods
│   ├── 🌟 AHFAL.py              # Our proposed method
│   ├── 📊 kafal.py               # KAFAL baseline
│   ├── 🎯 entropy.py             # Entropy sampling
│   ├── 🎪 badge.py               # BADGE sampling
│   └── 🎲 random.py              # Random baseline
|   |__ ...
├── 🏋️ training/                  # Federated learning framework
│   ├── trainer.py               # Main FL trainer
│   ├── evaluation.py            # Model evaluation
│   └── utils.py                 # Training utilities
├── 🏗️ models/                   # Neural architectures
├── 📊 data/                     # Data partitioning
├── 📈 analysis/                 # Logging utilities
└── 📁 analysis_logs/            # Experiment results (JSON)
```

---

## 💾 Output & Results

### Experiment Logs
Results are automatically saved to `analysis_logs/` as JSON files containing:
- ✅ Global model accuracy per cycle
- 📊 Per-class accuracy breakdown  
- 🎯 Sample selection statistics
- 📈 Class distribution analysis

### Example Output File
```
analysis_logs/AHFAL_c10_trial1.json
```

---
## 🔬 Reproducibility

### Deterministic Execution
-  All experiments use comprehensive seeding 
- Results are deterministic given same environment
- We used seed 44 for the first trial of each experiment
