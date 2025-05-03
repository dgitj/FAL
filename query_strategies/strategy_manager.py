from query_strategies.kafal import KAFALSampler
from query_strategies.entropy import EntropySampler
from query_strategies.badge import BADGESampler
from query_strategies.random import RandomSampler
from query_strategies.noise_stability import NoiseStabilitySampler
from query_strategies.feal import FEALSampler
from query_strategies.logo import LoGoSampler
from query_strategies.entropy_global_optimal import ClassBalancedEntropySampler
from query_strategies.coreset import CoreSetSampler
from query_strategies.coreset_global_optimal import ClassBalancedCoreSetSampler
from query_strategies.pseudo_entropy import PseudoClassBalancedEntropySampler
from query_strategies.pseudo_confidence import PseudoClassBalancedConfidenceSampler

from config import ACTIVE_LEARNING_STRATEGY

class StrategyManager:
    def __init__(self, strategy_name, loss_weight_list=None, device="cuda", global_autoencoder=None, confidence_threshold=None):
        self.device = device
        self.strategy_name = strategy_name
        self.loss_weight_list = loss_weight_list
        self.clients_processed = 0
        self.total_clients = 0
        self.labeled_set_list = None
        self.confidence_threshold = confidence_threshold
        
        # Initialize the sampling strategy
        self.sampler = self._initialize_strategy(strategy_name, loss_weight_list)
    
    def set_total_clients(self, num_clients):
        """Set the total number of clients for global optimization strategies."""
        self.total_clients = num_clients
    
    def set_labeled_set_list(self, labeled_set_list):
        """Set the labeled set list for the strategies that need it."""
        self.labeled_set_list = labeled_set_list
    
    def _initialize_strategy(self, strategy_name, loss_weight_list):
        """
        Initialize the appropriate active learning sampling strategy.
        
        Args:
            strategy_name: Name of the strategy to initialize
            loss_weight_list: Class weights for KAFAL strategy
            
        Returns:
            An initialized strategy sampler object
        
        Raises:
            ValueError: If an invalid strategy name is provided
        """
        print(f"Initializing {strategy_name} active learning strategy...")
        
        if strategy_name == "KAFAL":
            if loss_weight_list is None:
                raise ValueError("KAFAL strategy requires loss_weight_list")
            return KAFALSampler(loss_weight_list, self.device)
        
        elif strategy_name == "GlobalOptimal":
            return ClassBalancedEntropySampler(self.device)
            
        elif strategy_name == "CoreSetGlobalOptimal":
            return ClassBalancedCoreSetSampler(self.device)
        elif strategy_name == "Entropy":
            return EntropySampler(self.device)
            
        elif strategy_name == "BADGE":
            return BADGESampler(self.device)
            
        elif strategy_name == "Random":
            return RandomSampler(self.device)
            
        elif strategy_name == "Noise":
            # Default parameters from the paper
            return NoiseStabilitySampler(
                device=self.device, 
                noise_scale=0.001,  # Default value from paper
                num_sampling=50     # Default value from paper
            )
            
        elif strategy_name == "FEAL":
            # FEAL with explicit parameters from the paper
            return FEALSampler(
                device=self.device, 
                n_neighbor=5,   # Default from paper implementation
                cosine=0.85     # Default from paper implementation
            )
        elif strategy_name == "LOGO":
            return LoGoSampler(self.device)
            
        elif strategy_name == "CoreSet":
            return CoreSetSampler(self.device)
            
        elif strategy_name == "PseudoEntropy":
            # Use provided confidence threshold or default to 0.0
            confidence = self.confidence_threshold if self.confidence_threshold is not None else 0.0
            print(f"[StrategyManager] Initializing PseudoEntropy with confidence threshold: {confidence}")
            return PseudoClassBalancedEntropySampler(self.device, confidence_threshold=confidence)
            
        elif strategy_name == "PseudoConfidence":
            print(f"[StrategyManager] Initializing PseudoConfidence strategy")
            return PseudoClassBalancedConfidenceSampler(self.device)

        else:
            raise ValueError(f"Invalid strategy name: {strategy_name}")
            
        print(f"{strategy_name} strategy initialized successfully")
        
    def select_samples(self, model, model_server, unlabeled_loader, c, unlabeled_set, num_samples, seed=None):
        """
        Select samples using the specified active learning strategy.
        
        Args:
            model: Client model
            model_server: Server model (only used for some strategies)
            unlabeled_loader: DataLoader for unlabeled data
            c: Client ID (only used for KAFAL)
            unlabeled_set: List of unlabeled sample indices
            num_samples: Number of samples to select
            
        Returns:
            tuple: (selected_samples, remaining_unlabeled)
        """
        if not self.strategy_name:
            raise ValueError("Strategy not set. Use set_strategy() to set the strategy.")
            
        # Handle different parameter requirements for each strategy
        if self.strategy_name == "KAFAL":
            # KAFAL needs client ID for its specialized knowledge component
            return self.sampler.select_samples(model, model_server, unlabeled_loader, c, unlabeled_set, num_samples, seed=seed) 
        elif self.strategy_name == "FEAL":
            # FEAL needs both global and local models for discrepancy
            return self.sampler.select_samples(global_model=model_server, local_model=model, unlabeled_loader=unlabeled_loader, unlabeled_set=unlabeled_set, num_samples=num_samples, seed=seed) 
        elif self.strategy_name == "Noise":
            # NoiseStability just needs the local model
            return self.sampler.select_samples(model, unlabeled_loader, unlabeled_set, num_samples, seed=seed)
        elif self.strategy_name == "BADGE":
            # BADGE just needs the local model
            return self.sampler.select_samples(model, unlabeled_loader, unlabeled_set, num_samples, seed=seed)
        elif self.strategy_name == "Entropy":
            # Entropy just needs the local model
            return self.sampler.select_samples(model, unlabeled_loader, unlabeled_set, num_samples, seed=seed)
        elif self.strategy_name == "Random":
            return self.sampler.select_samples(model, unlabeled_loader, unlabeled_set, num_samples, seed=seed)
        elif self.strategy_name == "LOGO":
            return self.sampler.select_samples(model, model_server, unlabeled_loader, c, unlabeled_set, num_samples, seed=seed)
        elif self.strategy_name == "CoreSet":
            # CoreSet can benefit from knowing which samples are already labeled
            labeled_set = None
            if self.labeled_set_list is not None and c < len(self.labeled_set_list):
                labeled_set = self.labeled_set_list[c]
            return self.sampler.select_samples(model, model_server, unlabeled_loader, c, unlabeled_set, num_samples, labeled_set=labeled_set, seed=seed)
        elif self.strategy_name in ["GlobalOptimal", "CoreSetGlobalOptimal", "PseudoEntropy", "PseudoConfidence"]:
          # These strategies need both models, client ID, and access to true labels
          labeled_set = None
          if self.labeled_set_list is not None and c < len(self.labeled_set_list):
              labeled_set = self.labeled_set_list[c]         
          # If this is the last client to be processed, allocate budget globally
          if self.clients_processed >= self.total_clients:
              # Reset counter for next round
              self.clients_processed = 0
              
              # Allocate budget across all clients
            #  client_ids = list(range(self.total_clients))
             # total_budget = num_samples * self.total_clients  # Total budget across all clients
             # self.sampler.allocate_global_budget(client_ids, total_budget)
          
          # Select samples based on global allocation
          return self.sampler.select_samples(model, model_server, unlabeled_loader, c, unlabeled_set, num_samples, labeled_set=labeled_set, seed=seed)    
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")