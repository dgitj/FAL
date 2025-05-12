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
from query_strategies.pseudo_confidence import PseudoClassBalancedConfidenceSampler
from query_strategies.pseudo_entropy import PseudoClassBalancedEntropySampler
from query_strategies.adaptive_entropy import AdaptiveEntropySampler
from query_strategies.hybrid_entropy_kafal import HybridEntropyKAFALSampler
from query_strategies.hybrid_entropy_kafal_entropy_first import HybridEntropyKAFALSampler as HybridEntropyKAFALEntropyFirstSampler
from query_strategies.hybrid_entropy_kafal_class_differentiated import HybridEntropyKAFALSampler as HybridEntropyKAFALClassDifferentiatedSampler
from query_strategies.ablation_class_uncertainty import AblationClassUncertaintySampler

from config import ACTIVE_LEARNING_STRATEGY

class StrategyManager:
    def __init__(self, strategy_name, loss_weight_list=None, device="cuda", global_autoencoder=None, confidence_threshold=None):
        self.device = device
        self.strategy_name = strategy_name
        self.loss_weight_list = loss_weight_list
        self.clients_processed = 0
        self.total_clients = 0
        self.labeled_set_list = None
        self.labeled_set_classes_list = None  # Added to store class information
        self.confidence_threshold = confidence_threshold
        
        # Initialize the sampling strategy
        self.sampler = self._initialize_strategy(strategy_name, loss_weight_list)
    
    def set_total_clients(self, num_clients):
        """Set the total number of clients for global optimization strategies."""
        self.total_clients = num_clients
    
    def set_labeled_set_list(self, labeled_set_list):
        """Set the labeled set list for the strategies that need it."""
        self.labeled_set_list = labeled_set_list
        
    def set_labeled_set_classes_list(self, labeled_set_classes_list):
        """Set the list of classes for labeled samples for strategies that need it."""
        self.labeled_set_classes_list = labeled_set_classes_list
    
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
            
            
        elif strategy_name == "PseudoConfidence":
            print(f"[StrategyManager] Initializing PseudoConfidence strategy")
            return PseudoClassBalancedConfidenceSampler(self.device)
            
        elif strategy_name == "PseudoEntropy":
            print(f"[StrategyManager] Initializing PseudoEntropy strategy")
            return PseudoClassBalancedEntropySampler(self.device)
            
        elif strategy_name == "AdaptiveEntropy":
            print(f"[StrategyManager] Initializing AdaptiveEntropy strategy")
            return AdaptiveEntropySampler(self.device)
            
        elif strategy_name == "HybridEntropyKAFAL":
            print(f"[StrategyManager] Initializing HybridEntropyKAFAL strategy")
            return HybridEntropyKAFALSampler(self.device)
            
        elif strategy_name == "HybridEntropyKAFALEntropyFirst":
            print(f"[StrategyManager] Initializing HybridEntropyKAFAL with Entropy-First approach")
            return HybridEntropyKAFALEntropyFirstSampler(self.device)
            
        elif strategy_name == "HybridEntropyKAFALClassDifferentiated":
            print(f"[StrategyManager] Initializing HybridEntropyKAFAL with Class-Differentiated approach")
            return HybridEntropyKAFALClassDifferentiatedSampler(self.device)
            
        elif strategy_name == "AblationClassUncertainty":
            print(f"[StrategyManager] Initializing Ablation: Class-Specific Uncertainty Only")
            return AblationClassUncertaintySampler(self.device)

        else:
            raise ValueError(f"Invalid strategy name: {strategy_name}")
            
        print(f"{strategy_name} strategy initialized successfully")
        
    def select_samples(self, model, model_server, unlabeled_loader, c, unlabeled_set, num_samples, labeled_set=None, seed=None, global_class_distribution=None, class_variance_stats=None, current_round=0, total_rounds=5):
        """
        Select samples using the specified active learning strategy.
        
        Args:
            model: Client model
            model_server: Server model (only used for some strategies)
            unlabeled_loader: DataLoader for unlabeled data
            c: Client ID (only used for KAFAL)
            unlabeled_set: List of unlabeled sample indices
            num_samples: Number of samples to select
            labeled_set: List of labeled sample indices (optional)
            seed: Random seed for reproducibility (optional)
            global_class_distribution: Global class distribution from server (optional)
            class_variance_stats: Statistics about class variance across clients (optional)
            current_round: Current active learning round (optional)
            total_rounds: Total number of active learning rounds (optional)
            
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
        elif self.strategy_name in ["PseudoConfidence", "PseudoEntropy"]:
            # Both strategies need global class distribution
            if labeled_set is None and self.labeled_set_list is not None and c < len(self.labeled_set_list):
                labeled_set = self.labeled_set_list[c]
                
            # Pass global class distribution to strategy
            return self.sampler.select_samples(
                model, model_server, unlabeled_loader, c, unlabeled_set, 
                num_samples, labeled_set=labeled_set, seed=seed,
                global_class_distribution=global_class_distribution
            )
            
        elif self.strategy_name == "AdaptiveEntropy":
            # AdaptiveEntropy needs global class distribution and variance stats
            if labeled_set is None and self.labeled_set_list is not None and c < len(self.labeled_set_list):
                labeled_set = self.labeled_set_list[c]
                
            # Pass all required parameters to AdaptiveEntropy strategy
            return self.sampler.select_samples(
                model, model_server, unlabeled_loader, c, unlabeled_set, 
                num_samples, labeled_set=labeled_set, seed=seed,
                global_class_distribution=global_class_distribution,
                class_variance_stats=class_variance_stats,
                current_round=current_round,
                total_rounds=total_rounds
            )
            
        elif self.strategy_name == "HybridEntropyKAFAL":
            # This strategy needs both models, client ID, and class variance stats
            try:
                if labeled_set is None and self.labeled_set_list is not None and c < len(self.labeled_set_list):
                    labeled_set = self.labeled_set_list[c]
                
                # Get labeled set class information if available
                labeled_set_classes = None
                if hasattr(self, 'labeled_set_classes_list') and self.labeled_set_classes_list is not None \
                   and c < len(self.labeled_set_classes_list):
                    labeled_set_classes = self.labeled_set_classes_list[c]
                    
                # Pass all required parameters to HybridEntropyKAFAL strategy
                return self.sampler.select_samples(
                    model, model_server, unlabeled_loader, c, unlabeled_set, 
                    num_samples, labeled_set=labeled_set, seed=seed,
                    global_class_distribution=global_class_distribution,
                    class_variance_stats=class_variance_stats,
                    current_round=current_round,
                    total_rounds=total_rounds,
                    labeled_set_classes=labeled_set_classes
                )
            except ValueError as e:
                print(f"ERROR in HybridEntropyKAFAL: {str(e)}")
                raise  # Re-raise the exception to be handled by the caller
                
        elif self.strategy_name == "HybridEntropyKAFALEntropyFirst":
            # This strategy needs both models, client ID, and class variance stats
            try:
                if labeled_set is None and self.labeled_set_list is not None and c < len(self.labeled_set_list):
                    labeled_set = self.labeled_set_list[c]
                
                # Get labeled set class information if available
                labeled_set_classes = None
                if hasattr(self, 'labeled_set_classes_list') and self.labeled_set_classes_list is not None \
                   and c < len(self.labeled_set_classes_list):
                    labeled_set_classes = self.labeled_set_classes_list[c]
                    
                # Pass all required parameters to HybridEntropyKAFAL strategy
                return self.sampler.select_samples(
                    model, model_server, unlabeled_loader, c, unlabeled_set, 
                    num_samples, labeled_set=labeled_set, seed=seed,
                    global_class_distribution=global_class_distribution,
                    class_variance_stats=class_variance_stats,
                    current_round=current_round,
                    total_rounds=total_rounds,
                    labeled_set_classes=labeled_set_classes
                )
            except ValueError as e:
                print(f"ERROR in HybridEntropyKAFALEntropyFirst: {str(e)}")
                raise  # Re-raise the exception to be handled by the caller
                
        elif self.strategy_name in ["HybridEntropyKAFALClassDifferentiated", "AblationClassUncertainty"]:
            # These strategies need both models, client ID, and class variance stats
            try:
                if labeled_set is None and self.labeled_set_list is not None and c < len(self.labeled_set_list):
                    labeled_set = self.labeled_set_list[c]
                
                # Get labeled set class information if available
                labeled_set_classes = None
                if hasattr(self, 'labeled_set_classes_list') and self.labeled_set_classes_list is not None \
                   and c < len(self.labeled_set_classes_list):
                    labeled_set_classes = self.labeled_set_classes_list[c]
                    
                # Pass all required parameters to class-differentiated strategy
                return self.sampler.select_samples(
                    model, model_server, unlabeled_loader, c, unlabeled_set, 
                    num_samples, labeled_set=labeled_set, seed=seed,
                    global_class_distribution=global_class_distribution,
                    class_variance_stats=class_variance_stats,
                    current_round=current_round,
                    total_rounds=total_rounds,
                    labeled_set_classes=labeled_set_classes
                )
            except ValueError as e:
                print(f"ERROR in {self.strategy_name}: {str(e)}")
                raise  # Re-raise the exception to be handled by the caller
                
        elif self.strategy_name in ["GlobalOptimal", "CoreSetGlobalOptimal"]:
            # These strategies need both models, client ID, and access to true labels
            if labeled_set is None and self.labeled_set_list is not None and c < len(self.labeled_set_list):
                labeled_set = self.labeled_set_list[c]         
            # If this is the last client to be processed, allocate budget globally
            if self.clients_processed >= self.total_clients:
                # Reset counter for next round
                self.clients_processed = 0
            
            # Select samples based on global allocation
            return self.sampler.select_samples(model, model_server, unlabeled_loader, c, unlabeled_set, num_samples, labeled_set=labeled_set, seed=seed)    
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")
