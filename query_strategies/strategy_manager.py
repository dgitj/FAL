from query_strategies.kafal import KAFALSampler
from query_strategies.entropy import EntropySampler
from query_strategies.badge import BADGESampler
from query_strategies.random import RandomSampler
from query_strategies.noise_stability import NoiseStabilitySampler
from query_strategies.feal import FEALSampler

from config import ACTIVE_LEARNING_STRATEGY

class StrategyManager:
    def __init__(self, strategy_name, loss_weight_list=None, device="cuda"):
        self.device = device
        self.sampler = self._initialize_strategy(strategy_name , loss_weight_list)
        self.strategy_name = strategy_name
        self.loss_weight_list = loss_weight_list

    
    def _initialize_strategy(self, strategy_name, loss_weight_list):
        if strategy_name == "KAFAL":
            return KAFALSampler(loss_weight_list, self.device)
        elif strategy_name == "Entropy":
            return EntropySampler(self.device)
        elif strategy_name == "BADGE":
            return BADGESampler(self.device)
        elif strategy_name == "Random":
            return RandomSampler(self.device)
        elif strategy_name == "Noise":
            return NoiseStabilitySampler(self.device)
        elif strategy_name == "FEAL":
            return FEALSampler(self.device)
        else:
            raise ValueError(f"Invalid strategy name: {strategy_name}")
        
    def select_samples(self, model, model_server, unlabeled_loader, c, unlabeled_set, num_samples):
        if not self.strategy_name:
            raise ValueError("Strategy not set. Use set_strategy() to set the strategy.")  
        if self.strategy_name == "KAFAL":
            return self.sampler.select_samples(model, model_server, unlabeled_loader, c, unlabeled_set, num_samples) 
        elif self.strategy_name == "FEAL":
            return self.sampler.select_samples(
                global_model=model_server,            # Global model
                local_model=model,                    # Local model
                data_unlabeled=unlabeled_loader.dataset,
                unlabeled_set=unlabeled_set,
                query_num=num_samples,
                args=self.args
            ) 
        else:
            # These strategies do not require the model_server and c arguments
            return self.sampler.select_samples(model, unlabeled_loader, unlabeled_set, num_samples)
    
