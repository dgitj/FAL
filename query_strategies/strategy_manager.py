from query_strategies.kafal import KAFALSampler
from query_strategies.entropy import EntropySampler
from query_strategies.badge import BADGESampler

class StrategyManager:
    def __init__(self, strategy_name, loss_weight_list=None, device="cuda"):
        self.device = device
        self.sampler = self._initialize_strategy(strategy_name , loss_weight_list)
        self.loss_weight_list = loss_weight_list

    
    def _initialize_strategy(self, strategy_name, loss_weight_list):
        if strategy_name == "KAFAL":
            return KAFALSampler(loss_weight_list, self.device)
        elif strategy_name == "Entropy":
            return EntropySampler(self.device)
        elif strategy_name == "BADGE":
            return BADGESampler(self.device)
        else:
            raise ValueError(f"Invalid strategy name: {strategy_name}")
        
    def select_samples(self, *args, **kwargs):
        if not self.strategy_name:
            raise ValueError("Strategy not set. Use set_strategy() to set the strategy.")   
        return self.sampler.select_samples(*args, **kwargs) 
    
