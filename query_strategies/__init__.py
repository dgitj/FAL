import os
import importlib

# Dynamically load all query strategy modules in the directory
strategy_modules = {}

for filename in os.listdir(os.path.dirname(__file__)):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]  # Remove ".py" extension
        module = importlib.import_module(f"query_strategies.{module_name}")
        strategy_modules[module_name] = module

# Explicitly import key modules for easier access
from .badge import sample as badge_sample
from .entropy import sample as entropy_sample
from .kafal import sample as kafal_sample

__all__ = ["badge_sample", "entropy_sample", "kafal_sample", "strategy_modules"]
