import os
import sys
import argparse
import shutil
import subprocess
import re
import time
import threading
from datetime import datetime, timedelta
import json
from pathlib import Path

def create_experiment_directory(strategy_name, config_updates=None):
    """
    Create a separate experiment directory with all necessary files
    
    Args:
        strategy_name (str): The strategy name
        config_updates (dict): Configuration updates
        
    Returns:
        str: Path to the experiment directory
    """
    # Create a descriptive directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = "_".join([f"{k}{v}" for k, v in config_updates.items()]) if config_updates else ""
    exp_dir = f"exp_{strategy_name}_{config_str}_{timestamp}"
    
    # Create the directory
    os.makedirs(exp_dir, exist_ok=True)
    
    # Copy all necessary files to the directory
    required_files = [
        'main_multiprocessing.py',
        'multiprocessing_utils.py',
        'models',
        'query_strategies',
        'data',
        'config.py',
        'distribution'
    ]
    
    for item in required_files:
        if os.path.isfile(item):
            shutil.copy(item, exp_dir)
        elif os.path.isdir(item):
            shutil.copytree(item, os.path.join(exp_dir, item), dirs_exist_ok=True)
    
    # Create a modified config.py in the experiment directory
    config_path = os.path.join(exp_dir, 'config.py')
    
    with open(config_path, 'r') as f:
        config_content = f.read()
    
    # Update the strategy
    updated_config = re.sub(
        r'ACTIVE_LEARNING_STRATEGY\s*=\s*".*"', 
        f'ACTIVE_LEARNING_STRATEGY = "{strategy_name}"', 
        config_content
    )
    
    # Apply additional config updates
    if config_updates:
        for key, value in config_updates.items():
            # Handle different value types
            if isinstance(value, str):
                value_str = f'"{value}"'
            elif isinstance(value, list):
                value_str = str(value)
            else:
                value_str = str(value)
                
            # Use regex for precise replacement
            pattern = rf'^{key}\s*=\s*.*$'
            replacement = f'{key} = {value_str}'
            updated_config = re.sub(pattern, replacement, updated_config, flags=re.MULTILINE)
    
    with open(config_path, 'w') as f:
        f.write(updated_config)
    
    return exp_dir

def monitor_experiment(process, strategy_name, exp_dir, result_dir):
    """
    Monitor a running experiment process and collect results when done
    
    Args:
        process: The subprocess.Popen process object
        strategy_name: Name of the strategy
        exp_dir: Experiment directory path
        result_dir: Results directory path
    """
    # Get start time
    start_time = time.time()
    
    # Wait for the process to complete
    process.wait()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print completion message
    print(f"\nExperiment with {strategy_name} completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Create the result directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)
    
    # Move result files from experiment directory to results directory
    for filename in os.listdir(exp_dir):
        if filename.startswith('log') or filename.endswith(('.png', '.json', '.log')):
            file_path = os.path.join(exp_dir, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, os.path.join(result_dir, filename))
    
    # Copy config used
    shutil.copy(os.path.join(exp_dir, 'config.py'), os.path.join(result_dir, 'config.py'))
    
    # Create experiment summary file
    summary = {
        'strategy': strategy_name,
        'runtime_seconds': elapsed_time,
        'runtime_formatted': f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
        'exit_code': process.returncode,
        'completed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(result_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Output something to indicate process is complete
    print(f"Results for {strategy_name} saved to {result_dir}")
    
    # Optionally clean up experiment directory to save space
    # Uncomment if you want to automatically clean up:
    # shutil.rmtree(exp_dir)

def get_current_strategy():
    """Get the current strategy configured in config.py"""
    # Find the active learning strategy in config.py
    with open('config.py', 'r') as f:
        content = f.read()
    
    match = re.search(r'ACTIVE_LEARNING_STRATEGY\s*=\s*"([^"]*)"', content)
    if match:
        return match.group(1)
    return "Unknown"

def run_parallel_experiments(strategies, config_updates=None, num_processes=None, max_parallel=None):
    """
    Run multiple strategy experiments in parallel
    
    Args:
        strategies (list): List of strategies to evaluate
        config_updates (dict): Configuration updates to apply
        num_processes (int): Number of processes per experiment
        max_parallel (int): Maximum number of parallel experiments to run
    """
    # Create results base directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base_dir = f"results_parallel_{timestamp}"
    os.makedirs(results_base_dir, exist_ok=True)
    
    # Save experiment configuration
    experiment_config = {
        'strategies': strategies,
        'config_updates': config_updates,
        'num_processes': num_processes,
        'max_parallel': max_parallel,
        'started_at': timestamp
    }
    
    with open(os.path.join(results_base_dir, 'experiment_config.json'), 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    # Track all processes and their monitoring threads
    all_processes = []
    active_processes = []
    monitor_threads = []
    
    # Print header message
    print(f"\n{'='*80}")
    print(f"RUNNING {len(strategies)} EXPERIMENTS IN PARALLEL (max {max_parallel} at once)")
    print(f"Results will be saved to: {results_base_dir}")
    print(f"{'='*80}\n")
    
    # Run experiments up to max_parallel at a time
    for strategy in strategies:
        # Wait if we've reached max parallel processes
        while len(active_processes) >= max_parallel:
            # Check which processes have completed
            for i, p in enumerate(list(active_processes)):
                if p.poll() is not None:  # Process has terminated
                    active_processes.remove(p)
            time.sleep(1)  # Don't spin the CPU while waiting
        
        # Create experiment directory
        exp_dir = create_experiment_directory(strategy, config_updates)
        
        # Create specific result directory
        config_str = "_".join([f"{k}{v}" for k, v in config_updates.items()]) if config_updates else ""
        result_dir = os.path.join(results_base_dir, f"{strategy}_{config_str}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Build command with appropriate arguments
        cmd = [sys.executable, 'main_multiprocessing.py']
        
        # Add environment variables for controlling processes
        env = os.environ.copy()
        if num_processes is not None:
            env['NUM_PROCESSES'] = str(num_processes)
        
        # Start the experiment process
        print(f"Starting experiment with strategy: {strategy}")
        process = subprocess.Popen(
            cmd, 
            cwd=exp_dir,
            env=env,
            stdout=open(os.path.join(result_dir, f"{strategy}_output.log"), 'w'),
            stderr=subprocess.STDOUT
        )
        
        # Add to process lists
        all_processes.append(process)
        active_processes.append(process)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=monitor_experiment,
            args=(process, strategy, exp_dir, result_dir)
        )
        monitor_thread.start()
        monitor_threads.append(monitor_thread)
    
    # Wait for all processes to complete
    print(f"\nAll experiments started. Waiting for completion...")
    for thread in monitor_threads:
        thread.join()
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"ALL EXPERIMENTS COMPLETED")
    print(f"Results saved to: {results_base_dir}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Run federated active learning with multiple strategies in parallel')
    parser.add_argument('--strategies', nargs='+', default=["KAFAL", "Entropy", "BADGE", "Random", "FEAL", "LOGO", "Noise"],
                        help='List of strategies to evaluate')
    parser.add_argument('--processes', type=int, default=None,
                        help='Number of processes to use for each experiment')
    parser.add_argument('--max-parallel', type=int, default=2,
                        help='Maximum number of experiments to run in parallel')
    
    # Add arguments for common config parameters
    parser.add_argument('--clients', type=int, help='Number of clients (CLIENTS)')
    parser.add_argument('--budget', type=int, help='Budget for active learning (BUDGET)')
    parser.add_argument('--base', type=int, help='Base number of samples (BASE)')
    parser.add_argument('--epoch', type=int, help='Number of epochs (EPOCH)')
    parser.add_argument('--communication', type=int, help='Number of communication rounds (COMMUNICATION)')
    parser.add_argument('--cycles', type=int, help='Number of active learning cycles (CYCLES)')
    parser.add_argument('--ratio', type=float, help='Ratio of clients selected per round (RATIO)')
    parser.add_argument('--trials', type=int, help='Number of trials (TRIALS)')
    parser.add_argument('--lr', type=float, help='Learning rate (LR)')
    
    args = parser.parse_args()
    
    # Collect config updates from command line arguments
    config_updates = {}
    if args.clients is not None:
        config_updates['CLIENTS'] = args.clients
    if args.budget is not None:
        config_updates['BUDGET'] = args.budget
    if args.base is not None:
        config_updates['BASE'] = args.base
    if args.epoch is not None:
        config_updates['EPOCH'] = args.epoch
    if args.communication is not None:
        config_updates['COMMUNICATION'] = args.communication
    if args.cycles is not None:
        config_updates['CYCLES'] = args.cycles
    if args.ratio is not None:
        config_updates['RATIO'] = args.ratio
    if args.trials is not None:
        config_updates['TRIALS'] = args.trials
    if args.lr is not None:
        config_updates['LR'] = args.lr
    
    # Automatically adjust max_parallel based on available CPU cores
    import multiprocessing as mp
    available_cores = mp.cpu_count()
    
    # Warn if max_parallel is too high
    if args.max_parallel > available_cores // 2:
        print(f"Warning: Running {args.max_parallel} parallel experiments may overload your system.")
        print(f"You have {available_cores} CPU cores. Consider using a lower value for --max-parallel.")
        confirm = input("Continue anyway? (y/n): ")
        if confirm.lower() != 'y':
            print("Experiment canceled.")
            return
    
    # Display configuration
    print(f"Running {len(args.strategies)} experiments with strategies: {', '.join(args.strategies)}")
    print(f"Running up to {args.max_parallel} experiments in parallel")
    print(f"Using {args.processes if args.processes else 'default'} processes per experiment")
    
    if config_updates:
        print("\nConfiguration updates:")
        for key, value in config_updates.items():
            print(f"  {key} = {value}")
    
    # Run parallel experiments
    run_parallel_experiments(
        args.strategies,
        config_updates,
        args.processes,
        args.max_parallel
    )

if __name__ == "__main__":
    main()