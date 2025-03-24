import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

def analyze_experiment_results(result_files):
    """Analyze experiment results from JSON files."""
    # Load experiment data
    experiments = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            strategy = data.get('strategy_name')
            if strategy:
                experiments.append({
                    'strategy': strategy,
                    'data': data,
                    'file': os.path.basename(file_path)
                })
                print(f"Loaded data for strategy: {strategy}")
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    if not experiments:
        print("No valid experiment data found.")
        return
    
    # Create results directory
    results_dir = "analysis_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Print experiment summary
    print("\n===== EXPERIMENT SUMMARY =====")
    
    # Get common experiment parameters (from first experiment)
    exp_data = experiments[0]['data']
    num_clients = exp_data.get('num_clients')
    alpha = exp_data.get('config', {}).get('ALPHA')
    seed = exp_data.get('config', {}).get('SEED')
    
    print(f"Number of clients: {num_clients}")
    print(f"Dirichlet alpha: {alpha}")
    print(f"Random seed: {seed}")
    print(f"Strategies: {[exp['strategy'] for exp in experiments]}")
    
    # Classify strategies
    global_methods = ['KAFAL', 'FEAL', 'LOGO']
    local_methods = ['Entropy', 'BADGE', 'Random']
    
    # 2. Print accuracy summary
    print("\n===== ACCURACY SUMMARY =====")
    
    # Final accuracy for each strategy
    for exp in sorted(experiments, key=lambda x: get_final_accuracy(x['data']), reverse=True):
        strategy = exp['strategy']
        final_acc = get_final_accuracy(exp['data'])
        strategy_type = "Global-aware" if strategy in global_methods else "Local-only"
        print(f"{strategy} ({strategy_type}): {final_acc:.2f}%")
    
    # Global vs local comparison
    global_exps = [exp for exp in experiments if exp['strategy'] in global_methods]
    local_exps = [exp for exp in experiments if exp['strategy'] in local_methods]
    
    if global_exps and local_exps:
        global_accs = [get_final_accuracy(exp['data']) for exp in global_exps]
        local_accs = [get_final_accuracy(exp['data']) for exp in local_exps]
        
        avg_global = np.mean(global_accs)
        avg_local = np.mean(local_accs)
        
        print(f"\nGlobal-aware methods average: {avg_global:.2f}%")
        print(f"Local-only methods average: {avg_local:.2f}%")
        print(f"Difference: {abs(avg_global - avg_local):.2f}% ({'Global better' if avg_global > avg_local else 'Local better'})")
    
    # 3. Proceed with client distribution analysis
     # Extract client distribution data from experiment data
    print("\n===== CLIENT DISTRIBUTION ANALYSIS =====")
    
    # Get the initial class distribution from the first experiment
    # (should be same for all experiments as they share the same data partition)
    if experiments and 'sample_classes' in experiments[0]['data']:
        # Get initial distribution (cycle 0)
        initial_cycle = "0"
        if initial_cycle in experiments[0]['data']['sample_classes']:
            client_class_dist = {}
            
            # Extract distributions from initial labeled samples
            for client_id, classes in experiments[0]['data']['sample_classes'][initial_cycle].items():
                # Count classes
                class_counts = [0] * 10
                for c in classes:
                    class_idx = int(c)
                    if class_idx < 10:  # Ensure valid class index
                        class_counts[class_idx] += 1
                
                # Convert to percentages
                total = sum(class_counts)
                client_class_dist[client_id] = [count/total*100 if total > 0 else 0 for count in class_counts]
            
            # Plot heatmap of client distributions
            plt.figure(figsize=(14, 10))
            dist_matrix = np.zeros((len(client_class_dist), 10))
            client_ids = sorted(client_class_dist.keys(), key=lambda x: int(x))
            
            for i, client_id in enumerate(client_ids):
                for j, val in enumerate(client_class_dist[client_id]):
                    dist_matrix[i, j] = val
            
            ax = sns.heatmap(dist_matrix, cmap="YlGnBu", annot=True, fmt=".1f", 
                        xticklabels=range(10), yticklabels=client_ids)
            plt.title("Initial Class Distribution by Client (%)")
            plt.xlabel("Class ID")
            plt.ylabel("Client ID")
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "client_class_distribution.png"), bbox_inches='tight')
            
            # Calculate class imbalance across clients
            class_imbalance = np.std(dist_matrix, axis=0)
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(10), class_imbalance)
            
            # Highlight underperforming classes
            underperforming_classes = [2, 5, 8]  # Based on previous analysis
            for i, bar in enumerate(bars):
                if i in underperforming_classes:
                    bar.set_color('red')
            
            plt.title("Class Distribution Variance Across Clients")
            plt.xlabel("Class ID")
            plt.ylabel("Standard Deviation (%)")
            plt.xticks(range(10))
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "class_distribution_variance.png"), bbox_inches='tight')
            
            # Identify classes with highest distribution variance
            high_var_classes = np.argsort(-class_imbalance)[:3]
            print(f"Classes with highest distribution variance across clients: {list(high_var_classes)}")
            
            # Compare with underperforming classes
            print("\nComparison with underperforming classes:")
            print(f"High variance classes: {list(high_var_classes)}")
            print(f"Underperforming classes: {underperforming_classes}")
            
            overlap = set(high_var_classes).intersection(set(underperforming_classes))
            if overlap:
                print(f"Overlap between high variance and underperforming: {list(overlap)}")
                print(f"This suggests heterogeneity in these classes affects global methods negatively")
            
            # For each underperforming class, show its distribution across clients
            for class_id in underperforming_classes:
                class_dist = [dist[class_id] for dist in client_class_dist.values()]
                
                plt.figure(figsize=(10, 6))
                plt.hist(class_dist, bins=10)
                plt.title(f"Distribution of Class {class_id} Across Clients")
                plt.xlabel("Percentage in Client Dataset (%)")
                plt.ylabel("Number of Clients")
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"class_{class_id}_distribution.png"), bbox_inches='tight')
            
            # Class accuracy analysis by strategy
            class_acc_by_strategy = {}
            
            for exp in experiments:
                strategy = exp['strategy']
                data = exp['data']
                
                if 'class_accuracies' not in data:
                    continue
                
                last_round = max(int(r) for r in data['class_accuracies'].keys())
                class_accs = {int(k): float(v) for k, v in data['class_accuracies'][str(last_round)].items()}
                class_acc_by_strategy[strategy] = class_accs
            
            if class_acc_by_strategy:
                # Create dataset relating class variance to performance by strategy type
                corr_data = []
                
                for class_id in range(10):
                    for strategy, accs in class_acc_by_strategy.items():
                        strategy_type = 'Global' if strategy in global_methods else 'Local'
                        if class_id in accs:
                            corr_data.append({
                                'class': class_id,
                                'strategy': strategy,
                                'type': strategy_type,
                                'accuracy': accs[class_id],
                                'variance': class_imbalance[class_id]
                            })
                
                corr_df = pd.DataFrame(corr_data)
                
                # Variance vs accuracy correlation by strategy type
                global_corr_df = corr_df[corr_df['type'] == 'Global']
                local_corr_df = corr_df[corr_df['type'] == 'Local']
                
                if len(global_corr_df) > 3 and len(local_corr_df) > 3:
                    global_corr, global_p = pearsonr(global_corr_df['variance'], global_corr_df['accuracy'])
                    local_corr, local_p = pearsonr(local_corr_df['variance'], local_corr_df['accuracy'])
                    
                    print("\nCorrelation between class distribution variance and accuracy:")
                    print(f"Global methods: r={global_corr:.4f} (p={global_p:.4f})")
                    print(f"Local methods: r={local_corr:.4f} (p={local_p:.4f})")
                    
                    # Plot correlation
                    plt.figure(figsize=(10, 8))
                    sns.scatterplot(data=corr_df, x='variance', y='accuracy', hue='type', style='type', s=100)
                    
                    # Add regression lines
                    sns.regplot(data=global_corr_df, x='variance', y='accuracy', 
                                scatter=False, color='blue')
                    sns.regplot(data=local_corr_df, x='variance', y='accuracy', 
                                scatter=False, color='orange')
                    
                    # Annotate underperforming classes
                    for class_id in underperforming_classes:
                        global_points = global_corr_df[global_corr_df['class'] == class_id]
                        if not global_points.empty:
                            for _, point in global_points.iterrows():
                                plt.text(point['variance'], point['accuracy'], str(class_id), 
                                        fontsize=12, ha='center')
                    
                    plt.title("Class Accuracy vs. Distribution Variance")
                    plt.xlabel("Class Distribution Variance (%)")
                    plt.ylabel("Accuracy (%)")
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, "variance_vs_accuracy.png"), bbox_inches='tight')
                    
                    # Detailed analysis of the impact by class distribution variance
                    if global_corr < 0 and abs(global_corr) > abs(local_corr):
                        print("\nFinding: Global methods show stronger NEGATIVE correlation with class variance")
                        print("This indicates global methods perform WORSE on classes with high distribution variance")
                    elif local_corr < 0 and abs(local_corr) > abs(global_corr):
                        print("\nFinding: Local methods show stronger NEGATIVE correlation with class variance")
                        print("This indicates local methods perform WORSE on classes with high distribution variance")
                        
                # Compare performance on high-variance vs low-variance classes
                high_var = np.argsort(-class_imbalance)[:5]  # Top 5 highest variance classes
                low_var = np.argsort(class_imbalance)[:5]   # Top 5 lowest variance classes
                
                # Calculate average performance for each strategy type
                avg_perf = {}
                for strategy_type in ['Global', 'Local']:
                    # High variance classes
                    high_var_accs = [row['accuracy'] for _, row in corr_df[
                        (corr_df['type'] == strategy_type) & 
                        (corr_df['class'].isin(high_var))
                    ].iterrows()]
                    
                    # Low variance classes
                    low_var_accs = [row['accuracy'] for _, row in corr_df[
                        (corr_df['type'] == strategy_type) & 
                        (corr_df['class'].isin(low_var))
                    ].iterrows()]
                    
                    avg_perf[f"{strategy_type}_high_var"] = np.mean(high_var_accs) if high_var_accs else 0
                    avg_perf[f"{strategy_type}_low_var"] = np.mean(low_var_accs) if low_var_accs else 0
                
                print("\nPerformance on high vs low variance classes:")
                print(f"Global methods on high variance classes: {avg_perf.get('Global_high_var', 0):.2f}%")
                print(f"Global methods on low variance classes: {avg_perf.get('Global_low_var', 0):.2f}%")
                print(f"Difference: {abs(avg_perf.get('Global_high_var', 0) - avg_perf.get('Global_low_var', 0)):.2f}%")
                
                print(f"Local methods on high variance classes: {avg_perf.get('Local_high_var', 0):.2f}%")
                print(f"Local methods on low variance classes: {avg_perf.get('Local_low_var', 0):.2f}%")
                print(f"Difference: {abs(avg_perf.get('Local_high_var', 0) - avg_perf.get('Local_low_var', 0)):.2f}%")
                
                # Compare the differences
                global_gap = avg_perf.get('Global_low_var', 0) - avg_perf.get('Global_high_var', 0)
                local_gap = avg_perf.get('Local_low_var', 0) - avg_perf.get('Local_high_var', 0)
                
                if global_gap > local_gap:
                    print("\nConclusion: Global methods are MORE affected by class distribution variance")
                    print(f"Performance gap: Global methods {global_gap:.2f}% vs Local methods {local_gap:.2f}%")
                else:
                    print("\nConclusion: Local methods are MORE affected by class distribution variance")
                    print(f"Performance gap: Local methods {local_gap:.2f}% vs Global methods {global_gap:.2f}%")

    analyze_gradient_alignment(experiments)
    analyze_knowledge_gaps(experiments)

def get_final_accuracy(data):
    """Extract final accuracy from experiment data."""
    if 'global_accuracy' not in data:
        return 0
        
    global_acc = {int(k): float(v) for k, v in data['global_accuracy'].items()}
    if not global_acc:
        return 0
        
    return global_acc[max(global_acc.keys())]

# Add to your analyze_experiment_results function after the client distribution analysis

def analyze_gradient_alignment(experiments):
    """Analyze gradient alignment between local and global models."""
    print("\n===== GRADIENT ALIGNMENT ANALYSIS =====")

    print("\nGradient alignment refers to the cosine similarity between direction of parameter updates from a local client model and direction of parameter updates from the global (server) model.")
    print("\nValues range from 1: perfect alignment and -1 complete opposite direction")
    print("\nConflict ratio: percentage of model paraemeters where the local and global updates are in opposite directions")

    
    # Check if gradient alignment data is available
    for exp in experiments:
        if 'gradient_alignments' not in exp['data'] or not exp['data']['gradient_alignments']:
            print(f"No gradient alignment data available for {exp['strategy']}")
            continue
        
        strategy = exp['strategy']
        alignment_data = exp['data']['gradient_alignments']
        
        # Get last cycle data
        cycles = sorted([int(c) for c in alignment_data.keys()])
        if not cycles:
            continue
            
        last_cycle = str(max(cycles))
        
        # Calculate average alignment and conflict ratio across clients
        alignments = []
        conflicts = []
        
        for client, metrics in alignment_data[last_cycle].items():
            if 'alignment' in metrics:
                alignments.append(metrics['alignment'])
            if 'conflict_ratio' in metrics:
                conflicts.append(metrics['conflict_ratio'])
        
        avg_alignment = np.mean(alignments) if alignments else 0
        avg_conflict = np.mean(conflicts) if conflicts else 0
        
        print(f"{strategy}: Avg gradient alignment = {avg_alignment:.4f}, Conflict ratio = {avg_conflict:.4f}")
    
    # Compare global vs local methods
    global_methods = ['KAFAL', 'FEAL', 'LOGO']
    local_methods = ['Entropy', 'BADGE', 'Random']
    
    global_alignments = []
    global_conflicts = []
    local_alignments = []
    local_conflicts = []
    
    for exp in experiments:
        if 'gradient_alignments' not in exp['data'] or not exp['data']['gradient_alignments']:
            continue
            
        strategy = exp['strategy']
        alignment_data = exp['data']['gradient_alignments']
        
        # Get last cycle data
        cycles = sorted([int(c) for c in alignment_data.keys()])
        if not cycles:
            continue
            
        last_cycle = str(max(cycles))
        
        # Calculate averages for this strategy
        alignments = []
        conflicts = []
        
        for client, metrics in alignment_data[last_cycle].items():
            if 'alignment' in metrics:
                alignments.append(metrics['alignment'])
            if 'conflict_ratio' in metrics:
                conflicts.append(metrics['conflict_ratio'])
        
        if not alignments:
            continue
            
        avg_alignment = np.mean(alignments)
        avg_conflict = np.mean(conflicts) if conflicts else 0
        
        # Add to appropriate group
        if strategy in global_methods:
            global_alignments.append(avg_alignment)
            global_conflicts.append(avg_conflict)
        elif strategy in local_methods:
            local_alignments.append(avg_alignment)
            local_conflicts.append(avg_conflict)
    
    # Print comparison if data is available
    if global_alignments and local_alignments:
        print("\nGradient alignment comparison:")
        print(f"Global methods: Alignment = {np.mean(global_alignments):.4f}, Conflict = {np.mean(global_conflicts):.4f}")
        print(f"Local methods: Alignment = {np.mean(local_alignments):.4f}, Conflict = {np.mean(local_conflicts):.4f}")
        
        # Determine if there's a meaningful difference
        align_diff = np.mean(global_alignments) - np.mean(local_alignments)
        conflict_diff = np.mean(global_conflicts) - np.mean(local_conflicts)
        
        print("\nFindings:")
        if abs(align_diff) > 0.05:  # Threshold for meaningful difference
            if align_diff > 0:
                print("- Global methods show BETTER gradient alignment with global model")
            else:
                print("- Global methods show WORSE gradient alignment with global model")
        
        if abs(conflict_diff) > 0.05:  # Threshold for meaningful difference
            if conflict_diff > 0:
                print("- Global methods have MORE conflicting gradients")
            else:
                print("- Global methods have FEWER conflicting gradients")

def analyze_knowledge_gaps(experiments):
    """Analyze knowledge gaps between local and global models."""
    print("\n===== KNOWLEDGE GAP ANALYSIS =====")
    
    # Check if knowledge gap data is available
    for exp in experiments:
        if 'knowledge_gaps' not in exp['data'] or not exp['data']['knowledge_gaps']:
            print(f"No knowledge gap data available for {exp['strategy']}")
            continue
        
        strategy = exp['strategy']
        gap_data = exp['data']['knowledge_gaps']
        
        # Get last cycle data
        cycles = sorted([int(c) for c in gap_data.keys()])
        if not cycles:
            continue
            
        last_cycle = str(max(cycles))
        
        # Aggregate knowledge gaps across clients and classes
        class_gaps = {}
        
        for client, class_metrics in gap_data[last_cycle].items():
            for class_id, metrics in class_metrics.items():
                if class_id not in class_gaps:
                    class_gaps[class_id] = []
                
                # Add the gap (global_acc - local_acc)
                if 'gap' in metrics:
                    class_gaps[class_id].append(metrics['gap'])
        
        # Calculate average gap for each class
        avg_gaps = {int(c): np.mean(gaps) for c, gaps in class_gaps.items() if gaps}
        
        # Find classes where global knowledge helps most
        helpful_classes = [c for c, gap in avg_gaps.items() if gap > 2]  # >2% improvement
        harmful_classes = [c for c, gap in avg_gaps.items() if gap < -2]  # >2% degradation
        
        print(f"\n{strategy}:")
        print(f"- Classes where global knowledge helps: {helpful_classes}")
        print(f"- Classes where global knowledge hurts: {harmful_classes}")
        
        # Calculate overall average gap
        all_gaps = [gap for gaps in class_gaps.values() for gap in gaps]
        avg_gap = np.mean(all_gaps) if all_gaps else 0
        print(f"- Average knowledge gap: {avg_gap:.2f}% ({'Global better' if avg_gap > 0 else 'Local better'})")
    
    # Compare different strategies
    global_methods = ['KAFAL', 'FEAL', 'LOGO']
    
    # For each global method, analyze class-specific performance
    for strategy in global_methods:
        exp = next((e for e in experiments if e['strategy'] == strategy), None)
        if not exp or 'knowledge_gaps' not in exp['data'] or not exp['data']['knowledge_gaps']:
            continue
            
        gap_data = exp['data']['knowledge_gaps']
        
        # Get last cycle data
        cycles = sorted([int(c) for c in gap_data.keys()])
        if not cycles:
            continue
            
        last_cycle = str(max(cycles))
        
        # Calculate class-specific gap profiles
        class_profiles = {}
        
        for client, class_metrics in gap_data[last_cycle].items():
            for class_id, metrics in class_metrics.items():
                if class_id not in class_profiles:
                    class_profiles[class_id] = {
                        'gaps': [],
                        'local_acc': [],
                        'global_acc': [],
                        'local_entropy': [],
                        'global_entropy': []
                    }
                
                if 'gap' in metrics:
                    class_profiles[class_id]['gaps'].append(metrics['gap'])
                if 'local_acc' in metrics:
                    class_profiles[class_id]['local_acc'].append(metrics['local_acc'])
                if 'global_acc' in metrics:
                    class_profiles[class_id]['global_acc'].append(metrics['global_acc'])
                if 'local_entropy' in metrics:
                    class_profiles[class_id]['local_entropy'].append(metrics['local_entropy'])
                if 'global_entropy' in metrics:
                    class_profiles[class_id]['global_entropy'].append(metrics['global_entropy'])
        
        # Get average values for each class
        class_avg = {}
        for class_id, profile in class_profiles.items():
            class_avg[int(class_id)] = {
                'gap': np.mean(profile['gaps']) if profile['gaps'] else 0,
                'local_acc': np.mean(profile['local_acc']) if profile['local_acc'] else 0,
                'global_acc': np.mean(profile['global_acc']) if profile['global_acc'] else 0,
                'entropy_diff': (np.mean(profile['global_entropy']) - np.mean(profile['local_entropy'])) 
                                 if profile['global_entropy'] and profile['local_entropy'] else 0
            }
        
        # Identify correlation between entropy difference and gap
        gaps = [metrics['gap'] for metrics in class_avg.values()]
        entropy_diffs = [metrics['entropy_diff'] for metrics in class_avg.values()]
        
        if len(gaps) > 2 and len(entropy_diffs) > 2:
            try:
                corr, p_value = pearsonr(gaps, entropy_diffs)
                print(f"\n{strategy} - Correlation between entropy difference and performance gap:")
                print(f"- Correlation: {corr:.4f} (p={p_value:.4f})")
                
                if abs(corr) > 0.5 and p_value < 0.1:
                    if corr > 0:
                        print("- FINDING: Higher global model uncertainty correlates with BETTER global performance")
                    else:
                        print("- FINDING: Higher global model uncertainty correlates with WORSE global performance")
            except:
                print(f"Could not calculate correlation for {strategy}")


if __name__ == "__main__":
    result_files = glob.glob("_trial*.json")
    
    if not result_files:
        print("No result files found.")
    else:
        analyze_experiment_results(result_files)