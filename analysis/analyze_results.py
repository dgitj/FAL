import json
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from collections import defaultdict

def comprehensive_analysis(log_dir="."):
    # Find all result files in current directory
    result_files = glob.glob(os.path.join(log_dir, "*results.json"))
    
    if not result_files:
        print("No result files found.")
        return

    print(f"Found {len(result_files)} result files: {result_files}")
    
    # Load all experiment data
    experiments = []
    for file_path in result_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract strategy name from filename or JSON
            strategy = data.get('strategy_name', os.path.basename(file_path).split('_')[0])
            experiments.append({
                'strategy': strategy,
                'data': data,
                'filename': os.path.basename(file_path)
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
    
    # Classify strategies
    global_methods = ['KAFAL', 'FEAL', 'LOGO']
    local_methods = ['Entropy', 'BADGE', 'Random']
    
    global_exps = [exp for exp in experiments if exp['strategy'] in global_methods]
    local_exps = [exp for exp in experiments if exp['strategy'] in local_methods]
    
    if not global_exps:
        print("No global-aware method experiments found.")
    
    if not local_exps:
        print("No local-only method experiments found.")
    
    # 1. Accuracy Analysis
    print("\n===== 1. ACCURACY ANALYSIS =====")
    accuracy_data = []
    
    for exp in experiments:
        strategy = exp['strategy']
        data = exp['data']
        
        if 'global_accuracy' not in data:
            print(f"No accuracy data for {strategy}")
            continue
        
        global_acc = {int(k): float(v) for k, v in data['global_accuracy'].items()}
        rounds = sorted(global_acc.keys())
        
        # Get final accuracy
        final_acc = global_acc[max(rounds)]
        print(f"{strategy}: Final accuracy = {final_acc:.2f}%")
        
        # Save progression data
        for r in rounds:
            accuracy_data.append({
                'strategy': strategy,
                'type': 'global-aware' if strategy in global_methods else 'local-only',
                'round': r,
                'accuracy': global_acc[r]
            })
    
    # Create accuracy DataFrame
    acc_df = pd.DataFrame(accuracy_data)
    
    if not acc_df.empty:
        # Plot accuracy curves
        plt.figure(figsize=(12, 8))
        
        for strategy in acc_df['strategy'].unique():
            strategy_data = acc_df[acc_df['strategy'] == strategy]
            strategy_type = 'global-aware' if strategy in global_methods else 'local-only'
            linestyle = '-' if strategy in global_methods else '--'
            plt.plot(strategy_data['round'], strategy_data['accuracy'], 
                     marker='o', linestyle=linestyle, label=f"{strategy} ({strategy_type})")
        
        plt.title("Accuracy Progression Comparison", fontsize=14)
        plt.xlabel("Active Learning Round", fontsize=12)
        plt.ylabel("Accuracy (%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.savefig(os.path.join(results_dir, "accuracy_progression.png"), dpi=300, bbox_inches='tight')
        
        # Compare global vs local methods
        if global_exps and local_exps:
            global_df = acc_df[acc_df['type'] == 'global-aware']
            local_df = acc_df[acc_df['type'] == 'local-only']
            
            # Group by round and calculate mean for each type
            global_avg = global_df.groupby('round')['accuracy'].mean().reset_index()
            local_avg = local_df.groupby('round')['accuracy'].mean().reset_index()
            
            plt.figure(figsize=(12, 8))
            plt.plot(global_avg['round'], global_avg['accuracy'], 'ro-', linewidth=2, label='Global-aware methods (avg)')
            plt.plot(local_avg['round'], local_avg['accuracy'], 'bs-', linewidth=2, label='Local-only methods (avg)')
            
            # Fill area between curves to highlight difference
            plt.fill_between(global_avg['round'], global_avg['accuracy'], local_avg['accuracy'], 
                             where=(local_avg['accuracy'] > global_avg['accuracy']),
                             color='blue', alpha=0.2, label='Local methods advantage')
            plt.fill_between(global_avg['round'], global_avg['accuracy'], local_avg['accuracy'],
                             where=(global_avg['accuracy'] > local_avg['accuracy']),
                             color='red', alpha=0.2, label='Global methods advantage')
            
            plt.title("Global vs Local Methods Comparison", fontsize=14)
            plt.xlabel("Active Learning Round", fontsize=12)
            plt.ylabel("Average Accuracy (%)", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.savefig(os.path.join(results_dir, "global_vs_local_accuracy.png"), dpi=300, bbox_inches='tight')
            
            # Final accuracy comparison
            final_round = acc_df['round'].max()
            final_acc_global = global_df[global_df['round'] == final_round]['accuracy'].mean()
            final_acc_local = local_df[local_df['round'] == final_round]['accuracy'].mean()
            
            print(f"\nFinal round comparison:")
            print(f"Global-aware methods average: {final_acc_global:.2f}%")
            print(f"Local-only methods average: {final_acc_local:.2f}%")
            print(f"Difference: {abs(final_acc_global - final_acc_local):.2f}% ({'Global better' if final_acc_global > final_acc_local else 'Local better'})")
    
    # 2. Model Distance Analysis
    print("\n===== 2. MODEL DISTANCE ANALYSIS =====")
    distance_data = []
    
    for exp in experiments:
        strategy = exp['strategy']
        data = exp['data']
        
        if 'model_distances' not in data:
            print(f"No model distance data for {strategy}")
            continue
        
        # Extract model distances for each round
        for round_key, client_distances in data['model_distances'].items():
            round_num = int(round_key)
            distances = [float(v) for v in client_distances.values()]
            
            # Compute statistics
            avg_distance = np.mean(distances)
            min_distance = np.min(distances)
            max_distance = np.max(distances)
            std_distance = np.std(distances)
            
            distance_data.append({
                'strategy': strategy,
                'type': 'global-aware' if strategy in global_methods else 'local-only',
                'round': round_num,
                'avg_distance': avg_distance,
                'min_distance': min_distance,
                'max_distance': max_distance,
                'std_distance': std_distance
            })
    
    # Create distance DataFrame
    dist_df = pd.DataFrame(distance_data)
    
    if not dist_df.empty:
        # Get last round data for each strategy
        last_round_dist = dist_df.loc[dist_df.groupby('strategy')['round'].idxmax()]
        
        print("Average model distances in final round:")
        for _, row in last_round_dist.sort_values('avg_distance', ascending=False).iterrows():
            print(f"{row['strategy']}: {row['avg_distance']:.4f} (std: {row['std_distance']:.4f})")
        
        # Plot model distance progression
        plt.figure(figsize=(12, 8))
        
        for strategy in dist_df['strategy'].unique():
            strategy_data = dist_df[dist_df['strategy'] == strategy]
            strategy_type = 'global-aware' if strategy in global_methods else 'local-only'
            linestyle = '-' if strategy in global_methods else '--'
            plt.plot(strategy_data['round'], strategy_data['avg_distance'], 
                     marker='o', linestyle=linestyle, label=f"{strategy} ({strategy_type})")
            
            # Add error bars for last point
            last_point = strategy_data.iloc[-1]
            plt.errorbar(last_point['round'], last_point['avg_distance'], 
                         yerr=last_point['std_distance'], fmt='o', capsize=5)
        
        plt.title("Model Distance Progression", fontsize=14)
        plt.xlabel("Active Learning Round", fontsize=12)
        plt.ylabel("Average Model Distance", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.savefig(os.path.join(results_dir, "model_distance_progression.png"), dpi=300, bbox_inches='tight')
        
        # Compare global vs local methods
        if global_exps and local_exps:
            global_dist_df = dist_df[dist_df['type'] == 'global-aware']
            local_dist_df = dist_df[dist_df['type'] == 'local-only']
            
            # Group by round and calculate mean for each type
            global_dist_avg = global_dist_df.groupby('round')['avg_distance'].mean().reset_index()
            local_dist_avg = local_dist_df.groupby('round')['avg_distance'].mean().reset_index()
            
            plt.figure(figsize=(12, 8))
            plt.plot(global_dist_avg['round'], global_dist_avg['avg_distance'], 'ro-', linewidth=2, label='Global-aware methods (avg)')
            plt.plot(local_dist_avg['round'], local_dist_avg['avg_distance'], 'bs-', linewidth=2, label='Local-only methods (avg)')
            
            plt.title("Global vs Local Methods - Model Distance Comparison", fontsize=14)
            plt.xlabel("Active Learning Round", fontsize=12)
            plt.ylabel("Average Model Distance", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=10)
            plt.savefig(os.path.join(results_dir, "global_vs_local_distance.png"), dpi=300, bbox_inches='tight')
    
    # 3. Class Distribution Analysis
    print("\n===== 3. CLASS DISTRIBUTION ANALYSIS =====")
    class_data = []
    
    for exp in experiments:
        strategy = exp['strategy']
        data = exp['data']
        
        if 'sample_classes' not in data:
            print(f"No class distribution data for {strategy}")
            continue
        
        # Extract class distribution for each round
        for round_key, client_classes in data['sample_classes'].items():
            round_num = int(round_key)
            
            # Combine all classes
            all_classes = []
            for client_id, classes in client_classes.items():
                all_classes.extend([int(c) for c in classes])
            
            # Count per class
            class_counts = {}
            for i in range(10):  # Assuming CIFAR10
                class_counts[i] = all_classes.count(i)
            
            total = len(all_classes)
            
            # Calculate imbalance metrics
            class_percentages = [class_counts.get(i, 0)/total*100 if total > 0 else 0 for i in range(10)]
            entropy = -sum([(p/100) * np.log2(p/100) if p > 0 else 0 for p in class_percentages])
            norm_entropy = entropy / np.log2(10)  # Normalized entropy (1 = perfectly balanced)
            
            # Save data for each class
            for class_id in range(10):
                class_data.append({
                    'strategy': strategy,
                    'type': 'global-aware' if strategy in global_methods else 'local-only',
                    'round': round_num,
                    'class': class_id,
                    'count': class_counts.get(class_id, 0),
                    'percentage': class_percentages[class_id],
                    'entropy': entropy,
                    'norm_entropy': norm_entropy
                })
    
    # Create class DataFrame
    class_df = pd.DataFrame(class_data)
    
    if not class_df.empty:
        # Get last round data
        last_round = class_df['round'].max()
        last_round_class = class_df[class_df['round'] == last_round]
        
        # Calculate balance metrics for each strategy
        balance_metrics = last_round_class.groupby('strategy')['norm_entropy'].mean().reset_index()
        print("\nClass balance (normalized entropy, higher = more balanced):")
        for _, row in balance_metrics.sort_values('norm_entropy', ascending=False).iterrows():
            print(f"{row['strategy']}: {row['norm_entropy']:.4f}")
        
        # Plot class distribution for each strategy
        for strategy in class_df['strategy'].unique():
            strategy_data = last_round_class[last_round_class['strategy'] == strategy]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='class', y='percentage', data=strategy_data)
            
            plt.title(f"Class Distribution - {strategy}", fontsize=14)
            plt.xlabel("Class ID", fontsize=12)
            plt.ylabel("Percentage (%)", fontsize=12)
            plt.xticks(range(10))
            plt.savefig(os.path.join(results_dir, f"{strategy}_class_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Compare global vs local methods class distributions
        global_class = last_round_class[last_round_class['type'] == 'global-aware']
        local_class = last_round_class[last_round_class['type'] == 'local-only']
        
        if not global_class.empty and not local_class.empty:
            global_class_avg = global_class.groupby('class')['percentage'].mean().reset_index()
            local_class_avg = local_class.groupby('class')['percentage'].mean().reset_index()
            
            plt.figure(figsize=(12, 8))
            
            # Create grouped bar chart
            bar_width = 0.35
            index = np.arange(10)
            
            plt.bar(index - bar_width/2, global_class_avg['percentage'], bar_width, label='Global-aware methods')
            plt.bar(index + bar_width/2, local_class_avg['percentage'], bar_width, label='Local-only methods')
            
            plt.title("Class Distribution Comparison", fontsize=14)
            plt.xlabel("Class ID", fontsize=12)
            plt.ylabel("Percentage (%)", fontsize=12)
            plt.xticks(index, range(10))
            plt.legend()
            plt.savefig(os.path.join(results_dir, "global_vs_local_class_distribution.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate JS divergence between distributions
            global_dist = global_class_avg['percentage'].values / 100
            local_dist = local_class_avg['percentage'].values / 100
            
            # Normalize to sum to 1
            global_dist = global_dist / global_dist.sum()
            local_dist = local_dist / local_dist.sum()
            
            # Calculate correlation
            corr, _ = pearsonr(global_dist, local_dist)
            print(f"\nCorrelation between global and local class distributions: {corr:.4f}")
            
            # Identify most different classes
            class_diff = abs(global_dist - local_dist)
            diff_classes = np.argsort(-class_diff)[:3]
            print(f"Classes with biggest distribution difference: {list(diff_classes)}")
    
    # 4. Class Accuracy Analysis
    print("\n===== 4. CLASS ACCURACY ANALYSIS =====")
    class_acc_data = []
    
    for exp in experiments:
        strategy = exp['strategy']
        data = exp['data']
        
        if 'class_accuracies' not in data:
            print(f"No class accuracy data for {strategy}")
            continue
        
        # Extract class accuracies for each round
        for round_key, class_accs in data['class_accuracies'].items():
            round_num = int(round_key)
            
            for class_id, accuracy in class_accs.items():
                class_acc_data.append({
                    'strategy': strategy,
                    'type': 'global-aware' if strategy in global_methods else 'local-only',
                    'round': round_num,
                    'class': int(class_id),
                    'accuracy': float(accuracy)
                })
    
    # Create class accuracy DataFrame
    class_acc_df = pd.DataFrame(class_acc_data)
    
    if not class_acc_df.empty:
        # Get last round data
        last_round = class_acc_df['round'].max()
        last_round_acc = class_acc_df[class_acc_df['round'] == last_round]
        
        # Identify worst-performing classes for each strategy
        print("\nWorst-performing classes per strategy:")
        for strategy in last_round_acc['strategy'].unique():
            strategy_data = last_round_acc[last_round_acc['strategy'] == strategy]
            worst_classes = strategy_data.sort_values('accuracy').head(3)
            print(f"{strategy}: Classes {list(worst_classes['class'])} with accuracies {list(worst_classes['accuracy'])}")
        
        # Plot class accuracies for each strategy
        for strategy in class_acc_df['strategy'].unique():
            strategy_last_round = last_round_acc[last_round_acc['strategy'] == strategy]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='class', y='accuracy', data=strategy_last_round)
            
            plt.title(f"Class Accuracies - {strategy}", fontsize=14)
            plt.xlabel("Class ID", fontsize=12)
            plt.ylabel("Accuracy (%)", fontsize=12)
            plt.xticks(range(10))
            plt.savefig(os.path.join(results_dir, f"{strategy}_class_accuracies.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Compare global vs local methods
        if global_exps and local_exps:
            global_acc = last_round_acc[last_round_acc['type'] == 'global-aware']
            local_acc = last_round_acc[last_round_acc['type'] == 'local-only']
            
            if not global_acc.empty and not local_acc.empty:
                global_acc_avg = global_acc.groupby('class')['accuracy'].mean().reset_index()
                local_acc_avg = local_acc.groupby('class')['accuracy'].mean().reset_index()
                
                plt.figure(figsize=(12, 8))
                
                # Create grouped bar chart
                bar_width = 0.35
                index = np.arange(10)
                
                plt.bar(index - bar_width/2, global_acc_avg['accuracy'], bar_width, label='Global-aware methods')
                plt.bar(index + bar_width/2, local_acc_avg['accuracy'], bar_width, label='Local-only methods')
                
                plt.title("Class Accuracy Comparison", fontsize=14)
                plt.xlabel("Class ID", fontsize=12)
                plt.ylabel("Accuracy (%)", fontsize=12)
                plt.xticks(index, range(10))
                plt.legend()
                plt.savefig(os.path.join(results_dir, "global_vs_local_class_accuracies.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Calculate differences
                class_acc_diff = pd.merge(global_acc_avg, local_acc_avg, on='class', suffixes=('_global', '_local'))
                class_acc_diff['diff'] = class_acc_diff['accuracy_global'] - class_acc_diff['accuracy_local']
                
                plt.figure(figsize=(12, 8))
                colors = ['red' if x < 0 else 'green' for x in class_acc_diff['diff']]
                plt.bar(class_acc_diff['class'], class_acc_diff['diff'], color=colors)
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                
                plt.title("Accuracy Difference (Global - Local Methods)", fontsize=14)
                plt.xlabel("Class ID", fontsize=12)
                plt.ylabel("Accuracy Difference (%)", fontsize=12)
                plt.xticks(range(10))
                plt.savefig(os.path.join(results_dir, "global_vs_local_accuracy_diff.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Identify classes where local methods significantly outperform
                local_better = class_acc_diff[class_acc_diff['diff'] < -2]  # 2% threshold
                if not local_better.empty:
                    print("\nClasses where local methods significantly outperform global methods:")
                    for _, row in local_better.iterrows():
                        print(f"Class {int(row['class'])}: Local better by {abs(row['diff']):.2f}% " +
                              f"({row['accuracy_local']:.2f}% vs {row['accuracy_global']:.2f}%)")
    
    # 5. Correlation Analysis
    print("\n===== 5. CORRELATION ANALYSIS =====")
    
    if not class_df.empty and not class_acc_df.empty:
        # Merge class distribution and accuracy data
        last_round = min(class_df['round'].max(), class_acc_df['round'].max())
        
        class_merged = pd.merge(
            class_df[(class_df['round'] == last_round)],
            class_acc_df[(class_acc_df['round'] == last_round)],
            on=['strategy', 'class', 'round'],
            suffixes=('_dist', '_acc')
        )
        
        # Calculate correlation for each strategy
        correlations = []
        for strategy in class_merged['strategy'].unique():
            strategy_data = class_merged[class_merged['strategy'] == strategy]
            
            if len(strategy_data) > 5:  # Need enough data points
                corr, p_value = pearsonr(strategy_data['percentage'], strategy_data['accuracy'])
                correlations.append({
                    'strategy': strategy,
                    'type': 'global-aware' if strategy in global_methods else 'local-only',
                    'correlation': corr,
                    'p_value': p_value
                })
        
        if correlations:
            corr_df = pd.DataFrame(correlations)
            print("Correlation between class distribution and accuracy:")
            for _, row in corr_df.sort_values('correlation').iterrows():
                sig = "significant" if row['p_value'] < 0.05 else "not significant"
                print(f"{row['strategy']}: {row['correlation']:.4f} ({sig})")
            
            # Compare global vs local methods
            global_corr = corr_df[corr_df['type'] == 'global-aware']['correlation'].mean()
            local_corr = corr_df[corr_df['type'] == 'local-only']['correlation'].mean()
            
            print(f"\nAverage correlation - Global methods: {global_corr:.4f}")
            print(f"Average correlation - Local methods: {local_corr:.4f}")
            
            # Visualize correlations
            plt.figure(figsize=(10, 6))
            sns.barplot(x='strategy', y='correlation', hue='type', data=corr_df)
            plt.title("Correlation: Class Distribution vs Accuracy", fontsize=14)
            plt.xlabel("Strategy", fontsize=12)
            plt.ylabel("Correlation Coefficient", fontsize=12)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "distribution_accuracy_correlation.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 6. Conclusion and Analysis
    print("\n===== 6. CONCLUSION & RECOMMENDATIONS =====")
    
    # Key findings
    if global_exps and local_exps:
        issues = []
        
        # Accuracy comparison
        if 'final_acc_global' in locals() and 'final_acc_local' in locals():
            if final_acc_global < final_acc_local:
                diff = final_acc_local - final_acc_global
                issues.append(f"1. Performance gap: Global methods underperform by {diff:.2f}% compared to local methods")
        
        # Model distance analysis
        if 'global_dist_avg' in locals() and 'local_dist_avg' in locals():
            last_global_dist = global_dist_avg.iloc[-1]['avg_distance']
            last_local_dist = local_dist_avg.iloc[-1]['avg_distance']
            
            if last_global_dist > last_local_dist:
                issues.append(f"2. Higher model divergence: Global methods show {(last_global_dist/last_local_dist):.2f}x " +
                              f"greater divergence between local and global models")
        
        # Class distribution difference
        if 'diff_classes' in locals():
            issues.append(f"3. Class distribution imbalance: Global methods sample differently for classes {list(diff_classes)}")
        
        # Class accuracy difference
        if 'local_better' in locals() and not local_better.empty:
            local_better_classes = list(local_better['class'])
            issues.append(f"4. Class-specific weaknesses: Global methods underperform on classes {local_better_classes}")
        
        # Correlation difference
        if 'global_corr' in locals() and 'local_corr' in locals():
            if global_corr < local_corr:
                issues.append(f"5. Sampling strategy mismatch: Global methods show weaker correlation between " +
                              f"class distribution and accuracy ({global_corr:.2f} vs {local_corr:.2f})")
        
        # Print conclusions
        print("Key issues identified with global-aware methods:")
        for issue in issues:
            print(issue)
        
        # Recommendations
        print("\nRecommendations to improve global-aware methods:")
        print("1. Reduce the weight of global information during model updates")
        print("2. Implement gradient-based methods that ensure local and global objectives align")
        print("3. Add class balancing constraints to sampling strategies")
        print("4. Focus more resources on underperforming classes (adaptive sampling)")
        print("5. Consider limiting global information to feature extraction rather than classification")
        
        # Save analysis to file
        with open(os.path.join(results_dir, "analysis_summary.txt"), 'w') as f:
            f.write("===== FEDERATED ACTIVE LEARNING ANALYSIS =====\n\n")
            f.write("Key issues identified with global-aware methods:\n")
            for issue in issues:
                f.write(f"- {issue}\n")
            
            f.write("\nRecommendations to improve global-aware methods:\n")
            f.write("1. Reduce the weight of global information during model updates\n")
            f.write("2. Implement gradient-based methods that ensure local and global objectives align\n")
            f.write("3. Add class balancing constraints to sampling strategies\n")
            f.write("4. Focus more resources on underperforming classes (adaptive sampling)\n")
            f.write("5. Consider limiting global information to feature extraction rather than classification\n")

if __name__ == "__main__":
    comprehensive_analysis()