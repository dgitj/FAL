import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

def analyze_experiment_results(result_files):
    experiments = []
    for file_path in result_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        strategy = data.get('strategy_name')
        if strategy:
            experiments.append({
                'strategy': strategy,
                'data': data,
                'file': os.path.basename(file_path)
            })
    
    results_dir = "analysis_results"
    os.makedirs(results_dir, exist_ok=True)

    exp_data = experiments[0]['data']
    num_clients = exp_data.get('num_clients')
    print(f"Number of clients: {num_clients}")

    # Extract initial class distribution
    initial_cycle = "0"
    client_class_dist = {}
    if 'sample_classes' in exp_data:
        for client_id, classes in exp_data['sample_classes'][initial_cycle].items():
            class_counts = [0] * 10
            for c in classes:
                class_idx = int(c)
                if class_idx < 10:
                    class_counts[class_idx] += 1
            total = sum(class_counts)
            client_class_dist[client_id] = [count/total*100 if total > 0 else 0 for count in class_counts]

    dist_matrix = np.array([client_class_dist[cid] for cid in sorted(client_class_dist.keys(), key=int)])
    class_imbalance = np.std(dist_matrix, axis=0)

    # Extract class accuracies from last round
    class_acc_by_strategy = {}
    for exp in experiments:
        data = exp['data']
        if 'class_accuracies' not in data:
            continue
        last_round = max(int(r) for r in data['class_accuracies'].keys())
        class_accs = {int(k): float(v) for k, v in data['class_accuracies'][str(last_round)].items()}
        class_acc_by_strategy[exp['strategy']] = class_accs

    # Prepare data for detailed plot
    viz_data = []
    for strategy, acc_dict in class_acc_by_strategy.items():
        for class_id in range(10):
            viz_data.append({
                'Class ID': class_id,
                'Strategy': strategy,
                'Accuracy': acc_dict.get(class_id, np.nan),
                'Std Dev': class_imbalance[class_id]
            })

    viz_df = pd.DataFrame(viz_data).dropna()

    # Detailed scatter plot with regression lines
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=viz_df, x='Std Dev', y='Accuracy', hue='Strategy', style='Strategy', s=120)

    for strategy in viz_df['Strategy'].unique():
        sns.regplot(
            data=viz_df[viz_df['Strategy'] == strategy],
            x='Std Dev', y='Accuracy', scatter=False,
            label=f'{strategy} Trend'
        )

    plt.title("Class Accuracy vs. Standard Deviation of Client Distribution")
    plt.xlabel("Standard Deviation of Class Distribution Across Clients (%)")
    plt.ylabel("Class Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "stddev_vs_accuracy_detailed.png"), bbox_inches='tight')
    plt.show()

    # Correlation analysis
    print("\nDetailed Correlation Analysis (Standard Deviation vs Accuracy):\n")
    for strategy in viz_df['Strategy'].unique():
        subset = viz_df[viz_df['Strategy'] == strategy]
        if len(subset) > 2:
            corr, p_value = pearsonr(subset['Std Dev'], subset['Accuracy'])
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            direction = "negative" if corr < 0 else "positive"
            print(f"Strategy: {strategy}")
            print(f"  Correlation: {corr:.4f}, p-value: {p_value:.4f}")
            print(f"  Interpretation: {significance} {direction} correlation\n")

if __name__ == "__main__":
    result_files = glob.glob("*_trial*.json")
    if not result_files:
        print("No result files found.")
    else:
        analyze_experiment_results(result_files)
