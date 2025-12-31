"""
Compare results from all 4 ablation study scenarios.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_results(result_dir):
    """Load test results from a scenario."""
    results_path = os.path.join(result_dir, 'test_results.json')

    if not os.path.exists(results_path):
        print(f"Warning: Results not found at {results_path}")
        return None

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results


def create_comparison_table(scenarios_results):
    """Create comparison table across all scenarios."""
    data = []

    for scenario_name, results in scenarios_results.items():
        if results is None:
            continue

        data.append({
            'Scenario': scenario_name,
            'Test Accuracy (%)': f"{results['test_accuracy']:.2f}",
            'Macro Avg Precision (%)': f"{results['classification_report']['macro avg']['precision']*100:.2f}",
            'Macro Avg Recall (%)': f"{results['classification_report']['macro avg']['recall']*100:.2f}",
            'Macro Avg F1 (%)': f"{results['classification_report']['macro avg']['f1-score']*100:.2f}",
        })

    df = pd.DataFrame(data)
    return df


def plot_accuracy_comparison(scenarios_results, save_path):
    """Plot bar chart comparing test accuracy across scenarios."""
    scenario_names = []
    accuracies = []

    for scenario_name, results in scenarios_results.items():
        if results is None:
            continue
        scenario_names.append(scenario_name)
        accuracies.append(results['test_accuracy'])

    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(scenario_names, accuracies, color=colors[:len(scenario_names)])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Test Accuracy Comparison Across Scenarios', fontsize=14, fontweight='bold')
    plt.ylim([0, 100])
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy comparison saved to: {save_path}")


def plot_perclass_heatmap(scenarios_results, save_path):
    """Plot heatmap of per-class accuracy across scenarios."""
    # Get class names from first available results
    class_names = None
    for results in scenarios_results.values():
        if results is not None:
            class_names = results['class_names']
            break

    if class_names is None:
        print("No class names found!")
        return

    # Build matrix: rows=classes, cols=scenarios
    data = []
    scenario_names = []

    for scenario_name, results in scenarios_results.items():
        if results is None:
            continue

        scenario_names.append(scenario_name)
        class_accs = []

        for class_name in class_names:
            if class_name in results['classification_report']:
                recall = results['classification_report'][class_name]['recall'] * 100
                class_accs.append(recall)
            else:
                class_accs.append(0.0)

        data.append(class_accs)

    # Transpose so classes are rows
    data = np.array(data).T

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(data, annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=scenario_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Recall (%)'})
    plt.ylabel('Class', fontsize=12)
    plt.xlabel('Scenario', fontsize=12)
    plt.title('Per-Class Recall Across Scenarios', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class heatmap saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare ablation study results')
    parser.add_argument('--scenario1_dir', type=str,
                        default='/workspace/ddim/results/scenario1_synth_only/eval_on_real',
                        help='Path to scenario 1 results')
    parser.add_argument('--scenario2_dir', type=str,
                        default='/workspace/ddim/results/scenario2_real_only/eval',
                        help='Path to scenario 2 results')
    parser.add_argument('--scenario3_dir', type=str,
                        default='/workspace/ddim/results/scenario3_real_balanced/eval',
                        help='Path to scenario 3 results')
    parser.add_argument('--scenario4_dir', type=str,
                        default='/workspace/ddim/results/scenario4_hybrid/eval',
                        help='Path to scenario 4 results')
    parser.add_argument('--output_dir', type=str,
                        default='/workspace/ddim/results/comparison',
                        help='Directory to save comparison results')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all results
    print("Loading results from all scenarios...")
    scenarios_results = {
        'Scenario 1\n(Synth Only)': load_results(args.scenario1_dir),
        'Scenario 2\n(Real Only)': load_results(args.scenario2_dir),
        'Scenario 3\n(Real Balanced)': load_results(args.scenario3_dir),
        'Scenario 4\n(Hybrid)': load_results(args.scenario4_dir),
    }

    # Filter out None results
    available_scenarios = {k: v for k, v in scenarios_results.items() if v is not None}

    if not available_scenarios:
        print("Error: No results found!")
        return

    print(f"Found results for {len(available_scenarios)} scenarios")

    # Create comparison table
    print("\nCreating comparison table...")
    comparison_df = create_comparison_table(available_scenarios)
    print("\n" + "="*100)
    print("COMPARISON TABLE")
    print("="*100)
    print(comparison_df.to_string(index=False))
    print("="*100)

    # Save table
    table_path = os.path.join(args.output_dir, 'comparison_table.csv')
    comparison_df.to_csv(table_path, index=False)
    print(f"\nTable saved to: {table_path}")

    # Plot accuracy comparison
    print("\nCreating accuracy comparison plot...")
    acc_plot_path = os.path.join(args.output_dir, 'accuracy_comparison.png')
    plot_accuracy_comparison(available_scenarios, acc_plot_path)

    # Plot per-class heatmap
    print("\nCreating per-class accuracy heatmap...")
    heatmap_path = os.path.join(args.output_dir, 'perclass_heatmap.png')
    plot_perclass_heatmap(available_scenarios, heatmap_path)

    print(f"\n✓ Comparison complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
