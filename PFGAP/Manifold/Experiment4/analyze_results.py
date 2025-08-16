#!/usr/bin/env python3
"""
Results analysis script for the PFGAP vs KNN testing pipeline
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime

def load_results(results_dir="results"):
    """Load all results from the results directory"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Results directory {results_dir} not found!")
        return None
    
    # Load summary
    summary_file = results_dir / "pipeline_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"✓ Loaded pipeline summary")
    else:
        summary = None
        print("⚠ No pipeline summary found")
    
    # Load individual seed results
    seed_results = {}
    for seed_file in results_dir.glob("*_comprehensive_results.json"):
        seed = seed_file.stem.split('_')[0]
        try:
            with open(seed_file, 'r') as f:
                seed_results[seed] = json.load(f)
        except Exception as e:
            print(f"⚠ Error loading {seed_file}: {e}")
    
    print(f"✓ Loaded {len(seed_results)} seed results")
    
    return summary, seed_results

def analyze_knn_performance(seed_results):
    """Analyze KNN performance across different k values"""
    print("\n" + "="*60)
    print("KNN PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Collect all KNN results
    knn_data = []
    for seed, results in seed_results.items():
        if 'knn_results' in results:
            for k, metrics in results['knn_results'].items():
                knn_data.append({
                    'seed': seed,
                    'k': int(k.replace('k', '')),
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1_score'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall']
                })
    
    if not knn_data:
        print("No KNN results found!")
        return None
    
    df = pd.DataFrame(knn_data)
    
    # Summary statistics
    print("\nKNN Performance Summary:")
    print("-" * 40)
    summary_stats = df.groupby('k').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'f1_score': ['mean', 'std', 'min', 'max'],
        'precision': ['mean', 'std', 'min', 'max'],
        'recall': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print(summary_stats)
    
    # Best performing k value
    best_k = df.groupby('k')['accuracy'].mean().idxmax()
    best_accuracy = df.groupby('k')['accuracy'].mean().max()
    print(f"\nBest performing k value: {best_k} (accuracy: {best_accuracy:.4f})")
    
    return df

def analyze_pfgap_vs_knn(seed_results):
    """Compare PFGAP vs KNN performance across all k values"""
    print("\n" + "="*60)
    print("PFGAP vs KNN COMPARISON")
    print("="*60)
    
    comparison_data = []
    
    for seed, results in seed_results.items():
        if 'pfgap_results' in results and 'knn_results' in results:
            # Get PFGAP results for k=5 (default)
            pfgap_acc = results['pfgap_results']['accuracy']
            
            # Get best KNN accuracy for this seed
            knn_accuracies = [metrics['accuracy'] for metrics in results['knn_results'].values()]
            best_knn_acc = max(knn_accuracies) if knn_accuracies else None
            
            if best_knn_acc is not None:
                comparison_data.append({
                    'seed': seed,
                    'pfgap_accuracy': pfgap_acc,
                    'best_knn_accuracy': best_knn_acc,
                    'difference': pfgap_acc - best_knn_acc,
                    'pfgap_better': pfgap_acc > best_knn_acc
                })
    
    if not comparison_data:
        print("No comparison data found!")
        return None
    
    df = pd.DataFrame(comparison_data)
    
    print("\nPerformance Comparison (PFGAP k=5 vs Best KNN):")
    print("-" * 50)
    print(f"PFGAP average accuracy: {df['pfgap_accuracy'].mean():.4f} ± {df['pfgap_accuracy'].std():.4f}")
    print(f"Best KNN average accuracy: {df['best_knn_accuracy'].mean():.4f} ± {df['best_knn_accuracy'].std():.4f}")
    print(f"Average difference (PFGAP - KNN): {df['difference'].mean():.4f} ± {df['difference'].std():.4f}")
    print(f"PFGAP better in {df['pfgap_better'].sum()}/{len(df)} cases ({df['pfgap_better'].mean()*100:.1f}%)")
    
    return df

def analyze_pfgap_vs_knn_all_k(seed_results):
    """Compare PFGAP vs KNN performance across all k values"""
    print("\n" + "="*60)
    print("PFGAP vs KNN COMPARISON ACROSS ALL K VALUES")
    print("="*60)
    
    # Look for comprehensive PFGAP results
    all_k_comparison = []
    
    for seed, results in seed_results.items():
        # Check if we have comprehensive PFGAP results
        pfgap_file = f"{seed}_pfgap_all_k_results.json"
        pfgap_path = Path("results/metrics") / pfgap_file
        
        if pfgap_path.exists():
            try:
                with open(pfgap_path, 'r') as f:
                    pfgap_all_k = json.load(f)
                
                # Compare each k value
                for k in pfgap_all_k.keys():
                    if f'knn_k{k}' in results.get('knn_results', {}):
                        knn_acc = results['knn_results'][f'knn_k{k}']['accuracy']
                        pfgap_acc = pfgap_all_k[k]['accuracy']
                        
                        all_k_comparison.append({
                            'seed': seed,
                            'k': int(k.replace('k', '')),
                            'knn_accuracy': knn_acc,
                            'pfgap_accuracy': pfgap_acc,
                            'difference': pfgap_acc - knn_acc,
                            'pfgap_better': pfgap_acc > knn_acc
                        })
            except Exception as e:
                print(f"Warning: Could not load comprehensive PFGAP results for seed {seed}: {e}")
    
    if not all_k_comparison:
        print("No comprehensive k-value comparison data found!")
        return None
    
    df = pd.DataFrame(all_k_comparison)
    
    print("\nPerformance Comparison by K Value:")
    print("-" * 50)
    
    # Group by k value and show statistics
    k_stats = df.groupby('k').agg({
        'knn_accuracy': ['mean', 'std'],
        'pfgap_accuracy': ['mean', 'std'],
        'difference': ['mean', 'std'],
        'pfgap_better': 'sum'
    }).round(4)
    
    print(k_stats)
    
    # Show which method performs better for each k value
    print("\nPFGAP vs KNN Performance by K Value:")
    print("-" * 40)
    for k in sorted(df['k'].unique()):
        k_data = df[df['k'] == k]
        pfgap_better = k_data['pfgap_better'].sum()
        total = len(k_data)
        pfgap_win_rate = (pfgap_better / total) * 100 if total > 0 else 0
        avg_diff = k_data['difference'].mean()
        print(f"k={k}: PFGAP better in {pfgap_better}/{total} cases ({pfgap_win_rate:.1f}%), avg diff: {avg_diff:+.4f}")
    
    return df

def analyze_outlier_scores(seed_results):
    """Analyze outlier detection results for both PFGAP and KNN methods"""
    print("\n" + "="*60)
    print("OUTLIER DETECTION ANALYSIS")
    print("="*60)
    
    outlier_data = []
    
    for seed, results in seed_results.items():
        if 'outlier_scores' in results and results['outlier_scores']:
            outlier_scores = results['outlier_scores']
            
            # PFGAP outlier scores
            if 'pfgap' in outlier_scores and outlier_scores['pfgap']:
                pfgap_scores = np.array(outlier_scores['pfgap'])
                outlier_data.append({
                    'seed': seed,
                    'method': 'PFGAP',
                    'mean_score': pfgap_scores.mean(),
                    'std_score': pfgap_scores.std(),
                    'min_score': pfgap_scores.min(),
                    'max_score': pfgap_scores.max(),
                    'num_outliers': np.sum(pfgap_scores > 2.0),  # Threshold for outliers
                    'total_samples': len(pfgap_scores)
                })
            
            # KNN outlier scores for each k value
            if 'knn' in outlier_scores and outlier_scores['knn']:
                for k, knn_scores in outlier_scores['knn'].items():
                    if knn_scores is not None:
                        knn_scores = np.array(knn_scores)
                        outlier_data.append({
                            'seed': seed,
                            'method': f'KNN_k{k}',
                            'mean_score': knn_scores.mean(),
                            'std_score': knn_scores.std(),
                            'min_score': knn_scores.min(),
                            'max_score': knn_scores.max(),
                            'num_outliers': np.sum(knn_scores > 2.0),  # Threshold for outliers
                            'total_samples': len(knn_scores)
                        })
    
    if not outlier_data:
        print("No outlier detection results found!")
        return None
    
    df = pd.DataFrame(outlier_data)
    
    print("\nOutlier Score Summary by Method:")
    print("-" * 50)
    
    # Group by method and show statistics
    method_stats = df.groupby('method').agg({
        'mean_score': ['mean', 'std'],
        'std_score': ['mean', 'std'],
        'num_outliers': 'sum',
        'total_samples': 'sum'
    }).round(4)
    
    print(method_stats)
    
    # Calculate outlier rates
    print("\nOutlier Rates by Method:")
    print("-" * 30)
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        total_outliers = method_data['num_outliers'].sum()
        total_samples = method_data['total_samples'].sum()
        outlier_rate = (total_outliers / total_samples) * 100 if total_samples > 0 else 0
        print(f"{method}: {outlier_rate:.2f}% ({total_outliers}/{total_samples})")
    
    return df

def create_visualizations(seed_results, results_dir="results"):
    """Create visualization plots"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    results_dir = Path(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. KNN Performance across k values
    knn_df = analyze_knn_performance(seed_results)
    if knn_df is not None:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        sns.boxplot(data=knn_df, x='k', y='accuracy')
        plt.title('KNN Accuracy by k Value')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        
        plt.subplot(2, 2, 2)
        sns.boxplot(data=knn_df, x='k', y='f1_score')
        plt.title('KNN F1 Score by k Value')
        plt.xlabel('k')
        plt.ylabel('F1 Score')
        
        plt.subplot(2, 2, 3)
        sns.boxplot(data=knn_df, x='k', y='precision')
        plt.title('KNN Precision by k Value')
        plt.xlabel('k')
        plt.ylabel('Precision')
        
        plt.subplot(2, 2, 4)
        sns.boxplot(data=knn_df, x='k', y='recall')
        plt.title('KNN Recall by k Value')
        plt.xlabel('k')
        plt.ylabel('Recall')
        
        plt.tight_layout()
        plt.savefig(plots_dir / "knn_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved KNN performance plots")
    
    # 2. PFGAP vs KNN comparison
    comp_df = analyze_pfgap_vs_knn(seed_results)
    if comp_df is not None:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(comp_df['best_knn_accuracy'], comp_df['pfgap_accuracy'])
        plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
        plt.xlabel('Best KNN Accuracy')
        plt.ylabel('PFGAP Accuracy')
        plt.title('PFGAP vs Best KNN Accuracy')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(comp_df['difference'], bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Accuracy Difference (PFGAP - KNN)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Performance Differences')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "pfgap_vs_knn.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved PFGAP vs KNN comparison plots")
    
    # 3. Outlier scores distribution
    outlier_df = analyze_outlier_scores(seed_results)
    if outlier_df is not None:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(outlier_df['mean_score'], bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Mean Outlier Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Mean Outlier Scores')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        outlier_rates = outlier_df['num_outliers'] / outlier_df['total_samples']
        plt.hist(outlier_rates * 100, bins=10, alpha=0.7, edgecolor='black')
        plt.xlabel('Outlier Rate (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Outlier Rates')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "outlier_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved outlier analysis plots")
    
    print(f"\nAll plots saved to: {plots_dir}")

def generate_report(seed_results, results_dir="results"):
    """Generate a comprehensive analysis report"""
    print("\n" + "="*60)
    print("GENERATING ANALYSIS REPORT")
    print("="*60)
    
    results_dir = Path(results_dir)
    report_file = results_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w') as f:
        f.write("PFGAP vs KNN Testing Pipeline - Analysis Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total seeds analyzed: {len(seed_results)}\n\n")
        
        # KNN Analysis
        knn_df = analyze_knn_performance(seed_results)
        if knn_df is not None:
            f.write("KNN PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(knn_df.groupby('k').agg({
                'accuracy': ['mean', 'std'],
                'f1_score': ['mean', 'std']
            }).round(4).to_string())
            f.write("\n\n")
        
        # PFGAP vs KNN Comparison
        comp_df = analyze_pfgap_vs_knn(seed_results)
        if comp_df is not None:
            f.write("PFGAP vs KNN COMPARISON (k=5 vs Best KNN)\n")
            f.write("-" * 40 + "\n")
            f.write(f"PFGAP average accuracy: {comp_df['pfgap_accuracy'].mean():.4f} ± {comp_df['pfgap_accuracy'].std():.4f}\n")
            f.write(f"Best KNN average accuracy: {comp_df['best_knn_accuracy'].mean():.4f} ± {comp_df['best_knn_accuracy'].std():.4f}\n")
            f.write(f"PFGAP better in {comp_df['pfgap_better'].sum()}/{len(comp_df)} cases\n\n")
        
        # Comprehensive k-value comparison
        comp_all_k_df = analyze_pfgap_vs_knn_all_k(seed_results)
        if comp_all_k_df is not None:
            f.write("PFGAP vs KNN COMPARISON ACROSS ALL K VALUES\n")
            f.write("-" * 40 + "\n")
            for k in sorted(comp_all_k_df['k'].unique()):
                k_data = comp_all_k_df[comp_all_k_df['k'] == k]
                pfgap_better = k_data['pfgap_better'].sum()
                total = len(k_data)
                pfgap_win_rate = (pfgap_better / total) * 100 if total > 0 else 0
                avg_diff = k_data['difference'].mean()
                f.write(f"k={k}: PFGAP better in {pfgap_better}/{total} cases ({pfgap_win_rate:.1f}%), avg diff: {avg_diff:+.4f}\n")
            f.write("\n")
        
        # Outlier Analysis
        outlier_df = analyze_outlier_scores(seed_results)
        if outlier_df is not None:
            f.write("OUTLIER DETECTION SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average outlier rate: {(outlier_df['num_outliers'] / outlier_df['total_samples']).mean()*100:.2f}%\n")
            f.write(f"Average mean score: {outlier_df['mean_score'].mean():.4f} ± {outlier_df['mean_score'].std():.4f}\n\n")
    
    print(f"✓ Analysis report saved to: {report_file}")

def main():
    """Main analysis function"""
    print("PFGAP vs KNN Results Analysis")
    print("=" * 50)
    
    # Load results
    summary, seed_results = load_results()
    
    if not seed_results:
        print("No results found to analyze!")
        return
    
    # Run analyses
    analyze_knn_performance(seed_results)
    analyze_pfgap_vs_knn(seed_results)
    analyze_pfgap_vs_knn_all_k(seed_results)  # New comprehensive k-value comparison
    analyze_outlier_scores(seed_results)
    
    # Create visualizations
    create_visualizations(seed_results)
    
    # Generate report
    generate_report(seed_results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
