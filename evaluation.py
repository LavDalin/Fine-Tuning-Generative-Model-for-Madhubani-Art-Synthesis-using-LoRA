"""
Evaluation utilities for analyzing training results and model performance
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from PIL import Image


class TrainingAnalyzer:
    """Analyze training logs and metrics"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.metrics = self._load_metrics()
    
    def _load_metrics(self) -> Dict:
        """Load training metrics from logs"""
        # This assumes you're using tensorboard
        # You may need to adjust based on your logging format
        metrics = {
            'loss': [],
            'epoch': [],
            'step': [],
            'lr': []
        }
        
        # Try to load from JSON if available
        json_path = self.log_dir / "metrics.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                metrics = json.load(f)
        
        return metrics
    
    def plot_training_curve(self, save_path: str = None, window_size: int = 10):
        """Plot training loss curve"""
        if not self.metrics['loss']:
            print("No training metrics found")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw loss
        ax.plot(self.metrics['step'], self.metrics['loss'], 
                alpha=0.3, label='Raw Loss', color='blue')
        
        # Plot smoothed loss
        if len(self.metrics['loss']) > window_size:
            smoothed_loss = np.convolve(
                self.metrics['loss'], 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            smoothed_steps = self.metrics['step'][window_size-1:]
            ax.plot(smoothed_steps, smoothed_loss, 
                   label=f'Smoothed Loss (window={window_size})', 
                   color='red', linewidth=2)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved training curve to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compute_statistics(self, last_n_epochs: int = 10) -> Dict:
        """Compute training statistics"""
        if not self.metrics['loss']:
            return {}
        
        losses = np.array(self.metrics['loss'])
        epochs = np.array(self.metrics['epoch'])
        
        # Overall statistics
        stats = {
            'best_loss': float(np.min(losses)),
            'final_loss': float(losses[-1]),
            'mean_loss': float(np.mean(losses)),
            'std_loss': float(np.std(losses)),
        }
        
        # Last N epochs statistics
        if len(epochs) > 0:
            last_epochs_mask = epochs >= (epochs[-1] - last_n_epochs)
            last_losses = losses[last_epochs_mask]
            
            stats['last_n_epochs'] = {
                'n': last_n_epochs,
                'mean_loss': float(np.mean(last_losses)),
                'std_loss': float(np.std(last_losses)),
                'min_loss': float(np.min(last_losses)),
                'max_loss': float(np.max(last_losses)),
                'coefficient_of_variation': float(np.std(last_losses) / np.mean(last_losses) * 100),
            }
        
        # Loss gap (difference between best and final)
        stats['loss_gap'] = stats['final_loss'] - stats['best_loss']
        
        return stats
    
    def print_summary(self):
        """Print training summary"""
        stats = self.compute_statistics()
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Best Loss:     {stats['best_loss']:.4f}")
        print(f"Final Loss:    {stats['final_loss']:.4f}")
        print(f"Loss Gap:      {stats['loss_gap']:.4f}")
        print(f"Mean Loss:     {stats['mean_loss']:.4f}")
        print(f"Std Loss:      {stats['std_loss']:.4f}")
        
        if 'last_n_epochs' in stats:
            print(f"\nLast {stats['last_n_epochs']['n']} Epochs:")
            print(f"  Mean Loss:   {stats['last_n_epochs']['mean_loss']:.4f}")
            print(f"  Std Loss:    {stats['last_n_epochs']['std_loss']:.4f}")
            print(f"  CV:          {stats['last_n_epochs']['coefficient_of_variation']:.2f}%")
        
        print("="*50 + "\n")


def compare_models(log_dirs: Dict[str, str], save_path: str = None):
    """
    Compare multiple models' training curves
    
    Args:
        log_dirs: Dictionary mapping model names to log directories
        save_path: Path to save comparison plot
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_dirs)))
    
    for (name, log_dir), color in zip(log_dirs.items(), colors):
        analyzer = TrainingAnalyzer(log_dir)
        
        if analyzer.metrics['loss']:
            # Smooth the loss curve
            window_size = max(10, len(analyzer.metrics['loss']) // 50)
            if len(analyzer.metrics['loss']) > window_size:
                smoothed_loss = np.convolve(
                    analyzer.metrics['loss'], 
                    np.ones(window_size)/window_size, 
                    mode='valid'
                )
                smoothed_steps = analyzer.metrics['step'][window_size-1:]
                ax.plot(smoothed_steps, smoothed_loss, 
                       label=name, color=color, linewidth=2)
            else:
                ax.plot(analyzer.metrics['step'], analyzer.metrics['loss'], 
                       label=name, color=color, linewidth=2)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Model Comparison - Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_statistics(log_dirs: Dict[str, str], save_path: str = None):
    """
    Create a comparison table of model statistics
    
    Args:
        log_dirs: Dictionary mapping model names to log directories
        save_path: Path to save the comparison table
    """
    results = []
    
    for name, log_dir in log_dirs.items():
        analyzer = TrainingAnalyzer(log_dir)
        stats = analyzer.compute_statistics()
        
        if stats:
            results.append({
                'Model': name,
                'Best Loss': f"{stats['best_loss']:.4f}",
                'Final Loss': f"{stats['final_loss']:.4f}",
                'Loss Gap': f"{stats['loss_gap']:.4f}",
                'Mean Loss': f"{stats['mean_loss']:.4f}",
                'Std Loss': f"{stats['std_loss']:.4f}",
            })
    
    if not results:
        print("No results to compare")
        return
    
    # Print as table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Header
    headers = list(results[0].keys())
    header_str = " | ".join(f"{h:^15}" for h in headers)
    print(header_str)
    print("-" * len(header_str))
    
    # Data rows
    for result in results:
        row_str = " | ".join(f"{result[h]:^15}" for h in headers)
        print(row_str)
    
    print("="*80 + "\n")
    
    # Save to file if requested
    if save_path:
        with open(save_path, 'w') as f:
            f.write("MODEL COMPARISON\n")
            f.write("="*80 + "\n")
            f.write(header_str + "\n")
            f.write("-" * len(header_str) + "\n")
            for result in results:
                row_str = " | ".join(f"{result[h]:^15}" for h in headers)
                f.write(row_str + "\n")
        print(f"Saved comparison table to {save_path}")


def analyze_dataset_size_impact(
    results_dir: str,
    method: str = "lora",
    rank: int = 32,
    save_dir: str = None
):
    """
    Analyze the impact of dataset size on model performance
    
    Args:
        results_dir: Base directory containing experiment results
        method: Method name (lora, dora, etc.)
        rank: Rank value
        save_dir: Directory to save analysis plots
    """
    results_path = Path(results_dir)
    subset_sizes = [10, 20, 50]
    
    # Collect metrics for each subset size
    metrics_by_size = {}
    
    for size in subset_sizes:
        log_dir = results_path / f"{method}_r{rank}_subset{size}" / "logs"
        if log_dir.exists():
            analyzer = TrainingAnalyzer(log_dir)
            stats = analyzer.compute_statistics()
            if stats:
                metrics_by_size[size] = stats
    
    if not metrics_by_size:
        print(f"No results found for {method} r={rank}")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sizes = sorted(metrics_by_size.keys())
    
    # Plot 1: Best Loss vs Dataset Size
    best_losses = [metrics_by_size[s]['best_loss'] for s in sizes]
    axes[0, 0].plot(sizes, best_losses, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Dataset Size')
    axes[0, 0].set_ylabel('Best Loss')
    axes[0, 0].set_title('Best Loss vs Dataset Size')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Final Loss vs Dataset Size
    final_losses = [metrics_by_size[s]['final_loss'] for s in sizes]
    axes[0, 1].plot(sizes, final_losses, 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_xlabel('Dataset Size')
    axes[0, 1].set_ylabel('Final Loss')
    axes[0, 1].set_title('Final Loss vs Dataset Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Loss Gap vs Dataset Size
    loss_gaps = [metrics_by_size[s]['loss_gap'] for s in sizes]
    axes[1, 0].plot(sizes, loss_gaps, 'o-', linewidth=2, markersize=8, color='red')
    axes[1, 0].set_xlabel('Dataset Size')
    axes[1, 0].set_ylabel('Loss Gap')
    axes[1, 0].set_title('Loss Gap (Final - Best) vs Dataset Size')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Stability (Std) vs Dataset Size
    stds = [metrics_by_size[s]['last_n_epochs']['std_loss'] 
            for s in sizes if 'last_n_epochs' in metrics_by_size[s]]
    if stds:
        axes[1, 1].plot(sizes[:len(stds)], stds, 'o-', linewidth=2, markersize=8, color='green')
        axes[1, 1].set_xlabel('Dataset Size')
        axes[1, 1].set_ylabel('Std Loss (Last 10 Epochs)')
        axes[1, 1].set_title('Training Stability vs Dataset Size')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{method.upper()} r={rank}: Dataset Size Impact', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{method}_r{rank}_dataset_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved analysis to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate training results")
    parser.add_argument("--log_dir", type=str, help="Path to training logs")
    parser.add_argument("--compare", type=str, nargs='+', help="List of log dirs to compare")
    parser.add_argument("--compare_names", type=str, nargs='+', help="Names for comparison")
    parser.add_argument("--output", type=str, help="Output path for plots")
    
    args = parser.parse_args()
    
    if args.log_dir:
        # Analyze single run
        analyzer = TrainingAnalyzer(args.log_dir)
        analyzer.print_summary()
        analyzer.plot_training_curve(save_path=args.output)
    
    elif args.compare:
        # Compare multiple runs
        if args.compare_names and len(args.compare_names) == len(args.compare):
            log_dirs = dict(zip(args.compare_names, args.compare))
        else:
            log_dirs = {f"Model {i+1}": path for i, path in enumerate(args.compare)}
        
        compare_models(log_dirs, save_path=args.output)
        compare_statistics(log_dirs)
