#!/usr/bin/env python3
"""Script to visualize RoboCasa task results from results.json file."""

import argparse
import json
import logging
import pathlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from collections import defaultdict
import numpy as np


def load_results(results_file: pathlib.Path) -> List[Dict]:
    """Load results from JSON file."""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        if not isinstance(results, list):
            raise ValueError("Results file should contain a list of results")
        return results
    except FileNotFoundError:
        logging.error(f"Results file not found: {results_file}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format: {e}")
        return []


def create_success_rate_bar_chart(results: List[Dict], output_dir: pathlib.Path) -> None:
    """Create a bar chart showing success rates by task."""
    if not results:
        logging.warning("No results to plot")
        return
    
    # Calculate average success rate per task
    task_stats = defaultdict(list)
    for result in results:
        task_stats[result['task_name']].append(result['success_rate'])
    
    tasks = list(task_stats.keys())
    avg_success_rates = [np.mean(rates) for rates in task_stats.values()]
    std_success_rates = [np.std(rates) if len(rates) > 1 else 0 for rates in task_stats.values()]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    bars = plt.bar(tasks, avg_success_rates, yerr=std_success_rates, 
                   capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
    
    # Customize plot
    plt.title('Average Success Rate by Task', fontsize=16, fontweight='bold')
    plt.xlabel('Task Name', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, avg_success_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate_by_task.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved success rate bar chart to {output_dir / 'success_rate_by_task.png'}")


def create_timeline_chart(results: List[Dict], output_dir: pathlib.Path) -> None:
    """Create a timeline chart showing success rates over time."""
    if not results:
        logging.warning("No results to plot")
        return
    
    # Convert timestamps to datetime objects
    timestamps = []
    success_rates = []
    task_names = []
    
    for result in results:
        try:
            timestamp = datetime.fromisoformat(result['timestamp'])
            timestamps.append(timestamp)
            success_rates.append(result['success_rate'])
            task_names.append(result['task_name'])
        except (ValueError, KeyError) as e:
            logging.warning(f"Skipping invalid result: {e}")
            continue
    
    if not timestamps:
        logging.warning("No valid timestamps found")
        return
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create scatter plot with different colors for different tasks
    unique_tasks = list(set(task_names))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_tasks)))
    
    for task, color in zip(unique_tasks, colors):
        task_indices = [i for i, t in enumerate(task_names) if t == task]
        task_timestamps = [timestamps[i] for i in task_indices]
        task_rates = [success_rates[i] for i in task_indices]
        
        plt.scatter(task_timestamps, task_rates, label=task, color=color, 
                   alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    # Customize plot
    plt.title('Success Rate Timeline', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
    plt.xticks(rotation=45)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved timeline chart to {output_dir / 'success_rate_timeline.png'}")


def create_detailed_comparison_chart(results: List[Dict], output_dir: pathlib.Path) -> None:
    """Create a detailed comparison chart with multiple runs per task."""
    if not results:
        logging.warning("No results to plot")
        return
    
    # Group results by task
    task_groups = defaultdict(list)
    for result in results:
        task_groups[result['task_name']].append(result)
    
    # Create subplots
    n_tasks = len(task_groups)
    cols = min(3, n_tasks)
    rows = (n_tasks + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_tasks == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten() if n_tasks > 1 else axes
    
    for idx, (task_name, task_results) in enumerate(task_groups.items()):
        if idx >= len(axes_flat):
            break
            
        ax = axes_flat[idx]
        
        # Plot individual runs
        run_numbers = list(range(1, len(task_results) + 1))
        success_rates = [r['success_rate'] for r in task_results]
        
        ax.plot(run_numbers, success_rates, 'o-', linewidth=2, markersize=8, 
                color='steelblue', markerfacecolor='lightblue', markeredgecolor='navy')
        
        # Add average line
        avg_rate = np.mean(success_rates)
        ax.axhline(y=avg_rate, color='red', linestyle='--', linewidth=2, 
                  label=f'Average: {avg_rate:.2%}')
        
        ax.set_title(f'{task_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Run Number')
        ax.set_ylabel('Success Rate')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add value labels
        for x, y in zip(run_numbers, success_rates):
            ax.annotate(f'{y:.2%}', (x, y), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_tasks, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'detailed_task_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved detailed comparison chart to {output_dir / 'detailed_task_comparison.png'}")


def create_heatmap(results: List[Dict], output_dir: pathlib.Path) -> None:
    """Create a heatmap showing success rates across tasks and time periods."""
    if not results:
        logging.warning("No results to plot")
        return
    
    # Convert to DataFrame for easier manipulation
    df_data = []
    for result in results:
        try:
            timestamp = datetime.fromisoformat(result['timestamp'])
            df_data.append({
                'task_name': result['task_name'],
                'success_rate': result['success_rate'],
                'hour': timestamp.hour,
                'date': timestamp.date()
            })
        except (ValueError, KeyError):
            continue
    
    if not df_data:
        logging.warning("No valid data for heatmap")
        return
    
    df = pd.DataFrame(df_data)
    
    # Create pivot table for heatmap
    if len(df['date'].unique()) > 1:
        # If multiple dates, group by date
        pivot_table = df.groupby(['task_name', 'date'])['success_rate'].mean().unstack(fill_value=0)
    else:
        # If single date, group by hour
        pivot_table = df.groupby(['task_name', 'hour'])['success_rate'].mean().unstack(fill_value=0)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', 
                vmin=0, vmax=1, fmt='.2f', cbar_kws={'label': 'Success Rate'})
    
    if len(df['date'].unique()) > 1:
        plt.title('Success Rate Heatmap by Task and Date', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
    else:
        plt.title('Success Rate Heatmap by Task and Hour', fontsize=16, fontweight='bold')
        plt.xlabel('Hour of Day')
    
    plt.ylabel('Task Name')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved heatmap to {output_dir / 'success_rate_heatmap.png'}")


def create_summary_statistics(results: List[Dict], output_dir: pathlib.Path) -> None:
    """Create a summary statistics table and save as image."""
    if not results:
        logging.warning("No results to create summary")
        return
    
    # Calculate statistics per task
    task_stats = defaultdict(list)
    for result in results:
        task_stats[result['task_name']].append(result['success_rate'])
    
    # Create summary DataFrame
    summary_data = []
    for task_name, rates in task_stats.items():
        summary_data.append({
            'Task': task_name,
            'Runs': len(rates),
            'Mean Success Rate': f"{np.mean(rates):.2%}",
            'Std Dev': f"{np.std(rates):.3f}",
            'Min': f"{np.min(rates):.2%}",
            'Max': f"{np.max(rates):.2%}",
            'Last Run': f"{rates[-1]:.2%}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.sort_values('Mean Success Rate', ascending=False)
    
    # Create figure for the table
    fig, ax = plt.subplots(figsize=(14, max(6, len(summary_data) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df_summary.values,
                    colLabels=df_summary.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor('#f8f9fa' if i % 2 == 0 else '#ffffff')
    
    plt.title('Task Performance Summary Statistics', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as CSV
    df_summary.to_csv(output_dir / 'summary_statistics.csv', index=False)
    
    logging.info(f"Saved summary statistics to {output_dir / 'summary_statistics.png'} and .csv")


def main():
    """Main function to create all visualizations."""
    parser = argparse.ArgumentParser(description='Visualize RoboCasa task results')
    parser.add_argument('--results-file', '-r', type=pathlib.Path, 
                       default='data/robocasa/results.json',
                       help='Path to results JSON file')
    parser.add_argument('--output-dir', '-o', type=pathlib.Path,
                       default='data/robocasa/charts',
                       help='Output directory for charts')
    parser.add_argument('--chart-types', '-c', nargs='+',
                       choices=['bar', 'timeline', 'comparison', 'heatmap', 'summary', 'all'],
                       default=['all'],
                       help='Types of charts to generate')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    logging.info(f"Loading results from {args.results_file}")
    results = load_results(args.results_file)
    
    if not results:
        logging.error("No results found to visualize")
        return
    
    logging.info(f"Found {len(results)} results for {len(set(r['task_name'] for r in results))} unique tasks")
    
    # Set matplotlib style
    plt.style.use('seaborn-v0_8')
    
    # Generate charts based on user selection
    chart_types = args.chart_types
    if 'all' in chart_types:
        chart_types = ['bar', 'timeline', 'comparison', 'heatmap', 'summary']
    
    if 'bar' in chart_types:
        create_success_rate_bar_chart(results, args.output_dir)
    
    if 'timeline' in chart_types:
        create_timeline_chart(results, args.output_dir)
    
    if 'comparison' in chart_types:
        create_detailed_comparison_chart(results, args.output_dir)
    
    if 'heatmap' in chart_types:
        create_heatmap(results, args.output_dir)
    
    if 'summary' in chart_types:
        create_summary_statistics(results, args.output_dir)
    
    logging.info(f"All charts saved to {args.output_dir}")


if __name__ == "__main__":
    main()
