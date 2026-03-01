#!/usr/bin/env python3
"""
Training Log Visualization Script
Parses training log files and creates visualizations and reports.
"""

import argparse
import json
import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

sns.set_style("whitegrid")


def find_log_files(paths: List[str]) -> List[Path]:
    """Recursively find all log files (log-* or *.out) in the provided paths."""
    log_files = []
    for path in paths:
        p = Path(path)
        if p.is_file() and (p.suffix == '.out' or p.name.startswith('log-')):
            log_files.append(p)
        elif p.is_dir():
            log_files.extend(p.rglob('log-*'))
            log_files.extend(p.rglob('*.out'))
    return sorted(set(log_files))


def parse_mllog_line(line: str) -> Optional[Dict]:
    """Parse a single MLLOG JSON line."""
    if ':::MLLOG' not in line:
        return None
    try:
        json_str = line.split(':::MLLOG', 1)[1].strip()
        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return None


def extract_config(log_file: Path) -> Dict:
    """Extract configuration parameters from log file."""
    config = {
        'file': log_file.name,
        'path': str(log_file),
        'lr': None,
        'warmup_steps': None,
        'batch_size': None,
        'seed': None,
        'min_lr': None,
        'lr_decay_steps': None,
        'max_steps': None,
        'moe_router_force_load_balancing': None,
    }

    with open(log_file, 'r') as f:
        for line in f:
            # Parse MLLOG lines
            data = parse_mllog_line(line)
            if data:
                key = data.get('key')
                value = data.get('value')

                if key == 'opt_base_learning_rate':
                    config['lr'] = value
                elif key == 'opt_learning_rate_warmup_steps':
                    config['warmup_steps'] = value
                elif key == 'global_batch_size':
                    config['batch_size'] = value
                elif key == 'seed':
                    config['seed'] = value
                elif key == 'opt_end_learning_rate':
                    config['min_lr'] = value
                elif key == 'opt_learning_rate_decay_steps':
                    config['lr_decay_steps'] = value
                elif key == 'max_steps':
                    config['max_steps'] = value

            # Parse plain text config lines
            if 'moe_router_force_load_balancing:' in line:
                # Format: "  0:   moe_router_force_load_balancing: false"
                parts = line.split('moe_router_force_load_balancing:')
                if len(parts) == 2:
                    value_str = parts[1].strip().lower()
                    config['moe_router_force_load_balancing'] = (value_str == 'true')

    return config


def extract_metrics(log_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Extract training and validation metrics from log file. Returns batch_size as well."""
    train_data = []
    val_data = []
    eval_time_data = []
    batch_size = None

    with open(log_file, 'r') as f:
        for line in f:
            data = parse_mllog_line(line)
            if not data:
                continue

            key = data.get('key')
            value = data.get('value')
            metadata = data.get('metadata', {})

            if key == 'global_batch_size':
                batch_size = value
            elif key == 'tracked_stats' and isinstance(value, dict):
                if 'validation_time' in value:
                    step = metadata.get('step', 0)
                    eval_time_data.append({
                        'iteration': step,
                        'validation_time': value.get('validation_time'),
                    })
                else:
                    samples = metadata.get('samples_count', 0)
                    train_data.append({
                        'samples': samples,
                        'train_step_time': value.get('train_step_time'),
                        'train_loss': value.get('reduced_train_loss'),
                        'slb_loss': value.get('seq_load_balancing_loss'),
                    })
            elif key == 'eval_accuracy':
                samples = metadata.get('samples_count', 0)
                val_data.append({
                    'samples': samples,
                    'val_loss': value,
                })

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    # Filter out invalid training entries (those with NaN loss or 0/invalid samples)
    if not train_df.empty:
        train_df = train_df.dropna(subset=['train_loss'])
        train_df = train_df[train_df['samples'] > 0].reset_index(drop=True)
        # Use sequential iteration numbers for training
        train_df['iteration'] = range(1, len(train_df) + 1)

    # For validation: calculate iteration based on samples and batch size
    if batch_size and not val_df.empty:
        # Validation iteration = samples / batch_size (rounded to nearest int)
        val_df['iteration'] = (val_df['samples'] / batch_size).round().astype(int)
    elif not val_df.empty:
        val_df['iteration'] = range(1, len(val_df) + 1)

    # Merge validation time data into val_df
    if eval_time_data:
        eval_time_df = pd.DataFrame(eval_time_data)
        if not val_df.empty:
            val_df = val_df.merge(eval_time_df, on='iteration', how='left')
        else:
            val_df = eval_time_df

    return train_df, val_df, batch_size


def find_convergence_step(val_df: pd.DataFrame, target_loss: float = 2.55) -> Optional[int]:
    """Find the iteration where validation loss first reaches target."""
    if val_df.empty:
        return None

    converged = val_df[val_df['val_loss'] <= target_loss]
    if converged.empty:
        return None

    return converged.iloc[0]['iteration']


def create_summary_csv(log_files: List[Path], output_path: str, target_loss: float = 2.55):
    """Create CSV summary with key metrics for each log file."""
    summary_data = []

    for log_file in log_files:
        print(f"Processing {log_file.name} for summary...")
        config = extract_config(log_file)
        train_df, val_df, batch_size = extract_metrics(log_file)

        convergence_step = find_convergence_step(val_df, target_loss)

        # Calculate samples to convergence
        samples_to_convergence = None
        if convergence_step is not None and config['batch_size'] is not None:
            samples_to_convergence = convergence_step * config['batch_size']

        summary_data.append({
            'file': config['file'],
            'lr': config['lr'],
            'warmup_steps': config['warmup_steps'],
            'batch_size': config['batch_size'],
            'seed': config['seed'],
            'moe_router_force_load_balancing': config['moe_router_force_load_balancing'],
            'target_loss': target_loss,
            'steps_to_convergence': convergence_step,
            'samples_to_convergence': samples_to_convergence,
            'final_train_loss': train_df['train_loss'].iloc[-1] if not train_df.empty else None,
            'final_val_loss': val_df['val_loss'].iloc[-1] if not val_df.empty else None,
            'total_train_steps': len(train_df),
            'total_val_steps': len(val_df),
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    print(f"Summary CSV saved to {output_path}")

    # Also save as Excel
    excel_path = output_path.replace('.csv', '.xlsx')
    summary_df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"Summary Excel saved to {excel_path}")


def create_excel_report(log_files: List[Path], output_path: str):
    """Create Excel file with detailed metrics for each log file."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for log_file in log_files:
            print(f"Processing {log_file.name} for Excel...")
            config = extract_config(log_file)
            train_df, val_df, batch_size = extract_metrics(log_file)

            # Merge train and val data on iteration
            if not train_df.empty and not val_df.empty:
                # Create a combined dataframe
                combined_df = train_df.copy()
                # Merge validation data (include all val columns: val_loss, validation_time, etc.)
                val_merge_cols = [c for c in val_df.columns if c == 'iteration' or c not in combined_df.columns]
                combined_df = combined_df.merge(
                    val_df[val_merge_cols],
                    on='iteration',
                    how='left'
                )
            elif not train_df.empty:
                combined_df = train_df
            else:
                continue

            # Create a safe sheet name (Excel has 31 char limit)
            sheet_name = log_file.stem[:31]
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Excel report saved to {output_path}")


def create_visualizations(log_files: List[Path], output_dir: str, target_loss: float = 2.55, include_filename: bool = False):
    """Create visualization plots for all log files."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect all data
    all_train_data = []
    all_val_data = []

    for log_file in log_files:
        print(f"Processing {log_file.name} for visualization...")
        config = extract_config(log_file)
        train_df, val_df, batch_size = extract_metrics(log_file)

        if not train_df.empty:
            lr = config['lr']
            warmup = config['warmup_steps']
            gbs = config['batch_size']
            seed = config['seed']
            force_lb = config['moe_router_force_load_balancing']
            label = f"(lr={lr}, warmup_steps={warmup}, GBS={gbs}, seed={seed}, force_lb={force_lb}"
            if include_filename:
                label += f", filename={log_file.name}"
            label += ")"
            train_df['label'] = label
            train_df['lr'] = lr
            train_df['warmup'] = warmup
            train_df['gbs'] = gbs
            train_df['seed'] = seed
            train_df['force_lb'] = force_lb
            all_train_data.append(train_df)

        if not val_df.empty:
            lr = config['lr']
            warmup = config['warmup_steps']
            gbs = config['batch_size']
            seed = config['seed']
            force_lb = config['moe_router_force_load_balancing']
            label = f"(lr={lr}, warmup_steps={warmup}, GBS={gbs}, seed={seed}, force_lb={force_lb}"
            if include_filename:
                label += f", filename={log_file.name}"
            label += ")"
            val_df['label'] = label
            val_df['lr'] = lr
            val_df['warmup'] = warmup
            val_df['gbs'] = gbs
            val_df['seed'] = seed
            val_df['force_lb'] = force_lb
            all_val_data.append(val_df)

    if not all_train_data:
        print("No training data found!")
        return

    train_combined = pd.concat(all_train_data, ignore_index=True)
    val_combined = pd.concat(all_val_data, ignore_index=True) if all_val_data else pd.DataFrame()

    # Plot 1: Iteration vs Training Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in train_combined['label'].unique():
        data = train_combined[train_combined['label'] == label]
        linestyle = '--' if data['force_lb'].iloc[0] else '-'
        ax.plot(data['iteration'], data['train_loss'], label=label, alpha=0.7, linestyle=linestyle)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Iteration')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iter_train_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Iteration vs Validation Loss (with optional target line)
    if not val_combined.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        for label in val_combined['label'].unique():
            data = val_combined[val_combined['label'] == label]
            linestyle = '--' if data['force_lb'].iloc[0] else '-'
            ax.plot(data['iteration'], data['val_loss'], label=label, marker='o', alpha=0.7, linestyle=linestyle)
        if target_loss is not None:
            ax.axhline(y=target_loss, color='r', linestyle='--', linewidth=2, label=f'Target Loss ({target_loss})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Validation Loss (LM Loss)')
        ax.set_title('Validation Loss vs Iteration')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'iter_val_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 3: Iteration vs Train Step Time
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in train_combined['label'].unique():
        data = train_combined[train_combined['label'] == label]
        linestyle = '--' if data['force_lb'].iloc[0] else '-'
        ax.plot(data['iteration'], data['train_step_time'], label=label, alpha=0.7, linestyle=linestyle)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Train Step Time (seconds)')
    ax.set_title('Train Step Time vs Iteration')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iter_train_step_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Iteration vs Sequence Load Balancing Loss
    if 'slb_loss' in train_combined.columns and train_combined['slb_loss'].notna().any():
        fig, ax = plt.subplots(figsize=(12, 6))
        for label in train_combined['label'].unique():
            data = train_combined[train_combined['label'] == label]
            if data['slb_loss'].notna().any():
                linestyle = '--' if data['force_lb'].iloc[0] else '-'
                ax.plot(data['iteration'], data['slb_loss'], label=label, alpha=0.7, linestyle=linestyle)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Sequence Load Balancing Loss')
        ax.set_title('Sequence Load Balancing Loss vs Iteration')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'iter_slb_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 5: Samples vs Training Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    for label in train_combined['label'].unique():
        data = train_combined[train_combined['label'] == label]
        linestyle = '--' if data['force_lb'].iloc[0] else '-'
        ax.plot(data['samples'], data['train_loss'], label=label, alpha=0.7, linestyle=linestyle)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Samples')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples_train_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 6: Samples vs Validation Loss (with optional target line)
    if not val_combined.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        for label in val_combined['label'].unique():
            data = val_combined[val_combined['label'] == label]
            linestyle = '--' if data['force_lb'].iloc[0] else '-'
            ax.plot(data['samples'], data['val_loss'], label=label, marker='o', alpha=0.7, linestyle=linestyle)
        if target_loss is not None:
            ax.axhline(y=target_loss, color='r', linestyle='--', linewidth=2, label=f'Target Loss ({target_loss})')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Validation Loss (LM Loss)')
        ax.set_title('Validation Loss vs Samples')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'samples_val_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 7: Combined plot showing train and val loss vs iteration
    if not val_combined.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Training loss
        for label in train_combined['label'].unique():
            data = train_combined[train_combined['label'] == label]
            linestyle = '--' if data['force_lb'].iloc[0] else '-'
            ax1.plot(data['iteration'], data['train_loss'], label=label, alpha=0.7, linestyle=linestyle)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss vs Iteration')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Validation loss
        for label in val_combined['label'].unique():
            data = val_combined[val_combined['label'] == label]
            linestyle = '--' if data['force_lb'].iloc[0] else '-'
            ax2.plot(data['iteration'], data['val_loss'], label=label, marker='o', alpha=0.7, markersize=4, linestyle=linestyle)
        if target_loss is not None:
            ax2.axhline(y=target_loss, color='r', linestyle='--', linewidth=2, label=f'Target ({target_loss})')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Validation Loss (LM Loss)')
        ax2.set_title('Validation Loss vs Iteration')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_losses.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 8: Combined plot showing train and val loss vs samples
    if not val_combined.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Training loss vs samples
        for label in train_combined['label'].unique():
            data = train_combined[train_combined['label'] == label]
            linestyle = '--' if data['force_lb'].iloc[0] else '-'
            ax1.plot(data['samples'], data['train_loss'], label=label, alpha=0.7, linestyle=linestyle)
        ax1.set_xlabel('Samples')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss vs Samples')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Validation loss vs samples
        for label in val_combined['label'].unique():
            data = val_combined[val_combined['label'] == label]
            linestyle = '--' if data['force_lb'].iloc[0] else '-'
            ax2.plot(data['samples'], data['val_loss'], label=label, marker='o', alpha=0.7, markersize=4, linestyle=linestyle)
        if target_loss is not None:
            ax2.axhline(y=target_loss, color='r', linestyle='--', linewidth=2, label=f'Target ({target_loss})')
        ax2.set_xlabel('Samples')
        ax2.set_ylabel('Validation Loss (LM Loss)')
        ax2.set_title('Validation Loss vs Samples')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_losses_samples.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Plot 9: GBS vs Samples to Convergence
    # Collect convergence data for each log file
    convergence_data = []
    for log_file in log_files:
        config = extract_config(log_file)
        train_df, val_df, batch_size = extract_metrics(log_file)
        convergence_step = find_convergence_step(val_df, target_loss)

        if convergence_step is not None and config['batch_size'] is not None:
            samples_to_conv = convergence_step * config['batch_size']
            label = f"(lr={config['lr']}, warmup={config['warmup_steps']}, seed={config['seed']}, force_lb={config['moe_router_force_load_balancing']}"
            if include_filename:
                label += f", filename={log_file.name}"
            label += ")"
            convergence_data.append({
                'gbs': config['batch_size'],
                'samples_to_convergence': samples_to_conv,
                'lr': config['lr'],
                'warmup': config['warmup_steps'],
                'seed': config['seed'],
                'force_lb': config['moe_router_force_load_balancing'],
                'label': label,
            })

    if convergence_data:
        conv_df = pd.DataFrame(convergence_data)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each unique configuration
        for label in conv_df['label'].unique():
            data = conv_df[conv_df['label'] == label].sort_values('gbs')
            linestyle = '--' if data['force_lb'].iloc[0] else '-'
            ax.plot(data['gbs'], data['samples_to_convergence'], marker='o', label=label, alpha=0.7, markersize=8, linewidth=2, linestyle=linestyle)

        ax.set_xlabel('Global Batch Size (GBS)')
        ylabel = 'Samples to Convergence'
        if target_loss is not None:
            ylabel += f' (target loss: {target_loss})'
        ax.set_ylabel(ylabel)
        ax.set_title('Sample Efficiency vs Batch Size')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gbs_vs_samples_to_convergence.png'), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training curves from training log files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python visualize_training.py file.out

  # Multiple files
  python visualize_training.py file1.out file2.out

  # Directory (recursive search for *.out)
  python visualize_training.py logs_dir/

  # Mixed
  python visualize_training.py file.out logs_dir/

  # Custom output directory
  python visualize_training.py logs_dir/ --output-dir results/
        """
    )
    parser.add_argument(
        'inputs',
        nargs='+',
        help='Log files or directories to process (directories will be searched recursively for *.out files)'
    )
    parser.add_argument(
        '--output-dir',
        default='visualization_output',
        help='Output directory for visualizations and reports (default: visualization_output)'
    )
    parser.add_argument(
        '--target-loss',
        type=float,
        default=None,
        help='Target validation loss for convergence (optional, will draw target line on plots if specified)'
    )
    parser.add_argument(
        '--include-filename',
        action='store_true',
        default=False,
        help='Include the log filename (not full path) in the experiment label to distinguish between runs'
    )

    args = parser.parse_args()

    # Find all log files
    log_files = find_log_files(args.inputs)

    if not log_files:
        print("No .out files found!")
        return

    print(f"Found {len(log_files)} log file(s):")
    for f in log_files:
        print(f"  - {f}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate outputs
    print("Creating summary CSV...")
    create_summary_csv(log_files, os.path.join(args.output_dir, 'summary.csv'), args.target_loss)
    print()

    print("Creating Excel report...")
    create_excel_report(log_files, os.path.join(args.output_dir, 'detailed_metrics.xlsx'))
    print()

    print("Creating visualizations...")
    create_visualizations(log_files, args.output_dir, args.target_loss, args.include_filename)
    print()

    print(f"✓ All outputs saved to: {args.output_dir}/")
    print(f"  - summary.csv: Key metrics for each run")
    print(f"  - detailed_metrics.xlsx: Per-iteration metrics (one tab per log file)")
    print(f"  - *.png: Visualization plots")


if __name__ == '__main__':
    main()
