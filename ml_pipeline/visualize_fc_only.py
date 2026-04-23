#!/usr/bin/env python3
"""
Visualize FC matrices when we don't have enough subjects to train a classifier.
Shows: individual FC heatmaps, group comparison, summary stats, network structure.

Usage:
    python3 visualize_fc_only.py \
        --fc-dir ~/CS4001/FINAL-PROJECT/classifier_data/fc_matrices \
        --participants ~/CS4001/FINAL-PROJECT/classifier_data/participants.tsv
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob


def load_fc_matrices(fc_dir):
    fc_files = sorted(glob.glob(os.path.join(fc_dir, 'sub-*_fc.npy')))
    matrices = {}
    for fpath in fc_files:
        fname = os.path.basename(fpath)
        sub_id = fname.replace('sub-', '').replace('_fc.npy', '')
        matrices[sub_id] = np.load(fpath)
    return matrices


def load_labels(participants_tsv):
    df = pd.read_csv(participants_tsv, sep='\t')
    df['sub_id'] = df['participant_id'].str.replace('sub-', '')
    return df


def fc_stats(fc):
    """Summary statistics for an FC matrix."""
    triu = fc[np.triu_indices_from(fc, k=1)]
    return {
        'n_rois': fc.shape[0],
        'n_edges': len(triu),
        'mean': float(np.mean(triu)),
        'std': float(np.std(triu)),
        'min': float(np.min(triu)),
        'max': float(np.max(triu)),
        'positive_edges_%': float(np.mean(triu > 0) * 100),
        'strong_edges_%': float(np.mean(np.abs(triu) > 0.5) * 100),
    }


def plot_individual_fcs(matrices, labels_df, output_dir):
    """Plot each subject's FC matrix."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]

    for i, (sub_id, fc) in enumerate(matrices.items()):
        label_row = labels_df[labels_df['sub_id'] == sub_id]
        group = label_row['group'].values[0] if len(label_row) else 'unknown'

        sns.heatmap(fc, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    square=True, xticklabels=False, yticklabels=False,
                    ax=axes[i], cbar_kws={'label': 'Correlation'})
        axes[i].set_title(f'sub-{sub_id} (group: {group})', fontsize=14)

    fig.suptitle('Individual Functional Connectivity Matrices (Schaefer 200 ROIs)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_dir / 'individual_fc_matrices.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_fc_comparison(matrices, labels_df, output_dir):
    """Plot difference between 2 subjects."""
    if len(matrices) != 2:
        print("  Comparison only works with 2 subjects, skipping")
        return

    keys = list(matrices.keys())
    fc1, fc2 = matrices[keys[0]], matrices[keys[1]]
    diff = fc1 - fc2

    label1 = labels_df[labels_df['sub_id'] == keys[0]]['group'].values
    label2 = labels_df[labels_df['sub_id'] == keys[1]]['group'].values
    g1 = label1[0] if len(label1) else '?'
    g2 = label2[0] if len(label2) else '?'

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, (fc, title) in enumerate([(fc1, f'sub-{keys[0]} (group {g1})'),
                                       (fc2, f'sub-{keys[1]} (group {g2})'),
                                       (diff, f'Difference')]):
        vmax = 1 if i < 2 else 0.5
        sns.heatmap(fc, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                    square=True, xticklabels=False, yticklabels=False,
                    ax=axes[i])
        axes[i].set_title(title, fontsize=14)

    fig.suptitle('FC Comparison Between Subjects', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = output_dir / 'fc_comparison.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Similarity stats
    corr = np.corrcoef(fc1.flatten(), fc2.flatten())[0, 1]
    print(f"\n  Pearson correlation between subjects' FC: {corr:.3f}")
    print(f"  Mean absolute difference:                 {np.mean(np.abs(diff)):.3f}")


def plot_distribution(matrices, labels_df, output_dir):
    """Plot distribution of edge weights."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for sub_id, fc in matrices.items():
        triu = fc[np.triu_indices_from(fc, k=1)]
        label_row = labels_df[labels_df['sub_id'] == sub_id]
        group = label_row['group'].values[0] if len(label_row) else '?'
        ax.hist(triu, bins=60, alpha=0.5, label=f'sub-{sub_id} (group {group})',
                edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Correlation value', fontsize=12)
    ax.set_ylabel('Number of edges', fontsize=12)
    ax.set_title('Distribution of FC Edge Weights', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = output_dir / 'edge_distribution.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_network_modules(matrices, output_dir):
    """Plot block structure using Schaefer 7-network organization.
    The 200 ROIs are organized into 7 canonical networks."""
    # Schaefer 200, 7 networks — approximate network boundaries
    # (VIS, SOM, DAN, VAN, LIM, FPN, DMN) — 100 per hemisphere
    n_rois = 200

    for sub_id, fc in matrices.items():
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(fc, cmap='RdBu_r', center=0, vmin=-0.8, vmax=0.8,
                    square=True, xticklabels=False, yticklabels=False, ax=ax,
                    cbar_kws={'label': 'Correlation'})
        # Add hemisphere divider
        ax.axhline(y=100, color='black', linewidth=1)
        ax.axvline(x=100, color='black', linewidth=1)
        ax.text(50, -5, 'Left Hemisphere', ha='center', fontsize=11, fontweight='bold')
        ax.text(150, -5, 'Right Hemisphere', ha='center', fontsize=11, fontweight='bold')
        ax.set_title(f'sub-{sub_id} — Schaefer 200 FC Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = output_dir / f'sub-{sub_id}_network.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def print_summary_table(matrices, labels_df):
    """Print a summary of stats per subject."""
    rows = []
    for sub_id, fc in matrices.items():
        label_row = labels_df[labels_df['sub_id'] == sub_id]
        group = label_row['group'].values[0] if len(label_row) else '?'
        stats = fc_stats(fc)
        rows.append({'subject': f'sub-{sub_id}', 'group': group, **stats})

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fc-dir', required=True)
    parser.add_argument('--participants', required=True)
    parser.add_argument('--output-dir', default='./results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("FC MATRIX VISUALIZATION PIPELINE")
    print("=" * 60)

    # Load
    matrices = load_fc_matrices(os.path.expanduser(args.fc_dir))
    labels_df = load_labels(os.path.expanduser(args.participants))
    print(f"\nLoaded {len(matrices)} FC matrices")
    print(f"Subjects: {list(matrices.keys())}")

    # Summary table
    print("\n--- SUMMARY STATS ---")
    summary = print_summary_table(matrices, labels_df)
    print(summary.to_string(index=False))
    summary.to_csv(output_dir / 'fc_summary.csv', index=False)

    # Visualizations
    print("\n--- GENERATING FIGURES ---")
    plot_individual_fcs(matrices, labels_df, fig_dir)
    plot_fc_comparison(matrices, labels_df, fig_dir)
    plot_distribution(matrices, labels_df, fig_dir)
    plot_network_modules(matrices, fig_dir)

    # Note about classification
    n_per_class = labels_df[labels_df['sub_id'].isin(matrices.keys())]['group'].value_counts()
    print(f"\n--- CLASSIFICATION STATUS ---")
    print(f"Subjects per class: {n_per_class.to_dict()}")
    print(f"Minimum n/class for CV: {n_per_class.min()}")
    if n_per_class.min() < 3:
        print("⚠️  Cannot train classifier — need at least 3 subjects per class")
        print("   This demonstration shows the pipeline works end-to-end.")
        print("   In production, 30+ subjects per class would be needed.")

    print(f"\n✅ Done! Results in: {output_dir}")


if __name__ == "__main__":
    main()
