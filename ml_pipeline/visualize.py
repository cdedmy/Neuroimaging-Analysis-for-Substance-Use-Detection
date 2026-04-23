"""
Visualization utilities for the classifier results.

Generates:
  - Confusion matrix heatmap
  - ROC curve (binary only)
  - Feature importance (top FC edges)
  - Group-averaged FC comparison heatmaps
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None, title=None):
    """Plot confusion matrix heatmap."""
    if labels is None:
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0],
                cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (counts)')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1],
                vmin=0, vmax=1, cbar_kws={'label': 'Rate'})
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Normalized (recall)')

    if title:
        fig.suptitle(title, y=1.02, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_scores, save_path=None, title='ROC Curve'):
    """Plot ROC curve (binary classification only)."""
    classes = np.unique(y_true)
    if len(classes) != 2:
        print("ROC curve only supports binary classification")
        return

    # Use second class as positive
    y_binary = (y_true == classes[1]).astype(int)
    fpr, tpr, _ = roc_curve(y_binary, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#e94560', lw=2.5,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
             label='Chance (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_group_fc_comparison(fc_matrices_dict, labels_dict, save_path=None,
                             atlas_name='Schaefer 200'):
    """
    Plot mean FC matrix per group + difference.

    Args:
        fc_matrices_dict: {subject_id: (n_rois, n_rois) matrix}
        labels_dict: {subject_id: group_label}
    """
    # Group matrices by label
    groups = {}
    for sub_id, fc in fc_matrices_dict.items():
        label = labels_dict.get(sub_id)
        if label is None:
            continue
        groups.setdefault(label, []).append(fc)

    group_names = sorted(groups.keys())
    n_groups = len(group_names)

    # Compute mean FC per group
    mean_fcs = {g: np.mean(np.array(groups[g]), axis=0) for g in group_names}

    # Plot: mean per group + pairwise diff
    n_plots = n_groups + (1 if n_groups == 2 else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for i, g in enumerate(group_names):
        sns.heatmap(mean_fcs[g], cmap='RdBu_r', center=0, vmin=-0.6, vmax=0.6,
                    square=True, xticklabels=False, yticklabels=False, ax=axes[i],
                    cbar_kws={'label': 'Correlation'})
        axes[i].set_title(f'Mean FC — {g} (n={len(groups[g])})')

    # Difference (binary only)
    if n_groups == 2:
        diff = mean_fcs[group_names[1]] - mean_fcs[group_names[0]]
        sns.heatmap(diff, cmap='RdBu_r', center=0, vmin=-0.3, vmax=0.3,
                    square=True, xticklabels=False, yticklabels=False, ax=axes[-1],
                    cbar_kws={'label': 'Δ Correlation'})
        axes[-1].set_title(f'Difference: {group_names[1]} − {group_names[0]}')

    fig.suptitle(f'Group FC Comparison ({atlas_name})', y=1.02, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_top_features(model, n_rois=200, top_k=30, save_path=None):
    """
    Visualize the top-k most discriminative FC edges identified by the model.
    Labels edges by Schaefer 7-network assignment (DMN, CON, VIS, etc.)

    Converts from feature index (upper triangle) back to ROI pairs, then maps
    to brain network names for interpretability.
    """
    from schaefer_networks import (
        edge_label, roi_to_network, roi_to_hemisphere,
        summarize_top_edges, NETWORK_COLORS, NETWORK_FULL_NAMES
    )

    # Get the SelectKBest step to find which features were selected
    try:
        selector = model.named_steps['select']
        selected_idx = selector.get_support(indices=True)
        scores = selector.scores_[selected_idx]
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return

    # Get classifier feature importances/coefficients
    clf = model.named_steps['clf']
    if hasattr(clf, 'coef_'):
        importance = np.abs(clf.coef_).mean(axis=0)
    elif hasattr(clf, 'feature_importances_'):
        importance = clf.feature_importances_
    else:
        importance = scores

    # Top-k most important
    top_k = min(top_k, len(importance))
    top_indices = np.argsort(importance)[-top_k:][::-1]

    # Map feature index back to (roi_i, roi_j) using upper triangle indexing
    triu_i, triu_j = np.triu_indices(n_rois, k=1)
    orig_feature_idx = selected_idx[top_indices]
    roi_pairs = [(triu_i[idx], triu_j[idx]) for idx in orig_feature_idx]
    importances_top = importance[top_indices]

    # Use the schaefer helper to summarize by network
    summary = summarize_top_edges(roi_pairs, importances_top, top_k=top_k)

    # ---- Plot: 4 panels ----
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # Panel 1: Top edges bar chart with network labels
    ax1 = fig.add_subplot(gs[0, 0])
    labels = [f'{e["label"]}' for e in summary['top_edges']]
    colors = []
    for e in summary['top_edges']:
        net_a = e['network_i']
        net_b = e['network_j']
        # Mix colors of the two networks
        c1 = NETWORK_COLORS.get(net_a, '#999')
        colors.append(c1)

    y_pos = np.arange(top_k)
    ax1.barh(y_pos, [e['importance'] for e in summary['top_edges']][::-1],
             color=colors[::-1], edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels[::-1], fontsize=8)
    ax1.set_xlabel('Importance')
    ax1.set_title(f'Top {top_k} Discriminative FC Edges\n(labeled by Schaefer 7-network)',
                  fontsize=12, fontweight='bold')

    # Panel 2: Network pair frequency
    ax2 = fig.add_subplot(gs[0, 1])
    pair_freq = summary['network_pair_frequency']
    if pair_freq:
        pairs = list(pair_freq.keys())[:15]  # top 15
        counts = [pair_freq[p] for p in pairs]
        y = np.arange(len(pairs))
        ax2.barh(y, counts, color='#e94560', edgecolor='black', linewidth=0.5)
        ax2.set_yticks(y)
        ax2.set_yticklabels(pairs, fontsize=9)
        ax2.set_xlabel(f'Count (out of top {top_k} edges)')
        ax2.set_title('Which network pairs are most discriminative?',
                      fontsize=12, fontweight='bold')
        ax2.invert_yaxis()

    # Panel 3: Network frequency (how often each network appears)
    ax3 = fig.add_subplot(gs[1, 0])
    net_freq = summary['network_frequency']
    nets = list(net_freq.keys())
    counts = [net_freq[n] for n in nets]
    colors_net = [NETWORK_COLORS.get(n, '#999') for n in nets]
    ax3.bar(nets, counts, color=colors_net, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel(f'Frequency in top {top_k} edges')
    ax3.set_title('Network involvement in discriminative edges',
                  fontsize=12, fontweight='bold')
    # Add full names as legend
    legend_items = [f'{n} = {NETWORK_FULL_NAMES.get(n, "?")}' for n in nets]
    ax3.text(1.02, 0.95, '\n'.join(legend_items), transform=ax3.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel 4: Connection matrix
    ax4 = fig.add_subplot(gs[1, 1])
    conn_mat = np.zeros((n_rois, n_rois))
    for (i, j), imp in zip(roi_pairs, importances_top):
        conn_mat[i, j] = imp
        conn_mat[j, i] = imp

    sns.heatmap(conn_mat, cmap='Reds', square=True, xticklabels=False,
                yticklabels=False, ax=ax4, cbar_kws={'label': 'Importance'})
    # Add hemisphere dividers
    ax4.axhline(y=100, color='black', linewidth=1)
    ax4.axvline(x=100, color='black', linewidth=1)
    ax4.set_title('Spatial distribution (LH/RH)',
                  fontsize=12, fontweight='bold')

    plt.suptitle('Discriminative FC Edges — Schaefer 7-Network Analysis',
                 fontsize=14, fontweight='bold', y=1.00)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

        # Also save network summary as text
        txt_path = str(save_path).replace('.png', '_summary.txt')
        with open(txt_path, 'w') as f:
            f.write('Top Discriminative FC Edges — Network Analysis\n')
            f.write('=' * 60 + '\n\n')
            f.write('Network pair frequency (top 10):\n')
            for pair, cnt in list(summary['network_pair_frequency'].items())[:10]:
                f.write(f'  {pair:20s} : {cnt} edges\n')
            f.write('\nMost involved networks:\n')
            for net, cnt in summary['network_frequency'].items():
                f.write(f'  {net} ({NETWORK_FULL_NAMES.get(net,"?")}): {cnt}\n')
            f.write('\nTop 20 individual edges:\n')
            for e in summary['top_edges'][:20]:
                f.write(f'  ROI {e["roi_i"]:3d} ({e["network_i"]}-{e["hemisphere_i"]}) ↔ '
                        f'ROI {e["roi_j"]:3d} ({e["network_j"]}-{e["hemisphere_j"]})  '
                        f'imp={e["importance"]:.4f}\n')
        print(f"  Saved: {txt_path}")
    plt.close()


def plot_model_comparison(results_df, save_path=None):
    """Bar chart comparing models across metrics."""
    metrics = [c for c in ['accuracy', 'balanced_accuracy', 'f1_macro', 'roc_auc']
               if c in results_df.columns]

    df_plot = results_df.set_index('model')[metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot.plot(kind='bar', ax=ax, colormap='viridis', edgecolor='black', width=0.8)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.set_title('Model Performance Comparison')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.05))
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Demo plots
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 1])
    plot_confusion_matrix(y_true, y_pred, labels=[0, 1], save_path='demo_cm.png')
    print("Demo plots generated.")
