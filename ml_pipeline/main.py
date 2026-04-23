#!/usr/bin/env python3
"""
Main pipeline: load FC matrices -> train classifier -> generate figures -> save results.

Usage:
    python3 main.py --fc-dir ~/sudmex/fc_matrices --participants ~/sudmex/bids/participants.tsv

For combining cocaine + cannabis datasets:
    python3 main.py \
        --fc-dir ~/sudmex/fc_matrices ~/cannabis/fc_matrices \
        --participants ~/sudmex/bids/participants.tsv ~/cannabis/bids/participants.tsv \
        --tags cocaine cannabis
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from load_data import build_dataset, combine_datasets, load_fc_matrices, load_labels
from classifier import train_all_models, build_models
from visualize import (
    plot_confusion_matrix, plot_roc_curve, plot_group_fc_comparison,
    plot_top_features, plot_model_comparison
)
from sklearn.model_selection import cross_val_predict, StratifiedKFold


def run_pipeline(fc_dirs, participants_tsvs, tags, output_dir,
                 label_column='group', n_splits=5, n_features_select=500):
    """Run the full classification pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)

    print(f"{'='*60}")
    print("SUBSTANCE USE DETECTION — ML PIPELINE")
    print(f"{'='*60}")
    print(f"Output dir: {output_dir}")

    # --- 1. Load data ---
    datasets = []
    all_fc_matrices = {}  # for group visualization later
    all_labels = {}

    for fc_dir, tsv, tag in zip(fc_dirs, participants_tsvs, tags):
        print(f"\n--- Loading {tag} dataset ---")
        X, y, ids = build_dataset(fc_dir, tsv, label_column=label_column,
                                    flatten=True, dataset_tag=tag)
        datasets.append((X, y, ids))

        # Also keep unflattened matrices for FC visualization
        matrices = load_fc_matrices(fc_dir)
        labels = load_labels(tsv, label_column)
        for sub, fc in matrices.items():
            all_fc_matrices[f"{tag}_sub-{sub}"] = fc
            if sub in labels:
                all_labels[f"{tag}_sub-{sub}"] = labels[sub]

    # Combine
    if len(datasets) > 1:
        X, y, ids = combine_datasets(datasets)
    else:
        X, y, ids = datasets[0]

    # --- 2. Train all models ---
    results_df, predictions = train_all_models(X, y, n_features_select=n_features_select,
                                                 n_splits=n_splits)
    results_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"\nSaved: {output_dir / 'model_comparison.csv'}")
    print("\n" + results_df.to_string(index=False))

    # --- 3. Identify best model ---
    valid = results_df.dropna(subset=['balanced_accuracy']) if 'balanced_accuracy' in results_df else results_df
    best_row = valid.loc[valid['balanced_accuracy'].idxmax()]
    best_name = best_row['model']
    y_pred_best = predictions[best_name]

    print(f"\n🏆 Best model: {best_name}")

    # Retrain best model on full data (for feature importance)
    models = build_models(n_features_select=n_features_select)
    best_model = models[best_name]
    best_model.fit(X, y)

    # --- 4. Generate figures ---
    print(f"\n--- Generating figures ---")

    plot_model_comparison(results_df, save_path=fig_dir / 'model_comparison.png')

    plot_confusion_matrix(y, y_pred_best,
                          save_path=fig_dir / 'confusion_matrix.png',
                          title=f'{best_name} — Confusion Matrix')

    # ROC (binary only)
    if len(np.unique(y)) == 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            y_proba = cross_val_predict(best_model, X, y, cv=cv,
                                        method='predict_proba', n_jobs=-1)
            plot_roc_curve(y, y_proba[:, 1],
                           save_path=fig_dir / 'roc_curve.png',
                           title=f'{best_name} — ROC Curve')
        except Exception as e:
            print(f"  ROC skipped: {e}")

    # Group FC comparison
    if all_fc_matrices:
        plot_group_fc_comparison(all_fc_matrices, all_labels,
                                  save_path=fig_dir / 'group_fc_comparison.png')

    # Feature importance
    n_rois = list(all_fc_matrices.values())[0].shape[0]
    plot_top_features(best_model, n_rois=n_rois,
                       save_path=fig_dir / 'top_features.png')

    # --- 5. Summary ---
    summary = {
        'n_subjects': len(y),
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'classes': dict(zip(*np.unique(y, return_counts=True))),
        'best_model': best_name,
        'best_balanced_accuracy': float(best_row['balanced_accuracy']),
        'best_accuracy': float(best_row['accuracy']),
    }
    pd.Series(summary).to_json(output_dir / 'summary.json', indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Pipeline complete")
    print(f"   Subjects:       {summary['n_subjects']}")
    print(f"   Features:       {summary['n_features']}")
    print(f"   Classes:        {summary['classes']}")
    print(f"   Best model:     {summary['best_model']}")
    print(f"   Balanced acc:   {summary['best_balanced_accuracy']:.3f}")
    print(f"{'='*60}")
    print(f"Results: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fc-dir', nargs='+', required=True,
                        help='One or more directories containing sub-XXX_fc.npy files')
    parser.add_argument('--participants', nargs='+', required=True,
                        help='participants.tsv file for each dataset (same order)')
    parser.add_argument('--tags', nargs='+', default=None,
                        help='Tag for each dataset (e.g. cocaine cannabis)')
    parser.add_argument('--output-dir', default='./results',
                        help='Where to save results and figures')
    parser.add_argument('--label-column', default='group',
                        help='Column in participants.tsv to use as label')
    parser.add_argument('--cv-splits', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--n-features', type=int, default=500,
                        help='Number of top features to select')

    args = parser.parse_args()

    if args.tags is None:
        args.tags = [f'dataset{i+1}' for i in range(len(args.fc_dir))]

    if len(args.fc_dir) != len(args.participants):
        print("ERROR: --fc-dir and --participants must have same number of entries")
        sys.exit(1)

    run_pipeline(
        fc_dirs=[os.path.expanduser(d) for d in args.fc_dir],
        participants_tsvs=[os.path.expanduser(p) for p in args.participants],
        tags=args.tags,
        output_dir=args.output_dir,
        label_column=args.label_column,
        n_splits=args.cv_splits,
        n_features_select=args.n_features,
    )


if __name__ == "__main__":
    main()
