#!/usr/bin/env python3
"""
Automated smoke tests for the ML pipeline.
Runs in ~5 seconds with synthetic data. No FABRIC required.

Usage:
    python3 test_pipeline.py
"""

import numpy as np
import tempfile
import os
import shutil
import sys


def test_load_data():
    """Test that load_data produces expected shapes."""
    from load_data import flatten_upper_triangle, build_dataset

    # Synthetic 200x200 symmetric matrix
    fc = np.random.rand(200, 200)
    fc = (fc + fc.T) / 2
    np.fill_diagonal(fc, 1.0)

    flat = flatten_upper_triangle(fc)
    assert flat.shape == (200 * 199 // 2,), f"Expected 19900 features, got {flat.shape}"
    print("  ✅ flatten_upper_triangle: 200x200 → 19,900 features")


def test_classifier():
    """Test that classifier trains on synthetic data."""
    from classifier import build_models, evaluate_model
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=20, n_features=500, n_classes=2, random_state=42)
    models = build_models(n_features_select=100)

    for name, model in models.items():
        metrics, _ = evaluate_model(model, X, y, n_splits=3)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        print(f"  ✅ {name}: accuracy={metrics['accuracy']:.2f}")


def test_schaefer_networks():
    """Test the network labeling."""
    from schaefer_networks import (
        roi_to_network, edge_label, summarize_top_edges,
        network_assignments_array
    )

    # Spot-check known networks
    assert roi_to_network(0) == 'VIS'  # first ROI is visual
    assert roi_to_network(99) == 'DMN'  # last LH ROI is DMN
    print(f"  ✅ roi_to_network: ROI 0 → VIS, ROI 99 → DMN")

    # Edge label
    label = edge_label(20, 90)
    assert '↔' in label or 'within' in label
    print(f"  ✅ edge_label(20, 90) = {label}")

    # Summary
    roi_pairs = [(0, 50), (10, 80), (20, 150)]
    scores = [0.9, 0.7, 0.5]
    summary = summarize_top_edges(roi_pairs, scores, top_k=3)
    assert 'top_edges' in summary
    assert 'network_frequency' in summary
    print(f"  ✅ summarize_top_edges: {summary['network_pair_frequency']}")

    # Array
    arr = network_assignments_array()
    assert arr.shape == (200,)
    assert 'DMN' in arr
    print(f"  ✅ network_assignments_array: shape={arr.shape}, has DMN")


def test_end_to_end():
    """Full pipeline on synthetic FC matrices."""
    tmpdir = tempfile.mkdtemp()
    try:
        # Create synthetic FC data
        fc_dir = os.path.join(tmpdir, 'fc_matrices')
        os.makedirs(fc_dir)

        # 6 subjects, 3 per class
        labels = []
        for i in range(6):
            group = 1 if i < 3 else 2
            fc = np.random.rand(200, 200)
            fc = (fc + fc.T) / 2
            np.fill_diagonal(fc, 1.0)
            # Add class-specific signal
            if group == 1:
                fc[0:20, 0:20] += 0.3
            else:
                fc[50:80, 50:80] += 0.3
            fc = np.clip(fc, -1, 1)
            np.fill_diagonal(fc, 1.0)
            np.save(os.path.join(fc_dir, f'sub-{i+1:03d}_fc.npy'), fc)
            labels.append({'participant_id': f'sub-{i+1:03d}', 'group': group})

        # participants.tsv
        import pandas as pd
        tsv_path = os.path.join(tmpdir, 'participants.tsv')
        pd.DataFrame(labels).to_csv(tsv_path, sep='\t', index=False)

        # Run the pipeline
        from load_data import build_dataset
        from classifier import train_all_models

        X, y, ids = build_dataset(fc_dir, tsv_path)
        assert X.shape == (6, 19900)
        assert len(np.unique(y)) == 2
        print(f"  ✅ build_dataset: X={X.shape}, y classes={np.unique(y).tolist()}")

        results_df, predictions = train_all_models(X, y, n_splits=3)
        assert len(results_df) >= 1
        print(f"  ✅ train_all_models: {len(results_df)} models trained")

    finally:
        shutil.rmtree(tmpdir)


def main():
    print("=" * 60)
    print("ML PIPELINE SMOKE TESTS")
    print("=" * 60)

    tests = [
        ('load_data', test_load_data),
        ('classifier', test_classifier),
        ('schaefer_networks', test_schaefer_networks),
        ('end-to-end', test_end_to_end),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    print("=" * 60)

    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()
