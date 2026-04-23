"""
Load FC matrices from DeepPrep output and match them with subject labels
from participants.tsv.

Expected input:
    fc_matrices/sub-001_fc.npy   (200x200 numpy array)
    fc_matrices/sub-002_fc.npy
    ...
    participants.tsv             (has 'participant_id' and 'group' columns)

Output: feature matrix X (n_subjects, n_features) and label vector y (n_subjects,)
"""

import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path


def flatten_upper_triangle(fc_matrix):
    """
    Convert symmetric FC matrix to 1D feature vector using upper triangle
    (excluding diagonal). This removes redundancy since FC is symmetric.

    For 200x200 matrix, returns 19,900 features (200*199/2).
    """
    n = fc_matrix.shape[0]
    indices = np.triu_indices(n, k=1)  # k=1 excludes diagonal
    return fc_matrix[indices]


def load_fc_matrices(fc_dir):
    """
    Load all FC matrices from a directory.
    Returns dict: {subject_id: fc_matrix}
    """
    fc_files = sorted(glob.glob(os.path.join(fc_dir, 'sub-*_fc.npy')))
    if not fc_files:
        raise FileNotFoundError(f"No FC matrices found in {fc_dir}")

    matrices = {}
    for fpath in fc_files:
        fname = os.path.basename(fpath)
        # Extract subject ID: "sub-001_fc.npy" -> "001"
        sub_id = fname.replace('sub-', '').replace('_fc.npy', '')
        matrices[sub_id] = np.load(fpath)

    print(f"Loaded {len(matrices)} FC matrices from {fc_dir}")
    print(f"Matrix shape: {list(matrices.values())[0].shape}")
    return matrices


def load_labels(participants_tsv, label_column='group'):
    """
    Load subject labels from participants.tsv.
    Returns dict: {subject_id: label}
    """
    df = pd.read_csv(participants_tsv, sep='\t')

    if 'participant_id' not in df.columns:
        raise KeyError(f"participants.tsv missing 'participant_id' column")
    if label_column not in df.columns:
        raise KeyError(f"participants.tsv missing '{label_column}' column. "
                       f"Available: {df.columns.tolist()}")

    df['sub_id'] = df['participant_id'].str.replace('sub-', '')
    labels = dict(zip(df['sub_id'], df[label_column]))
    print(f"Loaded {len(labels)} subject labels")
    print(f"Label distribution: {df[label_column].value_counts().to_dict()}")
    return labels


def build_dataset(fc_dir, participants_tsv, label_column='group',
                  flatten=True, dataset_tag=None):
    """
    Build X, y arrays for ML training.

    Args:
        fc_dir: directory containing sub-XXX_fc.npy files
        participants_tsv: path to participants.tsv
        label_column: which column to use as target
        flatten: if True, flatten FC matrices to 1D vectors (upper triangle)
        dataset_tag: optional string tag to prepend to subject IDs
                     (useful when combining datasets)

    Returns:
        X: (n_subjects, n_features) array
        y: (n_subjects,) array of labels
        subject_ids: list of subject IDs in same order as X
    """
    matrices = load_fc_matrices(fc_dir)
    labels = load_labels(participants_tsv, label_column)

    # Match matrices with labels
    X_list = []
    y_list = []
    subject_ids = []

    for sub_id, fc in matrices.items():
        if sub_id not in labels:
            print(f"  ⚠ sub-{sub_id} not in participants.tsv — skipping")
            continue

        if flatten:
            X_list.append(flatten_upper_triangle(fc))
        else:
            X_list.append(fc)

        y_list.append(labels[sub_id])
        if dataset_tag:
            subject_ids.append(f"{dataset_tag}_sub-{sub_id}")
        else:
            subject_ids.append(f"sub-{sub_id}")

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"\nFinal dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {np.unique(y).tolist()}")
    return X, y, subject_ids


def combine_datasets(datasets):
    """
    Combine multiple datasets into one.

    Args:
        datasets: list of tuples (X, y, subject_ids)

    Returns:
        combined X, y, subject_ids
    """
    X_all = np.vstack([d[0] for d in datasets])
    y_all = np.concatenate([d[1] for d in datasets])
    ids_all = []
    for d in datasets:
        ids_all.extend(d[2])

    print(f"\nCombined dataset:")
    print(f"  X shape: {X_all.shape}")
    print(f"  Classes: {np.unique(y_all, return_counts=True)}")
    return X_all, y_all, ids_all


if __name__ == "__main__":
    # Test run — update paths to match your setup
    FC_DIR = os.path.expanduser("~/sudmex/fc_matrices")
    PARTICIPANTS = os.path.expanduser("~/sudmex/bids/participants.tsv")

    if not os.path.exists(FC_DIR):
        print(f"FC directory not found: {FC_DIR}")
        print("Run Connor's notebook first to generate FC matrices, or update the path.")
    else:
        X, y, ids = build_dataset(FC_DIR, PARTICIPANTS)
        print(f"\nSample subjects: {ids[:3]}")
        print(f"Sample labels:   {y[:3]}")
