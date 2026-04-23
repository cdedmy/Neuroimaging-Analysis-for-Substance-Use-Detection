"""
Schaefer 200 atlas to 7-network mapping.

The Schaefer 200 atlas organizes 200 ROIs into 7 canonical resting-state networks:
  VIS  — Visual
  SOM  — SomatoMotor
  DAN  — DorsalAttention
  SAL  — Salience/VentralAttention
  LIM  — Limbic
  CON  — Cont (FrontoParietal Control / Executive)
  DMN  — Default Mode

100 ROIs per hemisphere (LH first, then RH). Within each hemisphere, ROIs are
ordered by network. The official parcellation file lists each ROI's network
assignment.

This module gives us an approximate mapping so we can label discriminative edges
as "DMN ↔ CON" or "SAL ↔ LIM" etc. instead of "ROI_42 ↔ ROI_178".
"""

import numpy as np


# Approximate ROI counts per network in the Schaefer 200 (7-network version).
# Based on the official atlas organization: 100 ROIs per hemisphere, grouped by network.
# These are the approximate boundaries in ROI index order (0-99 for LH, 100-199 for RH).

NETWORK_RANGES_LH = [
    (0, 13, 'VIS'),     # Visual
    (13, 29, 'SOM'),    # SomatoMotor
    (29, 39, 'DAN'),    # DorsalAttention
    (39, 50, 'SAL'),    # Salience/VentralAttention
    (50, 54, 'LIM'),    # Limbic
    (54, 72, 'CON'),    # Control (FrontoParietal)
    (72, 100, 'DMN'),   # Default Mode
]

NETWORK_RANGES_RH = [
    (100, 114, 'VIS'),
    (114, 132, 'SOM'),
    (132, 144, 'DAN'),
    (144, 155, 'SAL'),
    (155, 160, 'LIM'),
    (160, 178, 'CON'),
    (178, 200, 'DMN'),
]

ALL_RANGES = NETWORK_RANGES_LH + NETWORK_RANGES_RH


NETWORK_FULL_NAMES = {
    'VIS': 'Visual',
    'SOM': 'SomatoMotor',
    'DAN': 'DorsalAttention',
    'SAL': 'Salience/VentralAttention',
    'LIM': 'Limbic',
    'CON': 'FrontoParietal Control',
    'DMN': 'Default Mode',
}


NETWORK_COLORS = {
    'VIS': '#781286',
    'SOM': '#4682B4',
    'DAN': '#00760E',
    'SAL': '#C43AFA',
    'LIM': '#DCF8A4',
    'CON': '#E69422',
    'DMN': '#CD3E4E',
}


def roi_to_network(roi_idx):
    """Return the network label (e.g. 'DMN') for a given ROI index (0-199)."""
    for start, end, net in ALL_RANGES:
        if start <= roi_idx < end:
            return net
    return 'UNK'


def roi_to_hemisphere(roi_idx):
    """Return 'LH' or 'RH' for a given ROI index."""
    return 'LH' if roi_idx < 100 else 'RH'


def edge_label(roi_i, roi_j, short=True):
    """Return a human-readable label for a connection between two ROIs.
    e.g. 'DMN-LH ↔ CON-RH' or 'DMN↔CON' """
    net_i = roi_to_network(roi_i)
    net_j = roi_to_network(roi_j)
    if short:
        if net_i == net_j:
            return f'{net_i} (within)'
        # Sort alphabetically for consistency
        a, b = sorted([net_i, net_j])
        return f'{a}↔{b}'
    else:
        h_i = roi_to_hemisphere(roi_i)
        h_j = roi_to_hemisphere(roi_j)
        return f'{net_i}-{h_i} ↔ {net_j}-{h_j}'


def network_assignments_array():
    """Return a (200,) array of network labels for each ROI."""
    return np.array([roi_to_network(i) for i in range(200)])


def summarize_top_edges(roi_pairs, importance_scores, top_k=30):
    """Given pairs of (roi_i, roi_j) and their importance scores, return a
    summary of which networks contribute most to the classification.

    Returns a dict with:
      - top_edges: list of (roi_i, roi_j, label, importance)
      - network_frequency: how often each network appears in top edges
      - network_pair_frequency: how often each pair of networks appears
    """
    from collections import Counter

    top = sorted(zip(roi_pairs, importance_scores), key=lambda x: -x[1])[:top_k]

    edge_records = []
    network_counter = Counter()
    pair_counter = Counter()

    for (i, j), imp in top:
        label = edge_label(i, j, short=True)
        edge_records.append({
            'roi_i': int(i),
            'roi_j': int(j),
            'hemisphere_i': roi_to_hemisphere(i),
            'hemisphere_j': roi_to_hemisphere(j),
            'network_i': roi_to_network(i),
            'network_j': roi_to_network(j),
            'label': label,
            'importance': float(imp),
        })
        network_counter[roi_to_network(i)] += 1
        network_counter[roi_to_network(j)] += 1
        pair_counter[label] += 1

    return {
        'top_edges': edge_records,
        'network_frequency': dict(network_counter.most_common()),
        'network_pair_frequency': dict(pair_counter.most_common()),
    }


if __name__ == "__main__":
    # Test
    print("Example ROI assignments:")
    for roi in [0, 50, 100, 150, 199]:
        print(f"  ROI {roi}: {roi_to_network(roi)} ({roi_to_hemisphere(roi)})")

    print("\nExample edge labels:")
    print(f"  ROI 20 ↔ ROI 90: {edge_label(20, 90)}")
    print(f"  ROI 5 ↔ ROI 105: {edge_label(5, 105)}")

    # Full labels for sanity check
    from collections import Counter
    assigns = network_assignments_array()
    print(f"\nROIs per network: {Counter(assigns)}")
