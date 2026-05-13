"""Seed-voxel to target-region functional connectivity (Fisher z).

Inputs
------
fmri_data    : (X, Y, Z, T) ndarray
seed_mask    : (X, Y, Z) bool ndarray   -- e.g. OFC voxels
target_mask  : (X, Y, Z) int  ndarray   -- e.g. parcellation, labels > 0

Output
------
fc_z : (n_seed_voxels, n_target_regions) Fisher-z transformed Pearson r.
"""

import numpy as np
from scipy.stats import pearsonr


def calculate_fc_z(fmri_data, seed_mask, target_mask):
    x, y, z, t = fmri_data.shape
    ts = fmri_data.reshape(-1, t)

    # Voxels with non-zero variance are the only valid ones
    valid = ts.std(axis=1) > 0
    seed_idx = np.where(seed_mask.ravel() & valid)[0]
    labels = np.unique(target_mask[target_mask > 0])

    fc = np.zeros((len(seed_idx), len(labels)))

    for i, lab in enumerate(labels):
        region_idx = np.where((target_mask.ravel() == lab) & valid)[0]
        if region_idx.size == 0:
            continue
        region_ts = ts[region_idx].mean(axis=0)

        for j, v in enumerate(seed_idx):
            r, _ = pearsonr(ts[v], region_ts)
            fc[j, i] = np.arctanh(np.clip(np.nan_to_num(r), -0.999999, 0.999999))

    return fc
