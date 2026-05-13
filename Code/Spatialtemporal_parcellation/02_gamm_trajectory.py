"""Hierarchical clustering of OFC voxel trajectories.

Pipeline:
1. correlation distance (1 - r) between voxel-wise developmental trajectories
2. Ward linkage; evaluate k = 2..K by WCSS + silhouette
3. cut tree at chosen k, write labels back into a NIfTI in MNI space.

Inputs
------
trajectories : (n_voxels, n_timepoints) ndarray
coords       : (n_voxels, 3) ndarray of MNI x,y,z
mask_img     : nibabel image -- target volume / affine for the output
"""

import numpy as np
import nibabel as nib
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score


def correlation_distance(trajectories):
    """1 - Pearson r between every pair of voxel trajectories."""
    return 1 - np.corrcoef(trajectories)


def evaluate_k(distance_matrix, k_range=range(2, 16), method="ward"):
    """Sweep k and return WCSS + silhouette for each."""
    Z = linkage(distance_matrix, method=method)
    out = []
    for k in k_range:
        lab = fcluster(Z, t=k, criterion="maxclust")
        wcss = sum(distance_matrix[lab == c][:, lab == c].sum()
                   for c in np.unique(lab))
        sil = silhouette_score(distance_matrix, lab, metric="precomputed")
        out.append((k, wcss, sil))
    return Z, out


def cluster(distance_matrix, k, method="ward"):
    """Ward linkage, cut at k, return (labels, linkage_matrix)."""
    Z = linkage(distance_matrix, method=method)
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels, Z


def labels_to_nifti(labels, coords_mni, mask_img):
    """Write cluster labels into a NIfTI volume using mask_img's affine."""
    out = np.zeros(mask_img.shape, dtype=np.int16)
    inv = np.linalg.inv(mask_img.affine)

    # MNI (n, 3) -> voxel (n, 3) via homogeneous coords
    hom = np.c_[coords_mni, np.ones(len(coords_mni))]
    vox = np.round(hom @ inv.T)[:, :3].astype(int)

    in_bounds = ((vox >= 0) & (vox < np.array(mask_img.shape))).all(axis=1)
    for (x, y, z), lab in zip(vox[in_bounds], labels[in_bounds]):
        out[x, y, z] = lab

    return nib.Nifti1Image(out, mask_img.affine, mask_img.header)
