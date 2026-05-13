"""Winner-takes-all hard parcellation of the brain by OFC subregion.

Given fitted FC maps from each of the K OFC subregions, assign every brain
voxel to the subregion it connects to most strongly. Positive and negative
connections are parcellated separately, since each carries different
functional meaning (co-activation vs anti-correlation).
"""

import numpy as np
import nibabel as nib


def winner_takes_all(fc_maps, threshold=0.2):
    """Hard-parcellate by strongest connection per voxel.

    fc_maps   : (K, X, Y, Z) ndarray of fitted FC, one volume per subregion
    threshold : ignore |FC| below this cutoff (assigned to background = 0)

    Returns (pos, neg) : two (X, Y, Z) int volumes with labels 1..K.
    Voxels not exceeding the threshold remain 0.
    """
    K = fc_maps.shape[0]

    # Positive: argmax over subregions where FC > threshold
    pos_vals = np.where(fc_maps > threshold, fc_maps, 0)
    pos = np.argmax(pos_vals, axis=0) + 1
    pos[pos_vals.max(axis=0) == 0] = 0

    # Negative: argmax of |FC| where FC < -threshold
    neg_vals = np.where(fc_maps < -threshold, np.abs(fc_maps), 0)
    neg = np.argmax(neg_vals, axis=0) + 1
    neg[neg_vals.max(axis=0) == 0] = 0

    return pos.astype(np.int16), neg.astype(np.int16)


def load_subregion_maps(paths):
    """Load K NIfTI files and stack as (K, X, Y, Z); return stack + reference."""
    imgs = [nib.load(p) for p in paths]
    stack = np.stack([img.get_fdata() for img in imgs], axis=0)
    return stack, imgs[0]


def save_parcellation(labels, reference_img, out_path):
    """Save an integer-labelled volume using the reference image's affine."""
    nib.save(nib.Nifti1Image(labels, reference_img.affine, reference_img.header),
             out_path)
