import logging

import numpy as np
from sklearn.cluster import KMeans

from mri_project.utility import get_outliers

logger = logging.getLogger(__name__)


def get_modes(values, **hist_args):
    peaks, _ = np.histogram(values, **hist_args)
    modes = []
    if peaks[0] > peaks[1]:
        modes.append(0)
    for i in range(1, len(peaks) - 1):
        val = peaks[i]
        if (val > peaks[i - 1]) & (val > peaks[i + 1]):
            modes.append(i)
    return modes


def cluster_masked_muscle(masked_muscle: np.ndarray, nclusters: int):
    """
    Parameters
    ----------
    masked_muscle: an image in which a single muscle is masked out and the rest is 0
    nclusters: the number of clusters used in thresholding

    Returns
    --------
    clusters: the clusters corresponding to masked_muscle[np.where(masked_muscle)]
    """
    km_m = KMeans(nclusters)
    pred_musc_pix = np.where(masked_muscle)
    x = masked_muscle[pred_musc_pix].reshape(-1, 1)
    km_m.fit(x)
    clusters = km_m.predict(x)
    return clusters


def post_mus_from_multiclass_mask(raw_image: np.ndarray, mask: np.ndarray, nclusters=2):
    """
    Parameters
    ----------
    raw_image: a 2d grayscale image
    mask: a 2d binary image with the same shape as raw_image

    Returns
    -------
    segmented_muscle: a 2d binary image where 1 means the pixel is a muscle and 0 not a muscle
    """
    # assumption: muscles constitute a larger portion of the masked muscle image
    assert raw_image.shape == mask.shape
    masked_muscle = raw_image * mask
    clusters = cluster_masked_muscle(masked_muscle, nclusters=nclusters)
    mask_mus_clus_num = np.argmax(np.bincount(clusters))
    clusters = clusters == mask_mus_clus_num
    segmented_muscle = np.zeros_like(masked_muscle, dtype='uint8')
    segmented_muscle[np.where(mask)] = clusters
    # todo: change the following code segment so that it works for more than two muscles
    muscle_mean = (raw_image[segmented_muscle.astype(bool)]).mean()
    nonmuscle_mean = (raw_image[(mask - segmented_muscle).astype(bool)]).mean()
    nonmuscle_std = (raw_image[(mask - segmented_muscle).astype(bool)]).std()
    if muscle_mean > nonmuscle_mean + nonmuscle_std / 6:
        segmented_muscle = (1 - segmented_muscle) * mask
    return segmented_muscle


def segment_mus_from_labels(orig, labels, processing_fun, *fun_args, **fun_kwargs):
    """
    Parameters
    ----------
    orig: 2d array of grayscale image
    labels: 2d array or predictions as a multiclass mask
    processing_fun: the function to be used for post-processing single classes. The function should take two params:
        the original image and a binary mask corresponding to a single muscle and additional arguments if necessary.
    fun_args: additional arguments passed to `processing_fun`
    fun_kwargs: additional arguments passed to `processing_fun`

    Return
    ---------
    a multilabel masked image with each class corresponding to a muscle just the same as `labels`
    """
    classes = sorted(np.unique(labels))[1:]
    out = np.zeros_like(labels)
    for cls in classes:
        mask = labels == cls
        out += processing_fun(orig, mask, *fun_args, **fun_kwargs) * cls
    return out


def muscle_mean_thresh(orig, mask, std_ratio):
    mean_ = orig[mask].mean()
    std_ = orig[mask].std()
    return (orig < mean_ + std_ * std_ratio) * mask


def muscle_outlier_thresh(orig, mask, r):
    indices = np.where(mask)
    values = orig[indices]
    good, bad = get_outliers(values, r=r)
    # print(len(good[0]), len(bad[0]))
    if len(bad[0]) == 0:
        # print("dropping back to mean")
        logger.warning("[*] Zero outliers found. Dropping back to mean ...")
        out = muscle_mean_thresh(orig, mask, 1/5)
    else:
        bad_indices = tuple(np.array(indices)[:, bad])
        out = mask.copy()
        out[bad_indices] = False
    return out



