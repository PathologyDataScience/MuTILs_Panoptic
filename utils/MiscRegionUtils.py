import os
from os.path import join as opj
import torch
import numpy as np
from pandas import DataFrame
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import binary_opening, disk
from scipy.ndimage import distance_transform_edt
from typing import Dict
from histomicstk.annotations_and_masks.annotation_and_mask_utils import \
    np_vec_no_jit_iou
from scipy.optimize import linear_sum_assignment

from MuTILs_Panoptic.utils.GeneralUtils import (
    load_configs, reverse_dict, _div, abserr
)
from MuTILs_Panoptic.mutils_panoptic.MuTILs import MuTILs
from MuTILs_Panoptic.utils.TorchUtils import load_torch_model, t2np
from MuTILs_Panoptic.configs.panoptic_model_configs import MuTILsParams
from MuTILs_Panoptic.utils.torchvision_transforms import PILToTensor

# =============================================================================
# Constants

# map np dtypes to vips
NP2VIPS_DTYPES = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}

VIPS2NP_DTYPES = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# =============================================================================


def numpy2vips(arr):
    """numpy array to vips image"""
    import pyvips

    height, width, bands = arr.shape
    linear = arr.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(
        linear.data, width, height,
        bands,
        NP2VIPS_DTYPES[str(arr.dtype)],
    )
    return vi


def vips2numpy(vim):
    """vips image to numpy array."""
    return np.ndarray(
        buffer=vim.write_to_memory(),
        dtype=VIPS2NP_DTYPES[vim.format],
        shape=[vim.height, vim.width, vim.bands]
    )


def pil2tensor(*args, **kwargs):
    """A functional wrapper areound PILToTensor"""
    return PILToTensor()(*args, **kwargs)


def get_configured_logger(logdir: str, prefix="", toscreen=True, tofile=True):
    """"""
    import logging
    import datetime
    assert any((toscreen, tofile)), "must log to screen and/or to file!"
    now = str(datetime.datetime.now()).replace(' ', '_')
    logfile = opj(logdir, f'{prefix}_{now}.log')
    handlers = []
    if toscreen:
        handlers.append(logging.StreamHandler())  # print to console
    if tofile:
        handlers.append(logging.FileHandler(logfile))  # save to log file
    # noinspection PyArgumentList
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    logger = logging.getLogger("")
    return logger


def load_region_configs(configs_path, warn=True):
    """"""
    if os.path.exists(configs_path):
        if warn:
            input(f"Loading existing configs from: {configs_path}. Continue?")
        cfg = load_configs(configs_path=configs_path)
    else:
        print("Loading default configs")
        import MuTILs_Panoptic.configs.panoptic_model_configs as cfg
    return cfg


def load_trained_mutils_model(ckpt_path: str, mtp: MuTILsParams):
    """"""
    # GPU vs CPU
    iscude = torch.cuda.is_available()
    device = torch.device('cuda') if iscude else torch.device('cpu')

    # init model
    model = MuTILs(**mtp.model_params)
    model.to(device)

    # load weights
    ckpt = load_torch_model(checkpoint_path=ckpt_path, model=model)
    model = ckpt['model']
    model.eval()

    return model


def logits2preds(logits, return_probabs=False, return_aggregate=False):
    """
    Convert logits to argmaxed predictions. If the first axis of logits
    has more than one image, it is assumed that these are multiple
    augmentations of the same image and their probabilities are aggregated.

    Parameters
    ----------
    logits: torch.Tensor
        a tensor of shape (n_augmentations, n_classes, m, n)
    return_probabs: bool
        also return prediction probabilities?
    return_aggregate: bool
        also return the mean of the per-pixel probab? This
        serves as a proxy for overall model confidence for this particular
        roi/hpf.

    Returns
    -------
    np.array
        argmaxed predictions, a uint8 np array of shape (m, n)
    np.array, optional
        prediction probabs, an np array of shape (n_classes, m, n)

    """
    assert logits.ndim == 4, "first axis must be the image index"
    if logits.shape[0] == 1:
        agg_logits = logits[0, ...]
    else:
        agg_logits = torch.softmax(logits, 1)
        agg_logits = agg_logits.sum(0)

    if return_probabs or return_aggregate:
        agg_logits = torch.softmax(agg_logits, 0)

    preds = t2np(agg_logits.argmax(0) + 1)

    out = [np.uint8(preds)]
    if return_probabs:
        out.append(t2np(agg_logits))
    if return_aggregate:
        out.append(float(t2np(agg_logits).max(0).mean()))

    return out


def _pixsum(msk: np.ndarray, code: int) -> int:
    """"""
    return int(np.sum(0 + (msk == code)))


def summarize_region_mask(
    mask: np.ndarray, rcd: Dict[str, int], prefix="pixelCount_"
) -> Dict[str, int]:
    """Summarize no. of pixels from various classes in region mask."""
    return {f"{prefix}{r}": _pixsum(mask, cd) for r, cd in rcd.items()}


def get_objects_from_binmask(
    binmask,
    *,
    open_first=True,
    selem=None,
    mindist=5,
    use_watershed=True,
    minpixels=10,
    maxpixels=None,
    _return_codes=True,  # unique object codes
):
    """
    Given a binary mask, get object mask where pixel values encode
    individual object membership.
    """
    # binary opening helps isolate touching objects
    if open_first:
        binmask = binary_opening(
            binmask, footprint=selem if selem is not None else np.ones((5, 5))
        )

    # watershed also helps isolate touching objects
    if use_watershed:
        # compute the exact Euclidean distance from every binary
        # pixel to the nearest zero pixel, then find peaks in this
        # distance map
        binmask = binmask * 255  # scipy prefers 255
        D = ndimage.distance_transform_edt(binmask)
        localmax = np.zeros(binmask.shape, dtype=bool)
        peak_idxs = peak_local_max(D, min_distance=mindist, labels=binmask)
        localmax[peak_idxs[:, 0], peak_idxs[:, 1]] = True

        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        markers = ndimage.label(localmax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=binmask, watershed_line=True)

    else:
        labels, _ = ndimage.label(binmask, structure=np.ones((3, 3)))

    # filter artifacts
    unique, counts = np.unique(labels, return_counts=True)
    if unique[0] == 0:
        unique = unique[1:]
        counts = counts[1:]
    keep = counts >= minpixels
    if maxpixels is not None:
        keep2 = counts < maxpixels
        keep = keep & keep2
    discard_vals = unique[~keep]
    labels[np.isin(labels, discard_vals)] = 0

    if not _return_codes:
        return labels

    return labels, unique[keep]


def get_region_within_x_pixels(
    center_mask: np.ndarray,
    surround_mask: np.ndarray,
    max_dist: int = 32,
    min_ref_pixels: int = None,
):
    """
    Get mask of regions from surround_mask within x pixels from center_mask.

    Parameters
    ----------
    center_mask: np.ndarray
        Binary mask of reference region (eg tumor)
    surround_mask: np.ndarray
        Binary mask of surround region (eg stroma).
        Must be same shape as `center_mask`
    max_dist: int
        Max distance in pixels from reference region.
    min_ref_pixels: int, default is None
        Min no of pixels to consider a reference region. If none, all
        reference regions are analyzed.

    Returns
    -------
    np.ndarray
        mask of surround region within max_dist of reference region
        for example, this could be the peri-tumoral stroma. It has the same
        shape as `center_mask`.

    """
    # open to remove small objects
    if min_ref_pixels is not None:
        center_mask = binary_opening(center_mask, footprint=disk(min_ref_pixels))
    # find proximity of each pixel from center regions (our reference)
    shape = center_mask.shape
    maxd = np.sqrt(np.sum([side ** 2 for side in shape]))
    if np.sum(0 + center_mask) > 5:
        distance_from_ref = distance_transform_edt(~center_mask)
        distance_from_ref[distance_from_ref > max_dist] = maxd
    else:
        distance_from_ref = maxd + np.zeros(shape, dtype=np.float32)
    # get mask of surround within x pixels from the reference region
    surround = distance_from_ref < maxd
    surround[~surround_mask] = False

    return surround


def summarize_nuclei_mask(obj2lbl: dict, ncd: dict):
    """Summarize HPF nuclei mask."""
    nst = "nNuclei"
    realn = [k for k in ncd.keys() if k not in ['EXCLUDE', 'BACKGROUND']]
    lbl2objs = {lbl: [] for lbl in realn}
    lbl2objs.update(reverse_dict(obj2lbl, preserve=True))
    out = {f"{nst}_all": len(obj2lbl)}
    out.update({
        f"{nst}_{cls}": len(lbl2objs[cls]) for cls in ncd if cls in lbl2objs
    })
    return out


def _aggregate_semsegm_stats(df: DataFrame, colpfx='', supermap=None):
    """"""
    # # Must be passed explicitely!!
    # supermap = {
    #     'NORMAL': 'TUMOR',
    #     'JUNK': 'OTHER',
    #     'BLOOD': 'OTHER',
    #     'WHITE': 'OTHER',
    # }
    if supermap is not None:
        discard = []
        for cat, supercat in supermap.items():
            for sfx in [
                'pixel_intersect', 'pixel_count', 'segm_intersect', 'segm_sums'
            ]:
                df.loc[:, f'roi-regions_{supercat}-{sfx}'] += (
                    df.loc[:, f'roi-regions_{cat}-{sfx}']
                )
                discard.append(f'roi-regions_{cat}-{sfx}')
        df = df.loc[:, [c for c in df.columns if c not in discard]]

    agg = {}
    # calculate overall accuracy
    for intersect in [j for j in df.columns if '-pixel_intersect' in j]:
        total = intersect.replace('_intersect', '_count')
        agg[intersect] = df.loc[:, intersect].sum()
        agg[total] = df.loc[:, total].sum()
        agg[intersect.replace('_intersect', '_accuracy')] = \
            _div(agg[intersect], agg[total])
    # calculate iou and dice
    for intersect in [j for j in df.columns if '-segm_intersect' in j]:
        total = intersect.replace('_intersect', '_sums')
        agg[intersect] = df.loc[:, intersect].sum()
        agg[total] = df.loc[:, total].sum()
        agg[intersect.replace('_intersect', '_iou')] = \
            _div(agg[intersect], agg[total] - agg[intersect])
        agg[intersect.replace('_intersect', '_dice')] = \
            _div(2. * agg[intersect], agg[total])
    # calculate tils score
    for pf in ['roi', 'hpf']:
        pfx = f'{colpfx}{pf}-CTA-score'
        for sfx in ['true', 'pred']:
            numer = f'{pfx}_numer_{sfx}'
            denom = f'{pfx}_denom_{sfx}'
            agg[numer] = df.loc[:, numer].sum()
            agg[denom] = df.loc[:, denom].sum()
            agg[f'{pfx}_{sfx}'] = _div(agg[numer], agg[denom])
        agg[f'{pfx}_abserror'] = abserr(agg[f'{pfx}_pred'], agg[f'{pfx}_true'])

    return agg

def map_bboxes_using_hungarian_algorithm(bboxes1, bboxes2, min_iou=1e-4):
    """Map bounding boxes using hungarian algorithm.

    Adapted from Lee A.D. Cooper.

    Parameters
    ----------
    bboxes1 : numpy array
        columns correspond to xmin, ymin, xmax, ymax

    bboxes2 : numpy array
        columns correspond to xmin, ymin, xmax, ymax

    min_iou : float
        minumum iou to match two bboxes to match to each other

    Returns
    -------
    np.array
        matched indices relative to x1, y1

    np.array
        matched indices relative to x2, y2, correspond to the first output

    np.array
        unmatched indices relative to x1, y1

    np.array
        unmatched indices relative to x2, y2

    """
    # generate cost matrix for mapping cells from user to anchors
    max_cost = 1 - min_iou
    costs = 1 - np_vec_no_jit_iou(bboxes1=bboxes1, bboxes2=bboxes2)
    costs[costs > max_cost] = 99.

    # perform hungarian algorithm mapping
    source, target = linear_sum_assignment(costs)

    # discard mappings that are non-allowable
    allowable = costs[source, target] <= max_cost
    source = source[allowable]
    target = target[allowable]

    # find indices of unmatched
    def _find_unmatched(coords, matched):
        potential = np.arange(coords.shape[0])
        return potential[~np.in1d(potential, matched)]
    unmatched1 = _find_unmatched(bboxes1, source)
    unmatched2 = _find_unmatched(bboxes2, target)

    return source, target, unmatched1, unmatched2


