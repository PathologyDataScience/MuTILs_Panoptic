import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from torch.jit.annotations import List, Dict, Tuple
from torch import Tensor
import numpy as np
from typing import Iterable
from torchvision.ops import roi_align

from Unet import UNet
from MuTILs_Panoptic.configs.panoptic_model_configs import RegionCellCombination


class MuTILsTransform(nn.Module):
    """
    Performs input transformation before feeding the data to a model.

    The transformations it perform are:
        - rgb normalization (mean subtraction and std division)
        - batch images from list/tuple to a single tensor
    """
    def __init__(
        self,
        image_mean: Iterable = None,
        image_std: Iterable = None,
        ignore_ks: Iterable = None
    ):

        super(MuTILsTransform, self).__init__()
        self.image_mean = image_mean or [0.485, 0.456, 0.406]
        self.image_std = image_std or [0.229, 0.224, 0.225]
        self.ignore_ks = ignore_ks or ['idx', 'roiname']

    def forward(self, data: dict):
        """"""
        # make a copy to avoid modifying it in-place
        # also, refactor so corresponding tensors are concatenated
        data_new = {k: [] for k in data[0].keys()}
        for roi in data:
            for k, v in roi.items():
                data_new[k].append(v)
        data = {k: v for k, v in data_new.items() if k not in self.ignore_ks}
        carryover = {k: v for k, v in data_new.items() if k in self.ignore_ks}

        # normalize the rgbs
        for k in ['highres_rgb', 'lowres_rgb']:
            if k in data:
                for idx, image in enumerate(data[k]):
                    if image.dim() != 4:
                        raise ValueError(
                            "images is expected to be a list of 4d tensors "
                            "of shape [1, C, H, W], got {}".format(image.shape)
                        )
                    image = self.normalize(image)
                    data[k][idx] = image

        # now concatenate batch
        data = {k: torch.cat(v, dim=0) for k, v in data.items()}

        # non-tensors, metadata, etc
        data.update(carryover)

        return data

    def normalize(self, image: Tensor):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[None, :, None, None]) / std[None, :, None, None]


# noinspection PyShadowingNames,DuplicatedCode,LongLine
class MuTILsEvaluator(nn.Module):
    """
    Calculate fit metrics for mutils (eg. DICE).
    """
    def __init__(self, transform: MuTILsTransform = None):

        super(MuTILsEvaluator, self).__init__()

        # for batch concatenation
        self.transform = transform or MuTILsTransform()

        # convenience
        rcc = RegionCellCombination
        self.nucleus_bckg_code = rcc.NUCLEUS_CODES['BACKGROUND']
        self.rregion_codes = RegionCellCombination.RREGION_CODES
        self.rnucleus_codes = RegionCellCombination.RNUCLEUS_CODES
        self.rtils_code = rcc.REGION_CODES['TILS']
        self.rstroma_code = rcc.REGION_CODES['STROMA']
        self.ntils_code1 = rcc.NUCLEUS_CODES['TILsCell']
        self.ntils_code2 = rcc.NUCLEUS_CODES['ActiveTILsCell']

        # by default, stroma is inclusive of TILs (but not vice versa!)
        self.acceptable_region_misclassif = [
            (rcc.REGION_CODES['STROMA'], rcc.REGION_CODES['TILS']),
        ]
        self.acceptable_nucleus_misclassif = []

    def forward(
        self,
        inference: dict,
        truth: dict,
        acceptable_region_misclassif: List[Tuple[int, int]] = None,
        acceptable_nucleus_misclassif: List[Tuple[int, int]] = None,
    ):
        """"""
        acceptable_region_misclassif = (
            acceptable_region_misclassif or self.acceptable_region_misclassif
        )
        acceptable_nucleus_misclassif = (
            acceptable_nucleus_misclassif or self.acceptable_nucleus_misclassif
        )

        # transform the truth
        truth = self.transform(truth)

        # only keep relevant channels for highres mask
        truth['highres_mask'] = truth['highres_mask'][:, 0:2, :, :]

        # crop HPF truth masks
        hpf_mask = self.parse_hpf_mask(
            mask=truth['highres_mask'], bounds=inference['hpf_hres_bounds'],
            roidxs=inference['hpf_roidx'])

        batch_size = truth['highres_mask'].shape[0]
        all_stats = []

        # there maybe multiple hpfs per roi
        n_hpfs = inference['hpf_region_logits'].shape[0]
        topk = n_hpfs // batch_size

        for roidx in range(batch_size):

            stats = {
                'epoch': np.nan, 'slide': np.nan,
                'roiname': truth['roiname'][roidx],
            }
            hstart = roidx * topk
            hend = (roidx + 1) * topk

            # stats for roi regions
            roi_rpred = torch.argmax(
                inference['roi_region_logits'][[roidx], ...], dim=1) + 1
            roi_rtrue = truth['lowres_mask'][[roidx], 0, ...]
            roi_rexcl = roi_rtrue == 0
            rstats = self.get_stats_for_all_classes(
                pred_mask=roi_rpred, true_mask=roi_rtrue,
                exclude_mask=roi_rexcl, rclsmap=self.rregion_codes,
                acceptable_misclassif=acceptable_region_misclassif,
            )
            stats.update({f'roi-regions_{k}': v for k, v in rstats.items()})

            # prep for hpf stats
            hpf_rpred = torch.argmax(
                inference['hpf_region_logits'][hstart:hend, ...], dim=1) + 1
            hpf_rtrue = hpf_mask[hstart:hend, 0, ...]
            hpf_rexcl = hpf_rtrue == 0

            # stats for hpf nuclei. NOTE: this gives the segmentation accuracy
            # without any form of majority voting or any detection/calssific.
            # statistics etc etc
            hpf_npred = torch.argmax(
                inference['hpf_nuclei'][hstart:hend, ...], dim=1) + 1
            hpf_ntrue = hpf_mask[hstart:hend, 1, ...]
            hpf_nexcl = hpf_ntrue == 0
            nstats = self.get_stats_for_all_classes(
                pred_mask=hpf_npred, true_mask=hpf_ntrue,
                exclude_mask=hpf_nexcl, rclsmap=self.rnucleus_codes,
                acceptable_misclassif=acceptable_nucleus_misclassif,
            )
            stats.update(
                {f'hpf-nuclei_{k}': v for k, v in nstats.items()}
            )

            # stats for hpf nuclei WITHOUT region constraint.
            hpf_prenpred = torch.argmax(
                inference['hpf_nuclei_pre'][hstart:hend, ...], dim=1) + 1
            prenstats = self.get_stats_for_all_classes(
                pred_mask=hpf_prenpred, true_mask=hpf_ntrue,
                exclude_mask=hpf_nexcl, rclsmap=self.rnucleus_codes,
                acceptable_misclassif=acceptable_nucleus_misclassif,
            )
            stats.update(
                {f'hpf-prenuclei_{k}': v for k, v in prenstats.items()}
            )

            # Computational TILs Assessment (CTA) score components

            # roi-based CTA (uses tils "regions")
            pfx = 'roi-CTA-score'
            stats[f'{pfx}_numer_true'], stats[f'{pfx}_denom_true'] = \
                self.get_cta_values(roi_rtrue)
            stats[f'{pfx}_numer_pred'], stats[f'{pfx}_denom_pred'] = \
                self.get_cta_values(roi_rpred, rexcl=roi_rexcl)

            # roi-based cta (uses tils nuclei)
            pfx = 'hpf-CTA-score'
            stats[f'{pfx}_numer_true'], stats[f'{pfx}_denom_true'] = \
                self.get_cta_values(hpf_rtrue, nuclei=hpf_ntrue)
            stats[f'{pfx}_numer_pred'], stats[f'{pfx}_denom_pred'] = \
                self.get_cta_values(
                    hpf_rpred, nuclei=hpf_npred,
                    rexcl=hpf_rexcl, nexcl=hpf_nexcl)

            all_stats.append(stats)

        return all_stats

    # noinspection PyUnresolvedReferences
    def get_cta_values(self, regions, nuclei=None, rexcl=None, nexcl=None):
        """Get tils score using dense tils regions OR individual tils."""
        reg = 0 + regions
        nucl = None if nuclei is None else 0 + nuclei

        # incorporate ignore region
        if rexcl is not None:
            reg[rexcl] = 0
        if nexcl is not None:
            nucl[nexcl] = 0

        # get numerator (tils region/cells) and denomenator (stroma region)
        rtils = 0 + (reg == self.rtils_code)
        rstroma = 0 + (reg == self.rstroma_code)
        if nucl is None:
            tils = float(rtils.sum())
        else:
            # ntils = 0 + (nucl == self.ntils_code)
            # tils = float(ntils.sum())
            ntils1 = 0 + (nucl == self.ntils_code1)
            ntils2 = 0 + (nucl == self.ntils_code2)
            tils = float(ntils1.sum() + ntils2.sum())

        return tils, float(rstroma.sum() + rtils.sum())

    def get_stats_for_all_classes(
        self,
        pred_mask,
        true_mask,
        exclude_mask,
        rclsmap: dict,
        acceptable_misclassif=None,
    ):
        """"""
        stats = {}

        # overall pixel accuracy
        overall = self.get_pred_vs_true_stats(
            pred_mask=pred_mask, true_mask=true_mask,
            exclude=exclude_mask, isbin=False)
        stats.update({'OVERALL-' + k: v for k, v in overall.items()})

        # class-by-class accuracy, dice, iou
        nclasses = max(rclsmap.keys())
        for cls in range(1, nclasses + 1):
            pmask = 0 + (pred_mask == cls)
            tmask = 0 + (true_mask == cls)

            # Handle if a misclassification is acceptable (eg ambiguous truth)
            # for eg, if acceptable_misclassif is [(2, 3)], where 2 and 3 are
            # the ground truth codes for stroma and tils regions, then
            # stromal regions are INCLUSIVE of TILs .. i.e. stromal accuracy
            # for this image is accuracy for the combined stroma-tils region
            # class. Note that the opposite is not true .. that is, the
            # "tils-region" accuracy only includes tils in unless
            # acceptable_misclassif also include the reverse case as such
            # [(2, 3), (3, 2)]. If the second element is None, this means that
            # this class cannot be assesed. For example, [(3, None)] means
            # that tils predictions are not necessarily wrong, but they can't
            # be assessed because the truth is ambiguous or not enough tissue
            # of that class if present within this ROI to actually classify it
            # as that class.
            for pair in acceptable_misclassif:
                if cls == pair[0]:
                    change_to = pair[1]
                    if change_to is not None:
                        pmask[pred_mask == pair[1]] = 1
                        tmask[true_mask == pair[1]] = 1
                    else:
                        tmask = pmask = 0 * pmask

            # get stats
            clstats = self.get_pred_vs_true_stats(
                pred_mask=pmask, true_mask=tmask, exclude=exclude_mask,
                isbin=True)
            clsnm = rclsmap[cls]
            stats.update({f'{clsnm}-{k}': v for k, v in clstats.items()})

        return stats

    @staticmethod
    def get_pred_vs_true_stats(pred_mask, true_mask, exclude, isbin=True):
        """"""
        excount = float(torch.sum(0 + exclude))
        pred = 0 + pred_mask
        true = 0 + true_mask
        pred[exclude] = 0  # ignore exclude
        true[exclude] = 0  # ignore exclude

        # pixel accuracy includes background
        intersect = float(torch.sum(0 + (pred == true)) - excount)
        total = float(pred.shape[0] * pred.shape[1] * pred.shape[2] - excount)
        ptst = {'pixel_intersect': intersect, 'pixel_count': total}

        # only pixel accuracy possible without class-by-class slicing
        if not isbin:
            return ptst

        # Intersection and union stats only look at the segmented area
        # Note that this ALREADY ignores excluded regions since these
        # were set to zero in both the prediction & truth at the start
        intersect = float(torch.sum(0 + ((pred + true) == 2)))
        total = float(pred.sum() + true.sum())
        ptst.update({'segm_intersect': intersect, 'segm_sums': total})

        return ptst

    @staticmethod
    def parse_hpf_mask(mask, bounds, roidxs):
        """Concatenate HPF mask. first dim is the roi, last dim is the hpf."""
        hpf_mask = []
        for hno in range(bounds.shape[0]):
            bd = [int(j) for j in bounds[hno, ...].tolist()]
            hpf_mask.append(
                mask[[int(roidxs[hno])], :, bd[1]:bd[3], bd[0]:bd[2]])
        return torch.cat(hpf_mask, dim=0)


# noinspection PyShadowingNames,DuplicatedCode
class MuTILsLoss(nn.Module):
    """
    Calculate loss for MuTILs model.
    """
    def __init__(
        self,
        nclasses_roi: int,
        nclasses_hpf: int,
        transform: MuTILsTransform = None,
        region_weights: Iterable = None,
        nucleus_weights: Iterable = None,
        loss_weights=None,
    ):

        super(MuTILsLoss, self).__init__()

        # these only include the REAL classes (no "exclude")
        self.nclasses_roi = nclasses_roi
        self.nclasses_hpf = nclasses_hpf

        # convenience
        rcc = RegionCellCombination
        self.nucleus_bckg_code = rcc.NUCLEUS_CODES['BACKGROUND']

        # weights for classes
        self.region_weights = region_weights or [0.] + [1.] * self.nclasses_roi
        self.nucleus_weights = \
            nucleus_weights or [0.] + [1.] * self.nclasses_hpf
        self.loss_weights = loss_weights
        self.normalize_weights()

        # for batch concatenation
        self.transform = transform or MuTILsTransform()

        # Note 1:
        # Since this is the init method, the following will get moved to the
        # right device when we move our loss criterion after instantiation.
        # Otherwise we would've had to explicitely specifiy the device

        # Note 2:
        # loss functions -- zero weight is assigned to zeros in the gtruth
        # which represents the EXCLUDE class in the mask. Note that this is
        # is necessary since by default, zero pixels in the gtruth mask are
        # a "class" that is present in the truth but not the model preduction.
        # This is IN ADDITION TO ignoring specific pixels later. Why? Because
        # the model can still make predictions in those ignore pixels, for
        # eg the truth mask says "0", while the model has a probability of
        # 0.65 for tumor at that location. This would be reflected in the tumor
        # channel loss, but we really want to completely ignore that pixel!
        # Hence, we assign a zero-weighted exclude channel for SHAPE
        # compatibility, but get rid of "exclude" pixels ourselves.

        self.xentropy_regions = nn.CrossEntropyLoss(
            weight=torch.as_tensor(self.region_weights, dtype=torch.float32),
            reduction='none',
        )
        self.xentropy_nuclei = nn.CrossEntropyLoss(
            weight=torch.as_tensor(self.nucleus_weights, dtype=torch.float32),
            reduction='none',
        )

    def normalize_weights(self):
        maxwt_r = max(self.region_weights)
        maxwt_n = max(self.nucleus_weights)
        self.region_weights = [j / maxwt_r for j in self.region_weights]
        self.nucleus_weights = [j / maxwt_n for j in self.nucleus_weights]

    # noinspection DuplicatedCode
    def forward(self, inference: dict, truth: dict):
        """"""
        # transform the truth
        truth = self.transform(truth)

        # only keep relevant channels for highres mask
        truth['highres_mask'] = truth['highres_mask'][:, 0:2, :, :]

        # crop HPF truth masks
        hpf_mask = self.parse_hpf_mask(
            mask=truth['highres_mask'], bounds=inference['hpf_hres_bounds'],
            roidxs=inference['hpf_roidx'])

        # calculate pixel-wise losses
        roi_rlossmat = self.xentropy_regions(
            input=self._addexcl(inference['roi_region_logits']),
            target=truth['lowres_mask'][:, 0, ...],
        )
        hpf_rlossmat = self.xentropy_regions(
            input=self._addexcl(inference['hpf_region_logits']),
            target=hpf_mask[:, 0, ...],
        )
        hpf_nlossmat_pre = self.xentropy_nuclei(
            input=self._addexcl(inference['hpf_nuclei_pre']),
            target=hpf_mask[:, 1, ...],
        )
        hpf_nlossmat = self.xentropy_nuclei(
            input=self._addexcl(inference['hpf_nuclei']),
            target=hpf_mask[:, 1, ...],
        )

        # get pixels to ignore
        roi_exclude = truth['lowres_mask'][:, 0, ...] == 0
        hpf_rexclude = hpf_mask[:, 0, ...] == 0
        hpf_nexclude = hpf_mask[:, 1, ...] == 0

        # loss weight is zero for irrelevant pixels
        roi_rlossmat[roi_exclude] = 0.
        hpf_rlossmat[hpf_rexclude] = 0.
        hpf_nlossmat_pre[hpf_nexclude] = 0.
        hpf_nlossmat[hpf_nexclude] = 0.

        # take average over relevant pixels ONLY
        lres_keep = ~roi_exclude
        lres_pixcount = lres_keep.type(torch.int).sum()
        hpf_rkeep = ~hpf_rexclude
        hpf_rpixcount = hpf_rkeep.type(torch.int).sum()
        hpf_nkeep = ~hpf_nexclude
        hpf_npixcount = hpf_nkeep.type(torch.int).sum()

        losses = {
            'roi_regions': self._div(roi_rlossmat.sum(), lres_pixcount),
            'hpf_regions': self._div(hpf_rlossmat.sum(), hpf_rpixcount),
            'hpf_nuclei_pre': self._div(hpf_nlossmat_pre.sum(), hpf_npixcount),
            'hpf_nuclei': self._div(hpf_nlossmat.sum(), hpf_npixcount),
        }

        # process & get aggregate (maybe weighted) MTL loss
        losses = self.process_losses(losses)

        return losses

    @staticmethod
    def _div(numer, denom):
        """Return a loss of zero instead of Inf or DivisionByZeroError."""
        if denom < 1e-8:
            # we multiply to keep same tensor type, incl. preserved grad.
            return 0. * numer
        return numer / denom

    def process_losses(self, losses: dict):
        """Get weighted multi-task loss."""
        lwts = self.loss_weights or {ln: 1.0 for ln in losses.keys()}
        lnames = list(losses.keys())
        losses['all'] = 0.
        for lname in lnames:
            losses['all'] += losses[lname] * lwts[lname]
        losses['all'] /= len(losses) - 1

        return losses

    @staticmethod
    def _addexcl(pred):
        """
        Append a channel of zeros as a first channel (exclude) to pred.
        This will be assigned zero weight anyways.
        """
        shape, device, dtype = pred.shape, pred.device, pred.dtype
        shape = list(shape)
        shape[1] = 1
        excl = torch.zeros(size=shape, device=device, dtype=dtype)

        return torch.cat([excl, pred], dim=1)

    @staticmethod
    def parse_hpf_mask(mask, bounds, roidxs):
        """Concatenate HPF mask."""
        hpf_mask = []
        for hno in range(bounds.shape[0]):
            bd = [int(j) for j in bounds[hno, ...].tolist()]
            hpf_mask.append(
                mask[[int(roidxs[hno])], :, bd[1]:bd[3], bd[0]:bd[2]])
        return torch.cat(hpf_mask, dim=0)


# noinspection PyShadowingNames,DuplicatedCode,PyAttributeOutsideInit
class MuTILs(nn.Module):
    """
    Multi-resolution U-Net based model for computational TILs assessment.
    """

    def __init__(
        self,
        training: bool,
        hpf_mpp: float,  # 20x = 0.5 MPP
        roi_mpp: float,  # 10x = 1.0 MPP
        roi_side: int,  # 256
        hpf_side: int,  # 256
        region_tumor_channel: int,  # no exclude (usually 0 in my code)
        region_stroma_channels: List[int],  # no exclude (1,2 in my code)
        nclasses_r: int,  # no of region classes
        nclasses_n: int,  # no of nuclei classes, incl. bckgrnd
        topk_hpf=1,  # usually 1 if train .. maybe > 1 at inference
        random_topk_hpf=False,  # if false, focus on salient stroma
        spool_overlap=0.25,  # saliency pool overlap
        roi_unet_params: dict = None,
        roi_interm_layer: int = 2,  # After 2nd upconv
        hpf_interm_layer: int = 0,  # bottleneck itself
        hpf_unet_params: dict = None,
        transform: MuTILsTransform = None,
    ):

        super(MuTILs, self).__init__()

        self.training = training
        self.hpf_mpp = hpf_mpp
        self.roi_mpp = roi_mpp
        self.roi_side = int(roi_side)
        self.hpf_side = int(hpf_side)
        self.rtumor_channel = region_tumor_channel
        self.rstroma_channels = region_stroma_channels
        self.nclasses_r = nclasses_r
        self.nclasses_n = nclasses_n
        self.topk_hpf = topk_hpf
        self.random_topk_hpf = random_topk_hpf
        self.spool_overlap = spool_overlap
        self.roi_unet_params = roi_unet_params
        self.hpf_unet_params = hpf_unet_params

        # required keys in forward mode for training vs inference
        self.req_fkeys = ['highres_rgb', 'lowres_rgb']

        # defaults if none given
        unet_params = {
            'wf': 6,  # default: 6
            'padding': True,  # MUST BE TRUE!
            'batch_norm': False,  # default: False
            'up_mode': 'upconv',  # default: upconv
        }

        # roi regions model
        if self.roi_unet_params is None:
            self.roi_unet_params = deepcopy(unet_params)
            self.roi_unet_params.update({
                'in_channels': 3,  # rgb
                'depth': 5,  # default: 5
            })
        self.roi_unet_params.update({
            'n_classes': self.nclasses_r,
            'padding': True,
        })
        self.roi_unet = UNet(**self.roi_unet_params)
        self.rilayer = roi_interm_layer
        self.hilayer = hpf_interm_layer

        # the UNet model downsizes the feature map
        rup = self.roi_unet_params
        hup = self.hpf_unet_params
        assert self.rilayer < rup['depth'] - 1, "interm. is deeper than model!"
        assert self.hilayer < hup['depth'] - 1, "interm. is deeper than model!"
        self.riside = int(roi_side * (2 ** (self.rilayer - rup['depth'] + 1)))
        self.hiside = int(hpf_side * (2 ** (self.hilayer - hup['depth'] + 1)))
        self.richannels = 2 ** (rup['wf'] + rup['depth'] - self.rilayer - 1)
        self.hichannels = 2 ** (hup['wf'] + hup['depth'] - self.hilayer - 1)

        # hpf nuclei model
        if self.hpf_unet_params is None:
            self.hpf_unet_params = deepcopy(unet_params)
            self.hpf_unet_params.update({
                'in_channels': 3,  # rgb
                'depth': 5,  # default: 5
            })
        self.hpf_unet_params.update({
            'n_classes': self.nclasses_n,
            'padding': True,  # must be true!
            'external_concat_layer': self.hilayer,
            'external_concat_nc': self.richannels,  # roi_interm_layer concat
        })
        self.hpf_unet = UNet(**self.hpf_unet_params)

        # normalization & batch concatenation
        self.transform = transform or MuTILsTransform(
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        # The size of our avergae saliency pooling kernel is such that each
        # resultant pixel of the pooled map is represent. of an area from the
        # region ROI corresponding to a PATCH_SIDE high-resolution region.
        # For eg., if our nuclei are assessed at a magnification of 40x, and
        # regions at a magnification of 10x, then a (256, 256) patch at 40x
        # for nuclear assessment is represented by a (64, 64) area in the
        # low-resolution image at 10x
        self.sf = self.roi_mpp / self.hpf_mpp
        self.spool_kernel = int(self.hpf_side / self.sf)

        # When the pooling stride is equal to the kernel, the ROI is divided
        # into non-overlapping HPFs, each corresponding to a PATCH_SIZE nucleus
        # HPFs (eg, a 64x64 non-overlapping 10x tile corresp. to 256x256 HPFs
        # at 40x mag.). The smaller the stride, the more overlap there is
        # in the HPF tiling, and more granular is your localizaion of the most
        # "salient" region for high-resolution analysis.
        self.spool_stride = int(self.spool_kernel * self.spool_overlap)
        assert self.spool_stride > 0, "spool_overlap should be > 0"

        # Learned & fixed weights to map region classes -> nucleus classes
        self.set_compatibility_kernels()

    def forward(self, rois: List[Dict]):
        """"""
        availks = rois[0].keys()
        for k in self.req_fkeys:
            assert k in availks, f"{k} not found in input dict!"

        assert rois[0]['highres_rgb'].ndim == 4, (
            "The first dim MUST be the image index since DataParallel "
            "slices along the first dim. So if, say, you provide an "
            "(C, H, W) tensor, torch DataParallel would slice along the "
            "channels, and send red to one GPU, blue to another, etc!!"
        )

        # normalize & concat batch
        rois = self.transform(rois)

        # region inference
        roi_logits, roi_intermed = self.roi_unet(
            rois['lowres_rgb'], fetch_layers=[self.rilayer])
        roi_intermed = roi_intermed[0]

        # get saliency scores (intra-tumoral stroma probability)
        ig = 'lowres_ignore'
        saliency_scores = self.get_saliency_scores(
            roi_logits=roi_logits,
            lowres_ignore=rois[ig] if ig in rois else None)

        # parse saliency bounds
        hpf_bounds, hpf_score = self.get_salient_bounds(saliency_scores)

        # process all hpfs
        hpf_out = self.process_hpfs(
            hres_rgb=rois['highres_rgb'], roi_logits=roi_logits,
            roi_intermed=roi_intermed, hpf_bounds=hpf_bounds)

        # output regardless of training mode
        inference = {
            'roi_region_logits': roi_logits,
            'roi_saliency_matrix': saliency_scores,
            'hpf_saliency_scores': torch.cat(hpf_score, dim=0),
        }
        inference.update(hpf_out)

        return inference

    def get_saliency_scores(self, roi_logits, lowres_ignore=None):
        """
        FOV intra-tumoral stroma score is just the multiple of its
        tumor score and stroma score (incl. TILs-rich regions of course).
        This ensures that the highest scoring FOVs are those that contain
        BOTH elements in high amounts. Keep in mind that both the tumor and
        stroma scores are normalized to [0, 1] so an FOV in a purely tumor
        region, for example, will have a score of near zero score for stroma
        , and vice versa.
        """
        pooled_tumor = torch.sigmoid(F.avg_pool2d(
            roi_logits[:, self.rtumor_channel, ...],
            kernel_size=self.spool_kernel, stride=self.spool_stride))
        stroma_or_tils = 0. + roi_logits[:, self.rstroma_channels[0], ...]
        for rsc in self.rstroma_channels[1:]:
            stroma_or_tils += roi_logits[:, rsc, ...]
        stroma_or_tils = stroma_or_tils / float(len(self.rstroma_channels))
        pooled_stroma = torch.sigmoid(F.avg_pool2d(
            stroma_or_tils,
            kernel_size=self.spool_kernel, stride=self.spool_stride))

        # The following step needs to happen AFTER FOV pooling!!! Why? To
        # ensure that we favor detection of ADJACENT tumor and stroma .. it's
        # not about the per-pixel scores, but the aggregation within a field.
        saliency_scores = pooled_tumor * pooled_stroma

        # Maybe ignore some regions from saliency.
        # This is not only useful during training, but also potentially
        # during inference in case we want to ignore some areas
        # IMPORTANT: lowres_ignore is assumed to be in the range [0, 1].
        # Usually it's just a float32 converted from a boolean mask
        if lowres_ignore is not None:
            pooled_ignore = F.avg_pool2d(
                lowres_ignore[:, 0, :, :],
                kernel_size=self.spool_kernel, stride=self.spool_stride)
            saliency_scores = saliency_scores * (1. - pooled_ignore)

        return saliency_scores

    def process_hpfs(self, hres_rgb, roi_logits, roi_intermed, hpf_bounds):
        """Process all HPFs in batch."""

        # Crop hpfs from the intermediate lowres roi convolutional map
        sf = self.riside / self.roi_side
        hpf_roi_intermed = roi_align(
            roi_intermed, boxes=[j * sf for j in hpf_bounds],
            output_size=[self.hiside, self.hiside])

        # crop hpf rgbs
        hpf_rgbs = roi_align(
            hres_rgb, boxes=[j * self.sf for j in hpf_bounds],
            output_size=[self.hpf_side, self.hpf_side])

        # crop hpf & bilinearly upsample region logits
        hpf_region_logits = roi_align(
            roi_logits, boxes=hpf_bounds,
            output_size=[self.hpf_side, self.hpf_side])

        # get hpf nucleus predictions
        hpf_nuclei_pre = self.hpf_unet(hpf_rgbs, cx=hpf_roi_intermed)

        # Use region logits & compatibility kernels to obtain pixel-wise
        # nucleus class scores that are compatible with regions
        hpf_attention = self.get_nucleus_attention_map(hpf_region_logits)

        # use attention map to enforce biological compatibility
        hpf_nuclei = hpf_nuclei_pre * hpf_attention

        # hpfs are first dim .. all concatenated together
        lowres_bounds = torch.cat(hpf_bounds, 0)

        # which roi was each hpf taken from?
        hpf_roidx = []
        for j in range(roi_intermed.shape[0]):
            hpf_roidx.extend([j] * self.topk_hpf)
        hpf_roidx = torch.as_tensor(
            hpf_roidx, device=roi_intermed.device, dtype=torch.int)

        return {
            'hpf_roidx': hpf_roidx,
            'hpf_lres_bounds': lowres_bounds,
            'hpf_hres_bounds': lowres_bounds * self.sf,
            'hpf_region_logits': hpf_region_logits,
            'hpf_nuclei_pre': hpf_nuclei_pre,
            'hpf_nuclei': hpf_nuclei,
        }

    def get_nucleus_attention_map(self, hpf_region_logits):
        """
        Use learned kernels to use region logits to get an attention map for
        each of the nuclei classes. The fixed kernels are used to mask/bias the
        results to ensure compatibility b/n regions and nuclei in a
        biologically-sensible manner.
        """
        # mask the learned kernel using biol. compatibility kernel
        ker = self.learned_kernel * self.fixed_kernel

        # Combination of regions to produce the attention map for this nucleus
        # class (using a simple linear model). NOTE: I tried doing this
        # using a convolutional block and it didn't work very well, so I'm
        # not treating each pixel independently.
        nucleus_logits = hpf_region_logits.permute(0, 3, 2, 1)
        nucleus_logits = F.linear(nucleus_logits, weight=ker)
        nucleus_logits = nucleus_logits.permute(0, 3, 2, 1)

        return nucleus_logits

    def set_compatibility_kernels(self):
        """
        Get kernel to linearly combine region logits to obtain a particular
        nucleus class logits.
        """
        # this is a fixed tensor -- no gradient, not learned
        fker = torch.tensor(
            RegionCellCombination.allowed_regions,
            dtype=torch.float32, requires_grad=False)
        self.fixed_kernel = nn.parameter.Parameter(fker, requires_grad=False)

        # this is a learned variable -- has gradient
        lker = torch.rand(
            (self.nclasses_n, self.nclasses_r),
            dtype=torch.float32, requires_grad=True)
        self.learned_kernel = nn.parameter.Parameter(lker, requires_grad=True)

    def get_salient_bounds(self, pooled_saliency):
        """
        Sort from max to min saliency and get corresponding locations
        relative to the LOW RESOLUTION rgb.
        """
        device, dtype = pooled_saliency.device, pooled_saliency.dtype
        batch_size, h, w = pooled_saliency.shape
        bigh = bigw = self.roi_side
        sf_h = bigh / h
        sf_w = bigw / w
        side = self.spool_kernel // 2
        hpf_bounds = []
        hpf_slogit = []
        for imno in range(batch_size):

            # from highest to lowest saliency
            psals = pooled_saliency[imno, ...].reshape(-1)
            if self.random_topk_hpf:
                hpf_idxs = torch.randint(0, len(psals), size=(self.topk_hpf,))
            else:
                hpf_idxs = torch.argsort(psals, descending=True)[:self.topk_hpf]
            hpf_slogit.append(psals[hpf_idxs])

            # append bounds relative to region rgb
            imlocs = []
            for idx in hpf_idxs:

                cy = float(idx) // h
                cx = float(idx) - cy * h
                cx += 0.5  # pixel center
                cy += 0.5
                center_x = int(cx * sf_w)
                center_y = int(cy * sf_h)

                # shift if near edge to avoid having to pad
                xmin = center_x - side
                xmax = center_x + side
                ymin = center_y - side
                ymax = center_y + side
                if xmin < 0:
                    xmax += -xmin
                    xmin = 0
                if xmax > bigw:
                    xmin -= (xmax - bigw)
                    xmax = bigw
                if ymin < 0:
                    ymax += -ymin
                    ymin = 0
                if ymax > bigh:
                    ymin -= (ymax - bigh)
                    ymax = bigh

                # append bounds
                imlocs.append(torch.as_tensor(
                    [xmin, ymin, xmax, ymax], device=device, dtype=dtype))

            hpf_bounds.append(torch.cat([j[None, ...] for j in imlocs], dim=0))

        return hpf_bounds, hpf_slogit


# =============================================================================

if __name__ == '__main__':
    pass
