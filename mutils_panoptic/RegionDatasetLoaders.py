from collections import Counter, defaultdict
import os
from os.path import join as opj
from copy import deepcopy
from pandas import DataFrame, read_csv, Series
import numpy as np
from imageio import imread
from PIL import Image
import torch
import random
from glob import glob
from torchvision.transforms.functional import InterpolationMode
from typing import Callable, Union

from MuTILs_Panoptic.utils.TorchUtils import transform_dlinput
import MuTILs_Panoptic.utils.torchvision_transforms as tvdt
from MuTILs_Panoptic.configs.region_model_configs import RegionCellCombination


# noinspection PyPep8Naming
def get_fov_bounds(
        height, width, fov_dims=(256, 256), shift_step=0,
        fix_size_at_edge=True):
    """
    Given an image, this get the bounds to cut it into smaller FOV's.
    Args:
    ------
        M, N - integers - image height and width
        fov_dims - x, y tuple, size of FOV's
        shift_step - if you'd like some overlap between FOV's, int
    Returns:
    --------
        FOV_bounds - list of lists, each corresponding to one FOV,
                     in the form of [rowmin, rowmax, colmin, colmax]
    """
    # sanity checks
    assert (fov_dims[0] <= height and fov_dims[1] <= width), \
        "FOV dims must be less than image dims"
    assert (shift_step < fov_dims[0] and shift_step < fov_dims[1]), \
        "shift step must be less than FOV dims"

    # Needed dimensions
    m, n = fov_dims

    # get the bounds of of the sub-images
    Bounds_m = list(range(0, height, m))
    Bounds_n = list(range(0, width, n))

    # Add the edge
    if Bounds_m[len(Bounds_m) - 1] < height:
        Bounds_m.append(height)
    if Bounds_n[len(Bounds_n) - 1] < width:
        Bounds_n.append(width)

    # Get min and max bounds
    Bounds_m_min = Bounds_m[:-1]
    Bounds_m_max = Bounds_m[1:]
    Bounds_n_min = Bounds_n[:-1]
    Bounds_n_max = Bounds_n[1:]

    # Fix final minimum coordinate
    if fix_size_at_edge:
        if Bounds_m_min[-1] > (height - m):
            Bounds_m_min[-1] = np.max([0, height - m])
        if Bounds_n_min[-1] > (width - n):
            Bounds_n_min[-1] = np.max([0, width - n])

    # noinspection PyPep8Naming,DuplicatedCode
    def _AppendShifted(Bounds, MaxShift):
        """Appends a shifted version of the bounds"""
        if shift_step > 0:
            Shifts = list(range(shift_step, MaxShift, shift_step))
            for coordidx in range(len(Bounds) - 2):
                for shift in Shifts:
                    Bounds.append((Bounds[coordidx] + shift))
        return Bounds

    # Append vertical shifts (along the m axis)
    Bounds_m_min = _AppendShifted(Bounds_m_min, m - 1)
    Bounds_m_max = _AppendShifted(Bounds_m_max, m - 1)

    # Append horizontal shifts (along the n axis)
    Bounds_n_min = _AppendShifted(Bounds_n_min, n - 1)
    Bounds_n_max = _AppendShifted(Bounds_n_max, n - 1)

    # Initialize FOV coordinate output matrix
    num_m = len(Bounds_m_min)
    num_n = len(Bounds_n_min)
    FOV_bounds = []

    # Get row, col coordinates of all FOVs
    fovidx = 0
    for fov_m in range(num_m):
        for fov_n in range(num_n):
            FOV_bounds.append(
                [Bounds_m_min[fov_m], Bounds_m_max[fov_m],
                 Bounds_n_min[fov_n], Bounds_n_max[fov_n]])
            fovidx += 1

    return FOV_bounds


# noinspection PyShadowingNames
def get_cv_fold_slides(train_test_splits_path, fold):
    train_slides = read_csv(
            opj(train_test_splits_path, f'fold_{fold}_train.csv')
        ).loc[:, 'slide_name'].to_list()
    test_slides = read_csv(
            opj(train_test_splits_path, f'fold_{fold}_test.csv')
        ).loc[:, 'slide_name'].to_list()
    return train_slides, test_slides


# =============================================================================


# noinspection PyShadowingNames,PyAttributeOutsideInit
class MuTILsDataset(object):
    def __init__(
            self, root: str, slides=None,
            # training or inference mode?
            training=True,
            # slide/data balancing
            force_nuclear_edges=True,  # True is faithful to UNet paper
            strong_slide_balance=False,  # False is best
            strong_class_balance=True,  # True is best
            # formats, sizing, magnification
            float16=False,
            original_mpp=0.25,  # 40x = 0.25 MPP
            hpf_mpp=0.5,
            roi_mpp=1.0,
            original_side=1024,
            roi_side=512,
            # augmentation
            crop_iscentral=False,
            scale_augm_ratio=0.1,
            transforms: Union[str, Callable] = 'defaults',
            # regions that have very noisy nuclei
            meh_regions=None,
            _shuf=True,  # internal
    ):

        # sanity check
        assert transforms is not None
        if not training:
            # warn('Defaulted to no augmentation since train is False!')
            scale_augm_ratio = None
            crop_iscentral = True

        self.root = root
        self.force_nuclear_edges = force_nuclear_edges
        # handling roi size & class imbalance
        self.strong_slide_balance = strong_slide_balance
        self.strong_class_balance = strong_class_balance
        # resolutions & distances
        self.original_mpp = original_mpp
        self.roi_mpp = roi_mpp
        self.hpf_mpp = hpf_mpp
        self.original_side = original_side
        self.roi_side = roi_side
        # my past experience is float16 reduces accuracy .. a last option
        self.float16 = float16
        self.training = training
        # setting _shuf to true ensures vis. has multiple slides per batch
        self._shuf = _shuf

        # augmentation transforms
        self.scale_augm_ratio = scale_augm_ratio
        if transforms == 'defaults':
            if self.training:
                self.transforms = transform_dlinput(
                    tlist=['hflip', 'augment_stain'],
                    make_tensor=False, flip_prob=0.5,
                    augment_stain_sigma1=0.5, augment_stain_sigma2=0.5)
            else:
                self.transforms = transform_dlinput(tlist=None, make_tensor=False)  # noqa
        else:
            self.transforms = transforms

        # resizes ROIs to high resolution magnif (same as HPF)
        sf = self.original_mpp / self.hpf_mpp
        if np.abs(sf - 1.) > 1e-6:
            self.hres_roiside = int(self.original_side * sf)
            self.orig2hres_roirgb = tvdt.Resize(
                self.hres_roiside, interpolation=InterpolationMode.BILINEAR)
            self.orig2hres_roimsk = tvdt.Resize(
                self.hres_roiside, interpolation=InterpolationMode.NEAREST)
        else:
            self.hres_roiside = self.original_side
            self.orig2hres_roirgb = None
            self.orig2hres_roimsk = None

        # Crops ROIs at hpf mpp, possibly with some jitter for scale augment.
        roi2hpf_sf = roi_mpp / hpf_mpp
        rcside = int(self.roi_side * roi2hpf_sf)
        pm = int(rcside * self.scale_augm_ratio) if self.training else None
        self.roi_cropper = tvdt.Cropper(size=rcside, plusminus=pm, iscentral=crop_iscentral)  # noqa
        self.cropped2hres_rgb = tvdt.Resize(rcside, interpolation=InterpolationMode.BILINEAR)  # noqa
        self.cropped2hres_msk = tvdt.Resize(rcside, interpolation=InterpolationMode.NEAREST)  # noqa

        # resizes ROIs to low resolution -- this happens AFTER cropping/augm.
        if np.abs(roi2hpf_sf - 1.) > 1e-6:
            side = int(rcside / roi2hpf_sf)
            self.hres2lres_roirgb = tvdt.Resize(side, interpolation=InterpolationMode.BILINEAR)  # noqa
            self.hres2lres_roimsk = tvdt.Resize(side, interpolation=InterpolationMode.NEAREST)  # noqa
        else:
            self.hres2lres_roirgb = None
            self.hres2lres_roimsk = None

        # converts the pillow images to tensors
        self._tensorize = True  # internal
        self.pil2tensor = tvdt.PILToTensor(float16=self.float16)

        # label codes for regions nuclei, etc
        self.set_labelmaps()

        # regions that have very noisy nuclei.
        # NOTE: This only applies to ACS .. why? In the CPS2 cohort, the
        # "exclude" regions were often labeled as "other", so this adds noise.
        # self.meh_regions = meh_regions or ['OTHER', 'JUNK', 'BLOOD', 'WHITE']
        self.meh_regions = meh_regions or ['OTHER', 'WHITE']
        self.meh_region_codes = [self.region_codes[j] for j in self.meh_regions]

        # set names of rois, slides, etc
        self.set_roinames_etc(slides=slides)

        # assign a weight to rois based on class abundance & other criteria
        self.set_roiweights()

    def set_labelmaps(self):
        self.region_codes = deepcopy(RegionCellCombination.REGION_CODES)
        self.rregion_codes = deepcopy(RegionCellCombination.RREGION_CODES)
        self.nucleus_codes = deepcopy(RegionCellCombination.NUCLEUS_CODES)
        self.rnucleus_codes = deepcopy(RegionCellCombination.RNUCLEUS_CODES)

        # convenience
        self.regions = list(self.region_codes.keys())
        self.nonexclude_regions = deepcopy(self.regions)
        self.nonexclude_regions.remove('EXCLUDE')
        self.nuclei = list(self.nucleus_codes.keys())
        self.nonexclude_nuclei = deepcopy(self.nuclei)
        self.nonexclude_nuclei.remove('EXCLUDE')

    def get_slide_roidxs(self):
        """Return the indices of rois associated with each slide."""
        slide_roidxs = defaultdict(list)
        for idx, roin in enumerate(self.roinames):
            slide_roidxs[self._r2s(roin)].append(idx)
        return slide_roidxs

    def get_roidxs_ordered_by_slide(self):
        """Get roi indices list, ordered by slide."""
        slide_roidxs = self.get_slide_roidxs()
        order = []
        for slide in self.slides:
            order.extend(slide_roidxs[slide])
        return order

    @staticmethod
    def _r2s(roiname):
        """roiname to slide name."""
        return roiname.split('_')[0]

    def _root(self, roiname):
        return self.tcga_root if roiname.startswith('TCGA') else self.acs_root

    def _get_roinames(self):
        """regions of interest (roi) -- multiple per slide."""
        tcga_roinames = glob(opj(self.tcga_root, 'masks', '*.png'))
        acs_roinames = glob(opj(self.acs_root, 'masks', '*.png'))
        if not self._shuf:
            tcga_roinames.sort()
            acs_roinames.sort()
        roinames = [os.path.basename(j) for j in tcga_roinames + acs_roinames]
        if self._shuf:
            random.shuffle(roinames)
        return roinames

    # noinspection DuplicatedCode
    def set_roinames_etc(self, slides: list):
        """Set names of ROIs, slides etc."""
        self.tcga_root = opj(self.root, 'tcga')
        self.acs_root = opj(self.root, 'acs')
        roinames = self._get_roinames()
        if slides is not None:
            roinames = [j for j in roinames if self._r2s(j) in slides]
        self.roinames = roinames
        self.slides = list({self._r2s(j): None for j in roinames}.keys())

    def _get_or_read_roi_summary(self, get_nuclei=False):
        """Get the summary dataframe"""
        try:
            region_summary = read_csv(
                opj(self.root, 'region_summary.csv'), index_col=0)
            region_summary = region_summary.loc[self.roinames, :]
            if get_nuclei:
                nuclei_summary = read_csv(
                    opj(self.root, 'nuclei_summary.csv'), index_col=0)
                nuclei_summary = nuclei_summary.loc[self.roinames, :]
            else:
                nuclei_summary = None

            return region_summary, nuclei_summary

        except FileNotFoundError:
            print('Region and/or nuclei summary not found! Getting one now ..')

        # we get region/nuclei summary for ALL roinames from all folds
        roinames = self._get_roinames()

        region_summary = []
        nuclei_summary = [] if get_nuclei else None
        nrois = len(roinames)

        for ridx, roiname in enumerate(roinames):

            if ridx % 10 == 0:
                print('Getting count summary: roi %d of %d' % (ridx, nrois))

            mask = imread(opj(self._root(roiname), 'masks', roiname))
            # regions
            regs, reg_counts = np.unique(mask[..., 0], return_counts=True)
            to_append = {reg: 1 for reg in self.regions}
            to_append.update({
                self.rregion_codes[r]: rc for r, rc in zip(regs, reg_counts)})
            region_summary.append(to_append)
            # nuclei
            if get_nuclei:
                nucs, nuc_cnts = np.unique(mask[..., 1], return_counts=True)
                to_append = {nucl: 1 for nucl in self.nuclei}
                to_append.update({
                    self.rnucleus_codes[n]: nc for n, nc in zip(nucs, nuc_cnts)})
                nuclei_summary.append(to_append)

        region_summary = DataFrame.from_records(region_summary)
        region_summary.index = roinames
        region_summary[region_summary.isnull()] = 0
        region_summary.to_csv(opj(self.root, 'region_summary.csv'))

        if get_nuclei:
            nuclei_summary = DataFrame.from_records(nuclei_summary)
            nuclei_summary.index = roinames
            nuclei_summary[nuclei_summary.isnull()] = 0
            nuclei_summary.to_csv(opj(self.root, 'nuclei_summary.csv'))

        # now after we saved the df, restrict to rois of interest
        region_summary = region_summary.loc[self.roinames, :]
        if get_nuclei:
            nuclei_summary = nuclei_summary.loc[self.roinames, :]

        return region_summary, nuclei_summary

    def set_roiweights(self):
        """Get weight to assign to various rois based on class abundance."""
        if self.training:
            self._set_roiweights_training()
        else:
            self._set_roiweights_testing()

    def _set_roiweights_testing(self):
        """No weights necessary since inference mode."""
        roi_slides = [self._r2s(r) for r in self.roinames]
        self.slide_roi_counts = Counter(roi_slides)
        self.region_weights = None
        self.ordered_region_weights = None
        self.roi_weights = None

    # noinspection DuplicatedCode
    def _set_roiweights_training(self):
        """Training weights for for class balancing."""
        region_summary, _ = self._get_or_read_roi_summary()
        considerations = DataFrame(index=self.roinames)

        # Weights are inversely proportional to the number of rois
        # that represent a single slide in our dataset. Some slides
        # are represented by more ROIs, and we don't want these to
        # dominate and skew the training
        roi_slides = [self._r2s(r) for r in self.roinames]
        self.slide_roi_counts = Counter(roi_slides)
        totc = len(self.roinames)
        # VISUAL INSPECTION: Weak slide balancing is best
        if self.strong_slide_balance:
            uncommon_slides = np.array([
                totc / self.slide_roi_counts[j] for j in roi_slides])
        else:
            uncommon_slides = np.array([
                (totc - self.slide_roi_counts[j]) / totc for j in roi_slides])
            uncommon_slides += 1
        considerations.loc[:, 'by_slide'] = uncommon_slides / np.sum(
            uncommon_slides)

        # Weights are inversely proportional to the relative frequency
        # of region class, with the exception of 'EXCLUDE' class which
        # is assigned zero weight
        regcounts = dict(region_summary.loc[:, self.nonexclude_regions].sum())
        totc = sum(regcounts.values())
        # VISUAL INSPECTION: Strong class balancing is best
        if self.strong_class_balance:
            regwts = {k: totc / v for k, v in regcounts.items()}
        else:
            regwts = {k: (totc - v) / totc for k, v in regcounts.items()}
        regwts['EXCLUDE'] = 0.
        totwt = sum(regwts.values())
        self.region_weights = {k: v / totwt for k, v in regwts.items()}
        self.ordered_region_weights = [
            self.region_weights[self.rregion_codes[i]]
            for i in range(len(self.regions))
        ]
        tmp_rs = region_summary.copy()
        for cls, clswt in self.region_weights.items():
            tmp_rs.loc[:, cls] *= clswt
        tmp_rs = tmp_rs.sum(1)
        considerations.loc[:, 'by_regions'] = tmp_rs / tmp_rs.sum()

        # overall weight -- must be in the same order as self.roinames
        # so do NOT convert to dict (keep as a series!)
        self.roi_weights = Series(1., index=list(considerations.index))
        for col in considerations.columns:
            self.roi_weights *= considerations.loc[:, col]
        self.roi_weights /= self.roi_weights.sum()

    def load_roi_to_hpf_mpp(self, roiname):
        """Load roi as it is saved on disk & resize to HPF MPP."""
        # read im, mask
        root = self._root(roiname)
        rgb = Image.fromarray(imread(opj(root, 'rgbs', roiname)))
        mask = imread(opj(root, 'masks', roiname))

        # Some regions dont contribute to nucleus truth or loss
        # This only applies to CPS-II with noisy truth!
        if not roiname.startswith('TCGA'):
            meh = np.in1d(mask[..., 0], self.meh_region_codes).reshape(
                mask[..., 0].shape)
            mask[..., 1][meh] = 0
            mask[..., 2][meh] = 0

        # maybe force edges between touching nuclei
        if self.force_nuclear_edges:
            bck = self.nucleus_codes['BACKGROUND']
            mask[..., 1][mask[..., 2] > 0] = bck

        mask = Image.fromarray(mask)

        # maybe resize the input image to the hpf mpp
        if self.orig2hres_roirgb is not None:
            rgb = self.orig2hres_roirgb(rgb)
            mask = self.orig2hres_roimsk(mask)

        return rgb, mask

    # noinspection DuplicatedCode
    def roi_transforms(self, rgb, mask):
        """Apply various augmentations to & resize ROI."""
        target = {'dense_mask': mask}

        # random crop, possibly with scale augmentation
        rgb, target = self.roi_cropper(rgb=rgb, targets=target)

        # after randomly jittered crop, size is not uniform, so resize
        # to desired roi side .. this is how the crop size jitter gets
        # translated into scale augmentation
        rgb = self.cropped2hres_rgb(rgb)
        target['dense_mask'] = self.cropped2hres_msk(target['dense_mask'])

        # apply transforms (eg random flip, stain augmentation, etc)
        # noinspection PyCallingNonCallable
        rgb, target = self.transforms(rgb, target)
        mask = target['dense_mask']

        # All of the above was at the magnification of the HPF, now we get
        # a small version that is at the smaller ROI magnification
        just_regions = Image.fromarray(np.uint8(mask)[..., 0].copy())
        lowres_rgb = self.hres2lres_roirgb(rgb)
        lowres_mask = self.hres2lres_roimsk(just_regions)

        return {
            'highres_rgb': rgb,
            'highres_mask': mask,
            'lowres_rgb': lowres_rgb,
            'lowres_mask': lowres_mask,
        }

    def __getitem__(self, idx):

        # load roi at HPF resolution
        roiname = self.roinames[idx]
        rgb, mask = self.load_roi_to_hpf_mpp(roiname)

        # apply transforms (crop, color augmentation, resized copy, etc)
        roi_dict = self.roi_transforms(rgb, mask)

        # now convert all to tensors
        if self._tensorize:
            for k, pilim in roi_dict.items():
                # tensorize ...
                tens, _ = self.pil2tensor(pilim, {}, isuint8=k.endswith('rgb'))
                # *** IMPORTANT!!!! ***
                # Also make first dim as image index in batch. This is
                # because Multi-GPU training slices along the first dimension!
                roi_dict[k] = tens[None, ...]
                # masks are long tensors (torch equivalent of int)
                if k.endswith('mask'):
                    roi_dict[k] = roi_dict[k].type(torch.LongTensor)

        # regions to be ignored when picking HPFs
        if self._tensorize:
            ignore = roi_dict['lowres_mask'] == 0
            ignore = ignore.type(
                torch.float16 if self.float16 else torch.float32)
        else:
            ignore = np.uint8(roi_dict['lowres_mask']) == 0
        roi_dict['lowres_ignore'] = ignore

        # Separate images (forward) from ground truth. This is an important
        # step since the DataLoader module expects TWO outputs. It's also
        # nice to be modular this way to separate inputs to the model in
        # forward mode (which includes inference on new slides) and the input
        # to the loss function class
        datks = ['highres_rgb', 'lowres_rgb', 'lowres_ignore']
        data = {k: v for k, v in roi_dict.items() if k in datks}
        truth = {'idx': idx, 'roiname': self.roinames[idx]}
        truth.update({k: v for k, v in roi_dict.items() if k not in datks})

        return data, truth

    def __len__(self):
        return len(self.roinames)


# noinspection PyShadowingNames,PyAttributeOutsideInit
class SimpleDataset(object):
    def __init__(
            self, root: str,
            # formats, sizing, magnification
            float16=False,
            original_mpp=0.25,  # 20x = 0.46 MPP
            hpf_mpp=0.5,  # 20x = 0.46 MPP
            roi_mpp=1.0,  # 10x = 0.92 MPP
            original_side=1024,
            roi_side=256,
            transforms: Union[str, Callable] = 'defaults',
            _shuf=True,  # internal
    ):

        # sanity check
        assert transforms is not None

        self.root = root
        self.original_mpp = original_mpp
        self.roi_mpp = roi_mpp
        self.hpf_mpp = hpf_mpp
        self.original_side = original_side
        self.roi_side = roi_side
        self.float16 = float16
        # setting _shuf to true ensures vis. has multiple slides per batch
        self._shuf = _shuf

        # augmentation transforms
        if transforms == 'defaults':
            self.transforms = transform_dlinput(tlist=None, make_tensor=False)  # noqa
        else:
            self.transforms = transforms

        # resizes ROIs to high resolution magnif (same as HPF)
        sf = self.original_mpp / self.hpf_mpp
        if np.abs(sf - 1.) > 1e-6:
            self.hres_roiside = int(self.original_side * sf)
            self.orig2hres_roirgb = tvdt.Resize(
                size=self.hres_roiside, interpolation=InterpolationMode.BILINEAR)
            self.orig2hres_roimsk = tvdt.Resize(
                size=self.hres_roiside, interpolation=InterpolationMode.NEAREST)
        else:
            self.hres_roiside = self.original_side
            self.orig2hres_roirgb = None
            self.orig2hres_roimsk = None

        # Crops ROIs at hpf mpp, possibly with some jitter for scale augment.
        roi2hpf_sf = roi_mpp / hpf_mpp
        rcside = int(self.roi_side * roi2hpf_sf)
        self.roi_cropper = tvdt.Cropper(size=rcside, iscentral=True)
        self.cropped2hres_rgb = tvdt.Resize(rcside, interpolation=InterpolationMode.BILINEAR)  # noqa
        self.cropped2hres_msk = tvdt.Resize(rcside, interpolation=InterpolationMode.NEAREST)  # noqa

        # resizes ROIs to low resolution -- this happens AFTER cropping/augm.
        if np.abs(roi2hpf_sf - 1.) > 1e-6:
            side = int(rcside / roi2hpf_sf)
            self.hres2lres_roirgb = tvdt.Resize(side, interpolation=InterpolationMode.BILINEAR)  # noqa
            self.hres2lres_roimsk = tvdt.Resize(side, interpolation=InterpolationMode.NEAREST)  # noqa
        else:
            self.hres2lres_roirgb = None
            self.hres2lres_roimsk = None

        # converts the pillow images to tensors
        self._tensorize = True  # internal
        self.pil2tensor = tvdt.PILToTensor(float16=self.float16)

        # set names of rois, slides, etc
        self.set_roinames_etc()

    # noinspection DuplicatedCode
    def set_roinames_etc(self):
        """Set names of ROIs, slides etc."""
        self.roinames = [j for j in os.listdir(self.root) if j.endswith('.png')]
        self.roinames.sort()
        if self._shuf:
            random.shuffle(self.roinames)

    def load_roi_to_hpf_mpp(self, roiname):
        """Load roi as it is saved on disk & resize to HPF MPP."""
        rgb = Image.fromarray(imread(opj(self.root, roiname))).convert('RGB')
        if self.orig2hres_roirgb is not None:
            rgb = self.orig2hres_roirgb(rgb)
        return rgb

    # noinspection DuplicatedCode
    def roi_transforms(self, rgb):
        """Apply various augmentations to & resize ROI."""
        target = {}
        rgb, _ = self.roi_cropper(rgb=rgb, targets={})
        rgb = self.cropped2hres_rgb(rgb)
        rgb, target = self.transforms(rgb, target)
        lowres_rgb = self.hres2lres_roirgb(rgb)
        return {'highres_rgb': rgb, 'lowres_rgb': lowres_rgb}

    def __getitem__(self, idx):

        # load roi at HPF resolution
        roiname = self.roinames[idx]
        rgb = self.load_roi_to_hpf_mpp(roiname)
        roi_dict = self.roi_transforms(rgb)

        # now convert all to tensors
        if self._tensorize:
            for k, pilim in roi_dict.items():
                # tensorize ...
                tens, _ = self.pil2tensor(pilim, {}, isuint8=k.endswith('rgb'))
                # *** NEW! IMPORTANT!!!! ***
                # Also make first dim as image index in batch. This is
                # because Multi-GPU training slices along the first dimension!
                roi_dict[k] = tens[None, ...]

        return roi_dict, {}

    def __len__(self):
        return len(self.roinames)


# =============================================================================

if __name__ == '__main__':

    from MuTILs_Panoptic.utils.RegionPlottingUtils import \
        get_visualization_ready_combined_mask
    import matplotlib.pylab as plt
    from MuTILs_Panoptic.configs.region_model_configs import VisConfigs

    BASEPATH = opj(os.path.expanduser('~'), 'Desktop', 'cTME')
    root = opj(BASEPATH, 'data', 'BootstrapNucleiManualRegions_05132021')

    # we'll work with one fold for this
    fold = 2
    train_slides, test_slides = get_cv_fold_slides(
        train_test_splits_path=opj(root, 'train_test_splits'), fold=fold)

    # init dataset
    dataset = MuTILsDataset(
        root=root, slides=train_slides,
        transforms='defaults',
        roi_side=256,
        hpf_mpp=0.5,
        roi_mpp=1.0,
        strong_class_balance=True,
        # strong_slide_balance=True,
        strong_slide_balance=False,
        training=True,
        # training=False,
    )

    # SPOT CHECK -- visualize ROIs in order of weight
    dataset._tensorize = False
    rws = dataset.roi_weights.copy()
    sampled = np.random.choice(
        np.arange(len(rws)), size=len(rws), p=rws.values,
        replace=False,
        # replace=True,
    )
    for idx in sampled:
        data, truth = dataset.__getitem__(idx)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(data['highres_rgb'])
        vis_mask = get_visualization_ready_combined_mask(
            np.uint8(truth['highres_mask']).copy())
        ax[1].imshow(vis_mask, cmap=VisConfigs.COMBINED_CMAP)
        plt.suptitle(dataset._r2s(truth['roiname']), fontsize=16)
        plt.tight_layout()
        plt.show()
        input()

    # # Let's visualize multiple augmentations of the same image
    # dataset._tensorize = False
    # idx = 300
    # for _ in range(5):
    #     data, truth = dataset.__getitem__(idx)
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #     ax[0].imshow(data['highres_rgb'])
    #     vis_mask = get_visualization_ready_combined_mask(
    #         np.uint8(truth['highres_mask']).copy())
    #     ax[1].imshow(vis_mask, cmap=VisConfigs.COMBINED_CMAP)
    #     plt.show()

    # # SPOT CHECK -- visualize ALL ROIs
    # dataset._tensorize = False
    # for idx in range(len(dataset)):
    #     data, truth = dataset.__getitem__(idx)
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #     ax[0].imshow(data['highres_rgb'])
    #     vis_mask = get_visualization_ready_combined_mask(
    #         np.uint8(truth['highres_mask']).copy())
    #     ax[1].imshow(vis_mask, cmap=VisConfigs.COMBINED_CMAP)
    #     plt.suptitle(dataset._r2s(truth['roiname']))
    #     plt.tight_layout()
    #     # plt.savefig(opj(root, 'vis', f"{idx}_{truth['roiname']}"))
    #     # plt.close()
    #     plt.show()
    #     input()