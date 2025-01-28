import warnings
import numpy as np
import torch
from PIL import Image
from pandas import DataFrame, concat, Series
from skimage.morphology import binary_dilation
from histomicstk.preprocessing.color_normalization import (
    deconvolution_based_normalization)
from histomicstk.preprocessing.color_deconvolution import (
    color_deconvolution_routine)
from histomicstk.features.compute_nuclei_features import (
    compute_nuclei_features)

from MuTILs_Panoptic.utils.TorchUtils import transform_dlinput
from MuTILs_Panoptic.utils.MiscRegionUtils import (
    load_trained_mutils_model, pil2tensor, logits2preds,
    get_objects_from_binmask, get_region_within_x_pixels,
)
from MuTILs_Panoptic.utils.GeneralUtils import unique_nonzero
from MuTILs_Panoptic.utils.CythonUtils import cy_argwhere
from histolab.util import np_to_pil


class MutilsInferenceRunner(object):
    """
    This gets and refactors MuTILs predictions at inference time.
    """
    def __init__(
            self,
            model_configs,
            model_path=None,
            *,
            # roi size and accounting for tile overlap
            roi_side_hres=1024,
            discard_edge_hres=0,  # 0.5 * tile overlap
            # color normalization & augmentation
            cnorm=True,
            cnorm_kwargs=None,
            ntta=0, dltransforms=None,
            maskout_regions_for_cnorm=None,
            # intra-tumoral stroma (saliency)
            filter_stromal_whitespace=False,
            min_tumor_for_saliency=4,
            max_salient_stroma_distance=64,
            # parsing nuclei from inference
            no_watershed_nucleus_classes=None,
            min_nucl_size=5,
            max_nucl_size=90,
            nprops_kwargs=None,
            # internal
            _debug=False,
    ):
        if _debug:
            warnings.warn("Running in DEBUG mode!!!")

        self._debug = _debug
        self.roi_side_hres = roi_side_hres
        self.discard_edge_hres = discard_edge_hres
        self.cfg = model_configs

        # color normalization
        self.cnorm = cnorm
        self.cnorm_kwargs = cnorm_kwargs or {
            'W_target': np.array([
                [0.5807549, 0.08314027, 0.08213795],
                [0.71681094, 0.90081588, 0.41999816],
                [0.38588316, 0.42616716, -0.90380025]
            ]),
            'stain_unmixing_routine_params': {
                'stains': ['hematoxylin', 'eosin'],
                'stain_unmixing_method': 'macenko_pca',
            },
        }

        # test-time color augmentation (0 = no augmentation)
        self.ntta = ntta
        self.dltransforms = dltransforms or transform_dlinput(
            tlist=['augment_stain'], make_tensor=False,
            augment_stain_sigma1=0.75, augment_stain_sigma2=0.75,
        )

        # nuclear prop extraction etc
        self.no_watershed_nucleus_classes = no_watershed_nucleus_classes or [
            'StromalCellNOS', 'ActiveStromalCellNOS']
        self.maskout_regions_for_cnorm = maskout_regions_for_cnorm or [
            'BLOOD', 'WHITE', 'EXCLUDE']
        self.filter_stromal_whitespace = filter_stromal_whitespace
        self.min_tumor_for_saliency = min_tumor_for_saliency
        self.max_salient_stroma_distance = max_salient_stroma_distance
        self.min_nucl_size = min_nucl_size
        self.max_nucl_size = max_nucl_size
        self.nprops_kwargs = nprops_kwargs or dict(
            fsd_bnd_pts=128, fsd_freq_bins=6, cyto_width=8,
            num_glcm_levels=32,
            morphometry_features_flag=True,
            fsd_features_flag=True,
            intensity_features_flag=True,
            gradient_features_flag=True,
            haralick_features_flag=True
        )

        # config shorthands
        self._set_convenience_cfg_shorthand()
        self._fix_mutils_params()

        # we may or may not use this class to do the actual inference
        # otherwise, we just use if to refactor inference we already have
        self.model_path = model_path
        self.device = None
        self.model = None
        self.load_model()

        # internal properties
        self._sldname = None
        self._fold = None
        self._roiname = None

    def load_model(self):
        if self.model_path is None:
            return
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.model = load_trained_mutils_model(self.model_path, mtp=self.mtp)
        self.model.eval()

    def maybe_color_normalize(self, rgb: Image, mask_out=None):
        """"""
        if self.cnorm:
            rgb = deconvolution_based_normalization(
                np.array(rgb), mask_out=mask_out, **self.cnorm_kwargs)
            rgb = np_to_pil(rgb)

        return rgb

    @torch.no_grad()
    def do_inference(self, rgb: Image, lres_ignore=None):
        """Do MuTILs model inference."""
        batchdata = self._prep_batchdata(rgb=rgb, lres_ignore=lres_ignore)
        inference = self.model(batchdata)

        return inference

    def refactor_inference(self, inference, hres_ignore=None):
        """
        Refactor predictions from tensors to actual nuclear masks & locations.
        """
        hres_ignore = self._get_ignore_mask(hres_ignore)

        # region predictions
        rpreds = logits2preds(inference['hpf_region_logits'])[0]
        rpreds[hres_ignore] = 0
        rpreds = self._maybe_filter_stromal_whitespace(rpreds)

        # nuclei raw semantic predictions and object mask
        npred, npred_probab = logits2preds(
            inference['hpf_nuclei'], return_probabs=True)
        objmask, objcodes = self._get_nuclei_objects_mask(npred)
        objmask[hres_ignore] = 0
        objmask, objcodes = self._remove_small_objects(objmask, objcodes)

        if len(objcodes) < 2:
            return

        # get nucleus-wise probabs and semantic-object mask
        classif_df, semantic_mask = self._refactor_nuclear_hpf_mask(
            objmask=objmask, objcodes=objcodes,
            semantic_probabs=npred_probab)
        semantic_mask[hres_ignore] = 0
        combined_mask = self._combine_mask(rpreds, semantic_mask)

        # nuclei prediction without region constraint for comparison
        _, prenpred_probab = logits2preds(
            inference['hpf_nuclei_pre'], return_probabs=True)
        pre_classif_df, presemantic_mask = self._refactor_nuclear_hpf_mask(
            objmask=objmask, objcodes=objcodes,
            semantic_probabs=prenpred_probab)
        presemantic_mask[hres_ignore] = 0
        precombined_mask = self._combine_mask(rpreds, presemantic_mask)

        # concat the classification dataframes
        pre_classif_df = pre_classif_df.drop('Identifier.ObjectCode', axis=1)
        pre_classif_df.columns = [
            'Unconstrained.' + j for j in pre_classif_df.columns]
        classif_df = concat([classif_df, pre_classif_df], axis=1)

        return {
            'nobjects_mask': objmask,  # pixel is object code
            'combined_mask': combined_mask,  # region + semantic nucl. + edges
            'precombined_mask': precombined_mask,  # same but unconstrained
            'classif_df': classif_df,
        }

    def get_salient_stroma_mask(self, mask):
        """
        Salient stroma is within x um from tumor (i.e. intra-tumoral).
        """
        stroma_mask = mask == self.rcd['STROMA']
        stroma_mask[mask == self.rcd['TILS']] = True
        params = {
            'surround_mask': stroma_mask,
            'max_dist': self.max_salient_stroma_distance,
            'min_ref_pixels': self.min_tumor_for_saliency,
        }
        peri_tumoral = get_region_within_x_pixels(
            center_mask=mask == self.rcd['TUMOR'], **params)

        # # THE FOLLOWING BADLY SKEWS RESULTS!!
        # # Normal acini are commonly only predicted as "normal" at the
        # # tumor-stroma edge, so we exclude stroma that is near it even
        # # if that stroma is also near "tumor"
        # peri_normal = get_region_within_x_pixels(
        #     center_mask=mask == self.rcd['NORMAL'], **params)
        # peri_tumoral[peri_normal] = False

        return peri_tumoral

    def get_nuclei_props_df(self, rgb, preds: dict):
        """
        Save morphologic features for nuclei in roi.
        """
        # get hematoxylin and eosin channels
        mask_out = np.in1d(
            preds['combined_mask'][..., 0],
            self.maskout_region_codes
        ).reshape(preds['combined_mask'].shape[:2])
        stains, _, _ = color_deconvolution_routine(
            im_rgb=rgb, mask_out=mask_out,
            stain_unmixing_method='macenko_pca')
        stains = 255 - stains
        # Get nuclei and cytoplasmic features
        # Identifier.ObjectCode ensures correspondence to classif. metadata
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nprops = compute_nuclei_features(
                im_label=preds['nobjects_mask'],
                im_nuclei=stains[..., 0],
                im_cytoplasm=stains[..., 1],
                **self.nprops_kwargs
            )
        nprops.rename(columns={'Label': 'Identifier.ObjectCode'}, inplace=True)
        for attrstr, attr in [
            ('roiname', self._roiname),
            ('slide', self._sldname),
            ('fold', self._fold)
        ]:
            if attr is not None:
                nprops.insert(0, attrstr, attr)

        return nprops

    # HELPERS -----------------------------------------------------------------

    def _maybe_filter_stromal_whitespace(self, mask):
        """
        Whitespace within x um from stroma is just hypodense stroma.
        """
        if not self.filter_stromal_whitespace:
            return mask

        stroma_mask = mask == self.rcd['STROMA']
        stroma_mask[mask == self.rcd['TILS']] = True
        hypodense_stroma = get_region_within_x_pixels(
            center_mask=stroma_mask,
            surround_mask=mask == self.rcd['WHITE'],
            max_dist=32,  # todo: make as param?
            min_ref_pixels=self.min_tumor_for_saliency,
        )
        mask[hypodense_stroma] = self.rcd['STROMA']

        return mask

    @staticmethod
    def _bounds2str(left, top, right, bottom):
        return f"_left-{left}_top-{top}_right-{right}_bottom-{bottom}"

    @staticmethod
    def _str2bounds(bstr):
        locs = ['left', 'top', 'right', 'bottom']
        return [int(bstr.split(f"_{loc}-")[1].split('_')[0]) for loc in locs]

    @staticmethod
    def _calibrate(mdl, x):
        """Calibrate computational to visual TILs assessment scores."""
        return mdl['slope'] * x + mdl['intercept']

    @staticmethod
    def _ksums(records, ks: list, k0=None):
        """Get sum of some value (eg pixels) from multiple rois."""
        whats = records if k0 is None else [j[k0] for j in records]
        whats = DataFrame.from_records(
            whats,
            exclude=[k for k, v in whats[0].items() if isinstance(v, dict)]
        ).loc[:, ks]

        return whats.sum(0).to_dict()

    def _combine_mask(self, regions, semantic):
        """
        Get a mask where first channel is region semantic segmentation mask,
        second is nuclei semantic segmentation mask, and third is nucl. edges.
        """
        tmp_msk = semantic.copy()
        tmp_msk[tmp_msk == self.ncd['BACKGROUND']] = 0
        tmp_msk = 0 + (tmp_msk > 0)
        dilated = binary_dilation(tmp_msk)
        edges = dilated - tmp_msk
        combined = np.uint8(np.concatenate(
            [regions[..., None], semantic[..., None], edges[..., None]], -1))

        return combined

    def _remove_small_objects(self, objmask, objcodes):
        """Remove objects with a side less then min. Important!"""
        new_objcodes = objcodes.copy()
        for obc in objcodes:
            yx = cy_argwhere.cy_argwhere2d(objmask == obc)
            if yx.shape[0] < self.min_nucl_size ** 2:
                objmask[yx[:, 0], yx[:, 1]] = 0
                new_objcodes.remove(obc)
                continue
            ymin, xmin = yx.min(0)
            ymax, xmax = yx.max(0)
            width = xmax - xmin
            height = ymax - ymin
            if (width < self.min_nucl_size) or (height < self.min_nucl_size):
                objmask[yx[:, 0], yx[:, 1]] = 0
                new_objcodes.remove(obc)

        return objmask, new_objcodes

    def _refactor_nuclear_hpf_mask(
            self, semantic_probabs, objmask, objcodes=None):
        """
        Take semantic segmentation logits and an object mask. Then, use the
        object mask to aggregate pixels to get per-nucleus classif. probab.
        and classification. Use this to also improve semantic segm. mask by
        quantizing things to that all pixels belonging to the same nucleus
        have the same label.

        Parameters:
            semantic_probabs (nd array): (nclasses, m, n) sem. segm. probab.
            objmask (nd array): (m, n) where pixel value is object membership
            objcodes (list): which objects to get labels for etc

        Returns:
            DataFrame: indexed by nucleus code in the mask containing classif.
            probability vectors and argmaxed classification

            nd array: (m, n) improved and argmaxed semantic segmentation mask.
        """
        # improve semantic segmentation by doing majority voting so that
        # each object's class is determined by the majority of pixels.
        classif_df, objmask2 = self._aggregate_pixels_for_nucleus(
            semprobs=semantic_probabs, objmask=objmask.copy(),
            objcodes=objcodes)
        # now parse improved semantic mask
        semantic_mask = np.zeros(semantic_probabs.shape[1:])
        coln = 'Classif.StandardClass'
        for clname in set(classif_df.loc[:, coln].values):
            obcs = classif_df.loc[
                classif_df.loc[:, coln] == clname,
                'Identifier.ObjectCode'].tolist()
            relevant_objs = np.in1d(objmask2, obcs).reshape(objmask2.shape)
            semantic_mask[relevant_objs] = self.rcc.NUCLEUS_CODES[clname]
        semantic_mask[semantic_mask == 0] = self.ncd['BACKGROUND']

        return classif_df, semantic_mask

    def _aggregate_pixels_for_nucleus(
            self, semprobs, objmask, objcodes=None, aggsuper=True):
        """
        Aggregates probabilities for all pixels belonging to the same
        nucleus then maybe aggregates to get superclass probab.Then argmax.

        Returns:
            DataFrame indexed by nucleus code in the mask containing classif.
            probability vectors and argmaxed classification
        """
        class2supercode = self.rcc.NClass2Superclass_codes
        code2name = self.rcc.RNUCLEUS_CODES
        supercode2name = self.rcc.RSUPERNUCLEUS_CODES

        if objcodes is None:
            objcodes = np.array(unique_nonzero(objmask))

        # we'll assign everything to this dataframe
        outdf = DataFrame(index=objcodes)
        outdf.loc[:, 'Identifier.ObjectCode'] = objcodes

        # This array holds the aggregated probability from pixels belonging to
        # each nucleus (rows) for each class (columns)
        lbls = np.arange(1, semprobs.shape[0] + 1)
        probvectors = DataFrame(0., index=objcodes, columns=lbls)
        for obc in objcodes:

            yx = cy_argwhere.cy_argwhere2d(objmask == obc)

            if yx.shape[0] < 1:
                continue

            probvectors.loc[obc, :] = semprobs[:, yx[:, 0], yx[:, 1]].mean(1)

            # While we're at it, let's assign the nucleus coordinates
            ymin, xmin = yx.min(0)
            ymax, xmax = yx.max(0)
            outdf.loc[obc, 'Identifier.Xmin'] = xmin
            outdf.loc[obc, 'Identifier.Ymin'] = ymin
            outdf.loc[obc, 'Identifier.Xmax'] = xmax
            outdf.loc[obc, 'Identifier.Ymax'] = ymax

            # IMPORTANT NOTE:
            #  The centroid here is critical, because we use it
            #  to reconstruct the object mask later, if need be, from
            #  the combined region-cell mask where second channel is semantic
            #  and third channel is edges. The idea is to make sure there are
            #  no holes at the pixel at the very center of the nucleus so that
            #  when we later read the semantic mask, do connected components,
            #  and would like to map the rows in this DataFrame with the codes
            #  in the labeled object mask, we can easily and efficiently do so
            #  since we KNOW that the indices from CentroidX and CentroidY
            #  will contain the code for the nucleus of interest.
            #
            cy, cx = np.int32(yx.mean(0))
            outdf.loc[obc, 'Identifier.CentroidX'] = cx
            outdf.loc[obc, 'Identifier.CentroidY'] = cy
            objmask[cy, cx] = obc

        # maybe aggregate superclass probabilities
        super_probvectors = None
        if aggsuper:
            slbls = np.arange(1, max(class2supercode.values()) + 1)
            super_probvectors = DataFrame(0., index=objcodes, columns=slbls)
            for lbl, slbl in class2supercode.items():
                super_probvectors.loc[:, slbl] += probvectors.loc[:, lbl]

        # more informative column names
        probvectors.columns = [
            f"ClassifProbab.{code2name[lbl]}" for lbl in probvectors.columns]
        if aggsuper:
            super_probvectors.columns = [
                f"SuperClassifProbab.{supercode2name[slbl]}"
                for slbl in super_probvectors.columns]

        # argmaxed final classification
        argmaxed_class = Series(
            np.argmax(probvectors.values, 1) + 1,
            index=probvectors.index).map(code2name)
        argmaxed_superclass = None
        if aggsuper:
            argmaxed_superclass = Series(
                np.argmax(super_probvectors.values, 1) + 1,
                index=super_probvectors.index).map(supercode2name)

        # assign to final dataframe
        outdf.loc[:, 'Classif.StandardClass'] = argmaxed_class
        outdf.loc[:, 'Classif.SuperClass'] = argmaxed_superclass
        outdf = concat([outdf, probvectors, super_probvectors], axis=1)

        return outdf, objmask

    def _get_nuclei_objects_mask(self, npred):
        """
        Get nuclear object mask and codes.

        This uses watershed for most nuclear classes. If we don't like to use
        watershed for some classes, this also accomodates this by getting
        PRELIMINARY object label using majority voting of ARGMAXED probabs.
        This is used to find fibroblasts and avoid using watershed for them
        since they are commonly elongated and watershed cuts them up!
        """
        ssegmask = npred.copy()
        ssegmask[ssegmask == self.ncd['BACKGROUND']] = 0
        lbls = unique_nonzero(ssegmask)
        # get PRELIMINARY object mask and codes -- uses watershed
        npred_binary = ssegmask.copy()
        npred_binary = 0 + (npred_binary > 0)
        objmask, objcodes = get_objects_from_binmask(
            binmask=npred_binary, open_first=True,
            minpixels=self.min_nucl_size ** 2,
            maxpixels=self.max_nucl_size ** 2,
            mindist=5, use_watershed=True,
        )

        # maybe this ROI only has watershed classes
        no_watershed = set(lbls).intersection(self.no_watershed_lbls)
        if len(no_watershed) < 1:
            return objmask, objcodes.tolist()

        # This array holds the PRELIMINARY pixel count for each nucleus (rows)
        # for each class (columns) for efficiency majority voting
        obj_labcounts = DataFrame(0., index=objcodes, columns=lbls)
        for lbl in lbls:
            lblmask = 0 + (ssegmask == lbl)
            lblobjs = objmask.copy()
            lblobjs[lblmask == 0] = 0
            unique, counts = np.unique(lblobjs, return_counts=True)
            unique, counts = unique[1:], counts[1:]
            obj_labcounts.loc[unique, lbl] = counts
        # nucleus label is obtained by majority voting
        lbl_idxs = np.argmax(obj_labcounts.values, axis=1)
        obj_lbls = [lbls[idx] for idx in lbl_idxs]
        # Isolate binary mask for nuclei subset where we don't want watershed.
        watershed_objs = [
            obc for lab, obc in zip(obj_lbls, objcodes)
            if lab not in no_watershed]
        binmask_subset = npred_binary.copy()
        ignore = np.in1d(objmask, watershed_objs).reshape(objmask.shape)
        binmask_subset[ignore] = 0
        objmask[binmask_subset > 0] = 0
        # object mask for classes where we don't like watershed (fibroblasts)
        objmask2, _ = get_objects_from_binmask(
            binmask=binmask_subset, open_first=True,
            minpixels=self.min_nucl_size ** 2,
            maxpixels=self.max_nucl_size ** 2,
            mindist=5, use_watershed=False,
        )
        # merge the two object masks
        bck2 = objmask2 == 0
        objmask2 += np.max(objcodes)
        objmask2[bck2] = 0
        objmask += objmask2
        objcodes = unique_nonzero(objmask)

        return objmask, objcodes

    def _get_ignore_mask(self, ignore):
        """Get hres ignore mask with discard edges."""
        if ignore is None:
            ignore = torch.zeros(
                (self.roi_side_hres, self.roi_side_hres),
                dtype=bool, device=self.device)
        if self.discard_edge_hres > 0:
            de = self.discard_edge_hres
            ignore[:de, :] = True
            ignore[-de:, :] = True
            ignore[:, :de] = True
            ignore[:, -de:] = True

        return ignore

    def _prep_batchdata(self, rgb: Image, lres_ignore=None):
        """Prep tensor batch data for MuTILs model inference."""
        # prep batch tensors
        batchdata = []
        for aug in range(self.ntta + 1):
            # maybe apply test-time augmentation
            rgb1 = rgb.copy()
            if aug > 0:
                rgb1, _ = self.dltransforms(rgb1, {})
            # resize low resolution rgb to desired mpp
            lres_rgb1 = rgb1.resize(
                (self.roi_side_lres, self.roi_side_lres), Image.LANCZOS)
            # tensorize
            rgb1, _ = pil2tensor(rgb1, {})
            lres_rgb1, _ = pil2tensor(lres_rgb1, {})
            bd = {
                'highres_rgb': rgb1[None, ...],
                'lowres_rgb': lres_rgb1[None, ...],
            }
            if lres_ignore is not None:
                ignore = torch.as_tensor(lres_ignore, dtype=torch.float32)
                bd['lowres_ignore'] = ignore[None, None, ...]
            batchdata.append(bd)
        # move all to device
        batchdata = [
            {k: v.to(self.device) for k, v in bd.items()} for bd in batchdata
        ]
        return batchdata

    def _set_convenience_cfg_shorthand(self):
        """Convenience shorthand."""
        self.mtp = self.cfg.MuTILsParams
        self.rcc = self.cfg.RegionCellCombination
        self.rcd = self.cfg.RegionCellCombination.REGION_CODES
        self.ncd = self.cfg.RegionCellCombination.NUCLEUS_CODES
        self.no_watershed_lbls = {
            self.ncd[cls] for cls in self.no_watershed_nucleus_classes}
        self.maskout_region_codes = [
            self.rcd[reg] for reg in self.maskout_regions_for_cnorm]

    def _fix_mutils_params(self):
        """params that must be true for inference"""
        # magnification & size settings
        self.hres_mpp = self.mtp.model_params['hpf_mpp']
        self.lres_mpp = self.mtp.model_params['roi_mpp']
        self.vlres_mpp = 2 * self.lres_mpp
        self.h2l = self.hres_mpp / self.lres_mpp
        self.h2vl = self.hres_mpp / self.vlres_mpp
        self.roi_side_lres = int(self.h2l * self.roi_side_hres)
        self.roi_side_vlres = int(self.h2vl * self.roi_side_hres)
        self.n_edge_pixels_discarded = 4 * self.discard_edge_hres * (
            self.roi_side_hres - self.discard_edge_hres)
        # MuTILs params
        self.mtp.model_params.update({
            'training': False,
            'roi_side': self.roi_side_lres,
            'hpf_side': self.roi_side_hres,  # predict all nuclei in roi
            'topk_hpf': 1,
        })
