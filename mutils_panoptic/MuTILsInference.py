import warnings
import numpy as np
import torch
import matplotlib.pylab as plt
import ast
from os.path import join as opj
from PIL import Image
from imageio import imwrite
from matplotlib.colors import ListedColormap
from pandas import DataFrame, concat
from skimage.morphology import binary_dilation
from dataclasses import dataclass
from skimage.measure import regionprops
import multiprocessing as mp

# histolab
from histolab.util import np_to_pil
from histolab.tile import Tile

# histomicstk
from histomicstk.preprocessing.color_normalization import (
    deconvolution_based_normalization,
)
from histomicstk.preprocessing.color_deconvolution import color_deconvolution_routine
from histomicstk.features.compute_nuclei_features import compute_nuclei_features

# mutils
from MuTILs_Panoptic.utils.RegionPlottingUtils import (
    get_visualization_ready_combined_mask as gvcm,
)
from MuTILs_Panoptic.utils.GeneralUtils import (
    unique_nonzero,
    CollectErrors,
    save_json,
    _divnonan,
    move_to_cpu_recursive,
)
from MuTILs_Panoptic.utils.CythonUtils import cy_argwhere
from MuTILs_Panoptic.utils.MiscRegionUtils import (
    logits2preds,
    summarize_nuclei_mask,
    get_objects_from_binmask,
    get_region_within_x_pixels,
    summarize_region_mask,
    get_configured_logger,
)
import MuTILs_Panoptic.configs.panoptic_model_configs as model_configs

collect_errors = CollectErrors()


@dataclass
class BatchData:
    """Dataclass of a single batch. It is the input of the model. It stores:
    - highres_rgb: np.ndarray - the high resolution RGB image
    - lowres_rgb: np.ndarray - the low resolution RGB image
    - lowres_ignore: np.ndarray - the low resolution ignore mask
    """

    highres_rgb: np.ndarray = np.array([], dtype=np.uint32)
    lowres_rgb: np.ndarray = np.array([], dtype=np.uint32)
    lowres_ignore: np.ndarray = np.array([], dtype=np.uint32)

    def to_cpu(self):
        self.highres_rgb = move_to_cpu_recursive(self.highres_rgb)
        self.lowres_rgb = move_to_cpu_recursive(self.lowres_rgb)
        self.lowres_ignore = move_to_cpu_recursive(self.lowres_ignore)


@dataclass
class ROI:
    """Dataclass of a single ROI. It stores:
    - number: int - the number of the ROI i.e. its index in the slide
    - name: str - the name of the ROI
    - model: str - the model assigned to the ROI to process it
    - batch: BatchData - the batch data of the ROI holding the RGB images
    - hres_ignore: np.ndarray - the ignore mask for the high resolution image
    - img: Image.Image - the RGB image of the ROI
    """

    number: int
    coords: list[int]
    name: str
    model: str
    batch: BatchData
    hres_ignore: np.ndarray
    img: Image.Image

    def to_cpu(self):
        self.batch.to_cpu()
        self.hres_ignore = move_to_cpu_recursive(self.hres_ignore)


@dataclass
class InferenceResult:
    """Dataclass of a single inference result. It stores:
    - number: int - the number of the ROI i.e. its index in the slide
    - coords: list[int] - the coordinates of the ROI
    - name: str - the name of the ROI
    - model: str - the model assigned to the ROI to process it
    - batch: BatchData - the batch data of the ROI holding the RGB images
    - hres_ignore: np.ndarray - the ignore mask for the high resolution image
    - img: Image.Image - the RGB image of the ROI
    - result: np.ndarray - the result of the inference
    """

    number: int
    coords: list[int]
    name: str
    model: str
    batch: BatchData
    hres_ignore: np.ndarray
    img: Image.Image
    result: np.ndarray

    def to_cpu(self):
        self.batch.to_cpu()
        self.hres_ignore = move_to_cpu_recursive(self.hres_ignore)
        self.result = move_to_cpu_recursive(self.result)


class ROIPreProcessor:
    def __init__(self, inference_queue: mp.Queue, roi_chunk: list, configs: dict):
        self.inference_queue = inference_queue
        self.roi_chunk = roi_chunk
        self._sldname = configs.get("_sldname")
        self._top_rois = configs.get("_top_rois")
        self._slide = configs.get("_slide")
        self.hres_mpp = configs.get("hres_mpp")
        self.roi_side_hres = configs.get("roi_side_hres")
        self.roi_side_lres = configs.get("roi_side_lres")
        self.cnorm = configs.get("cnorm")
        self.cnorm_kwargs = configs.get("cnorm_kwargs")

    def run(self):
        """Loop through one chunk."""
        for single_roi in self.roi_chunk:
            self.run_roi(single_roi)
            self.inference_queue.put(self.roi)

    def run_roi(self, single_roi: tuple):
        """Run the preprocessing for one ROI."""
        self.roi = ROI(
            number=single_roi[0],
            coords=self._get_roi_coords(single_roi[0]),
            name="",
            model=single_roi[1],
            batch=BatchData(),
            hres_ignore=np.array([]),
            img=None,
        )
        self.roi.name = (
            f"{self._sldname}_roi-{self.roi.number}"
            f"{self._bounds2str(*self.roi.coords)}"
        )
        # get and predict roi
        tile = self._slide.extract_tile(
            coords=self._top_rois[self.roi.number][1],
            mpp=self.hres_mpp,
            tile_size=(self.roi_side_hres, self.roi_side_hres),
        )
        self._provide_input(tile=tile)

    def _get_roi_coords(self, roi_number: int) -> list:
        """Get the coordinates of the ROI."""
        return [int(j) for j in self._top_rois[roi_number][1]]

    @staticmethod
    def _bounds2str(left, top, right, bottom):
        return f"_left-{left}_top-{top}_right-{right}_bottom-{bottom}"

    def _provide_input(self, tile: Tile):
        """Get MuTILS predictions for one ROI."""
        hres_ignore, lres_ignore = self._get_tile_ignore(tile)
        rgb = self._maybe_color_normalize(
            tile.image.convert("RGB"), mask_out=hres_ignore
        )
        self._prep_batchdata(rgb, lres_ignore)
        self.roi.hres_ignore = hres_ignore
        self.roi.img = rgb

    def _get_tile_ignore(self, tile: Tile, filter_tissue=False, get_lres=True):
        """Get region outside tissue (eg. marker pen & white space)"""
        tile._filter_tissue = filter_tissue
        hres_ignore = ~tile._tissue_mask
        hres_ignore, _ = get_objects_from_binmask(
            hres_ignore, minpixels=128, use_watershed=False
        )
        hres_ignore = Image.fromarray(hres_ignore)
        if get_lres:
            lres_ignore = hres_ignore.resize(
                (self.roi_side_lres, self.roi_side_lres), Image.NEAREST
            )
            lres_ignore = np.array(lres_ignore, dtype=bool)
        else:
            lres_ignore = None
        hres_ignore = np.array(hres_ignore, dtype=bool)

        return hres_ignore, lres_ignore

    def _maybe_color_normalize(self, rgb: Image, mask_out=None):
        """"""
        if self.cnorm:
            rgb = deconvolution_based_normalization(
                np.array(rgb), mask_out=mask_out, **self.cnorm_kwargs
            )
            rgb = np_to_pil(rgb)

        return rgb

    def _prep_batchdata(self, rgb: Image, lres_ignore=None):
        """Prep tensor batch data for MuTILs model inference."""

        # Resize low resolution rgb to desired mpp
        lres_rgb = rgb.resize((self.roi_side_lres, self.roi_side_lres), Image.LANCZOS)

        # Get numpy arrays from Image objects
        img = np.array(rgb, dtype=np.float32)
        rgb = np.transpose(img, (2, 0, 1)) / 255.0

        img = np.array(lres_rgb, dtype=np.float32)
        lres_rgb = np.transpose(img, (2, 0, 1)) / 255.0

        if lres_ignore is not None:
            ignore = np.asarray(lres_ignore, dtype=np.float32)
            ignore = ignore[None, None, ...]

        self.roi.batch = BatchData(
            highres_rgb=rgb[None, ...],
            lowres_rgb=lres_rgb[None, ...],
            lowres_ignore=ignore,
        )


# ==================================================================================================
class ROIInferenceProcessor:
    def __init__(
        self,
        gpu_id: int,
        inference_queue: mp.Queue,
        postprocess_queue: mp.Queue,
        model: dict,
    ):
        self.gpu_id = gpu_id
        self.inference_queue = inference_queue
        self.postprocess_queue = postprocess_queue
        self.model = model

    def run(self):
        while True:
            roi: ROI = self.inference_queue.get()
            if roi is None:
                self.put(end_signal=True)
                break
            result = self.inference(roi)
            roi.to_cpu()
            self.inference_result = InferenceResult(
                number=roi.number,
                coords=roi.coords,
                name=roi.name,
                model=roi.model,
                batch=roi.batch,
                hres_ignore=roi.hres_ignore,
                img=roi.img,
                result=result,
            )
            self.inference_result.to_cpu()
            torch.cuda.empty_cache()
            self.put()

    def inference(self, roi: ROI):

        highres_rgb = torch.as_tensor(roi.batch.highres_rgb, dtype=torch.float32)
        lowres_rgb = torch.as_tensor(roi.batch.lowres_rgb, dtype=torch.float32)
        lowres_ignore = torch.as_tensor(roi.batch.lowres_ignore, dtype=torch.float32)

        batchdata = [
            {
                "highres_rgb": highres_rgb.to(f"cuda:{self.gpu_id}"),
                "lowres_rgb": lowres_rgb.to(f"cuda:{self.gpu_id}"),
                "lowres_ignore": lowres_ignore.to(f"cuda:{self.gpu_id}"),
            }
        ]

        _model = self.model["model"]

        with torch.no_grad():
            inference = _model(batchdata)

        del batchdata

        return inference

    def put(self, end_signal=False):
        """Put inference result into the postprocess queue."""
        if end_signal:
            return

        self.postprocess_queue.put(self.inference_result)

        del self.inference_result
        torch.cuda.empty_cache()


# ==================================================================================================
class ROIPostProcessor:
    def __init__(self, process_number: int, postprocess_queue: mp.Queue, configs: dict):
        self.process_number = process_number
        self.postprocess_queue = postprocess_queue
        self._sldname = configs.get("_sldname")
        self._slide = configs.get("_slide")
        self.filter_stromal_whitespace = configs.get("filter_stromal_whitespace")
        self.min_nucl_size = configs.get("min_nucl_size")
        self.max_nucl_size = configs.get("max_nucl_size")
        self.no_watershed_lbls = configs.get("no_watershed_lbls")
        self.hres_mpp = configs.get("hres_mpp")
        self.roi_side_hres = configs.get("roi_side_hres")
        self.discard_edge_hres = configs.get("discard_edge_hres")
        self.n_edge_pixels_discarded = configs.get("n_edge_pixels_discarded")
        self.max_salient_stroma_distance = configs.get("max_salient_stroma_distance")
        self.min_tumor_for_saliency = configs.get("min_tumor_for_saliency")
        self.maskout_region_codes = configs.get("maskout_region_codes")
        self.nprops_kwargs = configs.get("nprops_kwargs")
        self._savedir = configs.get("_savedir")
        self.save_annotations = configs.get("save_annotations")
        self.save_wsi_mask = configs.get("save_wsi_mask")
        self.save_nuclei_meta = configs.get("save_nuclei_meta")
        self.save_nuclei_props = configs.get("save_nuclei_props")
        self._debug = configs.get("_debug")
        self.rcd = model_configs.RegionCellCombination.REGION_CODES
        self.rcc = model_configs.RegionCellCombination
        self.ncd = model_configs.RegionCellCombination.NUCLEUS_CODES
        self.VisConfigs = model_configs.VisConfigs

    def run(self):
        self.logger = get_configured_logger(
            logdir="/home/output/LOGS",
            prefix=f"MuTILsWSIRunner_chunk{self.process_number}",
            tofile=True,
        )
        collect_errors.logger = self.logger
        while True:
            roi: InferenceResult = self.postprocess_queue.get()
            if roi is None:
                break
            self.run_roi(roi)
            self.logger.info(f"Completed ROI {roi.number} by {roi.model}.")

    def run_roi(self, roi: InferenceResult) -> None:
        """Run postprocessing for the ROI.

        Parameters:
        ----------
        inference: dict
            Dictionary containing the model inference.
        hres_ignore: np.array
            High resolution ignore mask.
        rgb: Image
            RGB image of the ROI.
        """
        self._roicoords = roi.coords
        self._modelname = roi.model
        self._roiname = roi.name
        self._rid = roi.number

        preds = self.refactor_inference(roi.result, hres_ignore=roi.hres_ignore)
        if preds is None:
            self.logger.info(f"ROI {self._rid} has no nuclei!")
            return

        preds["sstroma"] = self.get_salient_stroma_mask(preds["combined_mask"][..., 0])
        self._maybe_save_roi_preds(rgb=roi.img, preds=preds)
        preds = self._simplify_roi_preds(preds)

        self._summarize_roi(preds)

    def refactor_inference(self, inference, hres_ignore=None):
        """
        Refactor predictions from tensors to actual nuclear masks & locations.
        """
        hres_ignore = self._get_ignore_mask(hres_ignore)

        # region predictions
        rpreds = logits2preds(inference["hpf_region_logits"])[0]
        rpreds[hres_ignore] = 0
        rpreds = self._maybe_filter_stromal_whitespace(rpreds)

        # nuclei raw semantic predictions and object mask
        npred, npred_probab = logits2preds(inference["hpf_nuclei"], return_probabs=True)
        objmask, objcodes = self._get_nuclei_objects_mask(npred)

        if len(objcodes) < 1:
            return

        objmask[hres_ignore] = 0
        objmask, objcodes = self._remove_small_objects(objmask)

        if len(objcodes) < 2:
            return

        # get nucleus-wise probabs and semantic-object mask
        classif_df, semantic_mask = self._refactor_nuclear_hpf_mask(
            objmask=objmask, objcodes=objcodes, semantic_probabs=npred_probab
        )
        semantic_mask[hres_ignore] = 0
        combined_mask = self._combine_mask(rpreds, semantic_mask)

        # nuclei prediction without region constraint for comparison
        _, prenpred_probab = logits2preds(
            inference["hpf_nuclei_pre"], return_probabs=True
        )
        pre_classif_df, presemantic_mask = self._refactor_nuclear_hpf_mask(
            objmask=objmask, objcodes=objcodes, semantic_probabs=prenpred_probab
        )
        presemantic_mask[hres_ignore] = 0
        precombined_mask = self._combine_mask(rpreds, presemantic_mask)

        # concat the classification dataframes
        pre_classif_df = pre_classif_df.drop("Identifier.ObjectCode", axis=1)
        pre_classif_df.columns = ["Unconstrained." + j for j in pre_classif_df.columns]
        classif_df = concat([classif_df, pre_classif_df], axis=1)

        return {
            "nobjects_mask": objmask,  # pixel is object code
            "combined_mask": combined_mask,  # region + semantic nucl. + edges
            "precombined_mask": precombined_mask,  # same but unconstrained
            "classif_df": classif_df,
        }

    def get_salient_stroma_mask(self, mask):
        """
        Salient stroma is within x um from tumor (i.e. intra-tumoral).
        """
        stroma_mask = mask == self.rcd["STROMA"]
        stroma_mask[mask == self.rcd["TILS"]] = True
        params = {
            "surround_mask": stroma_mask,
            "max_dist": self.max_salient_stroma_distance,
            "min_ref_pixels": self.min_tumor_for_saliency,
        }
        peri_tumoral = get_region_within_x_pixels(
            center_mask=mask == self.rcd["TUMOR"], **params
        )

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
        mask_out = np.isin(preds["combined_mask"][..., 0], self.maskout_region_codes)
        stains, _, _ = color_deconvolution_routine(
            im_rgb=rgb, mask_out=mask_out, stain_unmixing_method="macenko_pca"
        )
        stains = 255 - stains
        # Get nuclei and cytoplasmic features
        # Identifier.ObjectCode ensures correspondence to classif. metadata
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nprops = compute_nuclei_features(
                im_label=preds["nobjects_mask"],
                im_nuclei=stains[..., 0],
                im_cytoplasm=stains[..., 1],
                **self.nprops_kwargs,
            )
        nprops.rename(columns={"Label": "Identifier.ObjectCode"}, inplace=True)
        for attrstr, attr in [
            ("roiname", self._roiname),
            ("slide", self._sldname),
            ("fold", self._modelname),
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

        stroma_mask = mask == self.rcd["STROMA"]
        stroma_mask[mask == self.rcd["TILS"]] = True
        hypodense_stroma = get_region_within_x_pixels(
            center_mask=stroma_mask,
            surround_mask=mask == self.rcd["WHITE"],
            max_dist=32,  # todo: make as param?
            min_ref_pixels=self.min_tumor_for_saliency,
        )
        mask[hypodense_stroma] = self.rcd["STROMA"]

        return mask

    def _combine_mask(self, regions, semantic):
        """
        Get a mask where first channel is region semantic segmentation mask,
        second is nuclei semantic segmentation mask, and third is nucl. edges.
        """
        tmp_msk = semantic.copy()
        tmp_msk[tmp_msk == self.ncd["BACKGROUND"]] = 0
        tmp_msk = 0 + (tmp_msk > 0)
        dilated = binary_dilation(tmp_msk)
        edges = dilated - tmp_msk
        combined = np.uint8(
            np.concatenate(
                [regions[..., None], semantic[..., None], edges[..., None]], -1
            )
        )

        return combined

    def _remove_small_objects(self, objmask: np.array) -> tuple:
        """Remove objects with a side less then min. Important!

        Parameters:
        ----------
        objmask: np.array
            Object mask with labels.

        Returns:
        -------
        tuple
            Tuple containing the updated object mask and object codes.
        """
        # Compute region properties for each labeled object
        regions = regionprops(objmask)

        for region in regions:

            # Get bounding box and compute width/height
            minr, minc, maxr, maxc = region.bbox
            width = maxc - minc
            height = maxr - minr

            # Check area and size thresholds
            if (
                (region.area < self.min_nucl_size**2)
                or (width < self.min_nucl_size)
                or (height < self.min_nucl_size)
            ):

                coords = region.coords  # shape (N, 2): row (y), col (x)
                objmask[coords[:, 0], coords[:, 1]] = 0  # Zero out the object

        objcodes = unique_nonzero(objmask)

        return objmask, objcodes

    def _refactor_nuclear_hpf_mask(self, semantic_probabs, objmask, objcodes=None):
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
            semprobs=semantic_probabs, objmask=objmask.copy(), objcodes=objcodes
        )
        # now parse improved semantic mask
        semantic_mask = np.zeros(semantic_probabs.shape[1:])
        coln = "Classif.StandardClass"
        for clname in set(classif_df.loc[:, coln].values):
            obcs = classif_df.loc[
                classif_df.loc[:, coln] == clname, "Identifier.ObjectCode"
            ].tolist()
            relevant_objs = np.isin(objmask2, obcs)
            semantic_mask[relevant_objs] = self.rcc.NUCLEUS_CODES[clname]
        semantic_mask[semantic_mask == 0] = self.ncd["BACKGROUND"]

        return classif_df, semantic_mask

    def _aggregate_pixels_for_nucleus(
        self, semprobs, objmask, objcodes=None, aggsuper=True
    ):
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

        num_classes = semprobs.shape[0]
        lbls = np.arange(1, num_classes + 1)

        # Dictionary-based storage instead of DataFrame
        results = {obc: {} for obc in objcodes}
        probvectors = np.zeros((len(objcodes), num_classes))

        for i, obc in enumerate(objcodes):
            yx = cy_argwhere.cy_argwhere2d(objmask == obc)

            if yx.shape[0] < 1:
                continue

            probvectors[i, :] = semprobs[:, yx[:, 0], yx[:, 1]].mean(axis=1)

            # While we're at it, let's assign the nucleus coordinates
            ymin, xmin = yx.min(0)
            ymax, xmax = yx.max(0)

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
            objmask[cy, cx] = obc

            results[obc] = {
                "Identifier.ObjectCode": obc,
                "Identifier.Xmin": xmin,
                "Identifier.Ymin": ymin,
                "Identifier.Xmax": xmax,
                "Identifier.Ymax": ymax,
                "Identifier.CentroidX": cx,
                "Identifier.CentroidY": cy,
            }

        # maybe aggregate superclass probabilities
        super_probvectors = None
        if aggsuper:
            num_super_classes = max(class2supercode.values())
            super_probvectors = np.zeros((len(objcodes), num_super_classes))
            for lbl, slbl in class2supercode.items():
                super_probvectors[:, slbl - 1] += probvectors[:, lbl - 1]

        # Compute argmax classifications
        argmaxed_class = np.argmax(probvectors, axis=1) + 1
        argmaxed_class_names = np.vectorize(code2name.get)(argmaxed_class)

        argmaxed_superclass = None
        if aggsuper:
            argmaxed_superclass = np.argmax(super_probvectors, axis=1) + 1
            argmaxed_superclass = np.vectorize(supercode2name.get)(argmaxed_superclass)

        # Convert dictionary results to a DataFrame
        outdf = DataFrame.from_dict(results, orient="index")

        outdf["Classif.StandardClass"] = argmaxed_class_names
        if aggsuper:
            outdf["Classif.SuperClass"] = argmaxed_superclass

        # Convert probability vectors to DataFrame
        prob_df = DataFrame(
            probvectors,
            index=objcodes,
            columns=[f"ClassifProbab.{code2name[lbl]}" for lbl in lbls],
        )

        if aggsuper:
            super_prob_df = DataFrame(
                super_probvectors,
                index=objcodes,
                columns=[
                    f"SuperClassifProbab.{supercode2name[slbl]}"
                    for slbl in range(1, num_super_classes + 1)
                ],
            )
            outdf = concat([outdf, prob_df, super_prob_df], axis=1)
        else:
            outdf = concat([outdf, prob_df], axis=1)

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
        ssegmask[ssegmask == self.ncd["BACKGROUND"]] = 0
        lbls = unique_nonzero(ssegmask)
        # get PRELIMINARY object mask and codes -- uses watershed
        npred_binary = ssegmask.copy()
        npred_binary = 0 + (npred_binary > 0)
        objmask, objcodes = get_objects_from_binmask(
            binmask=npred_binary,
            open_first=True,
            minpixels=self.min_nucl_size**2,
            maxpixels=self.max_nucl_size**2,
            mindist=5,
            use_watershed=True,
        )

        # maybe this ROI only has watershed classes
        no_watershed = set(lbls).intersection(self.no_watershed_lbls)
        if len(no_watershed) < 1:
            return objmask, objcodes.tolist()

        if len(objcodes) < 1:
            return objmask, objcodes.tolist()

        # This array holds the PRELIMINARY pixel count for each nucleus (rows)
        # for each class (columns) for efficiency majority voting
        obj_labcounts = DataFrame(0.0, index=objcodes, columns=lbls)
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
            obc for lab, obc in zip(obj_lbls, objcodes) if lab not in no_watershed
        ]
        binmask_subset = npred_binary.copy()
        ignore = np.isin(objmask, watershed_objs)
        binmask_subset[ignore] = 0
        objmask[binmask_subset > 0] = 0
        # object mask for classes where we don't like watershed (fibroblasts)
        objmask2, _ = get_objects_from_binmask(
            binmask=binmask_subset,
            open_first=True,
            minpixels=self.min_nucl_size**2,
            maxpixels=self.max_nucl_size**2,
            mindist=5,
            use_watershed=False,
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
            # NOTE: We are off from the GPU
            # ignore = torch.zeros(
            #     (self.roi_side_hres, self.roi_side_hres),
            #     dtype=bool, device=self.device)
            ignore = np.zeros((self.roi_side_hres, self.roi_side_hres), dtype=bool)
        if self.discard_edge_hres > 0:
            de = self.discard_edge_hres
            ignore[:de, :] = True
            ignore[-de:, :] = True
            ignore[:, :de] = True
            ignore[:, -de:] = True

        return ignore

    @collect_errors()
    def _maybe_save_nuclei_annotation(self, classif_df):
        """
        Save nuclei locations annotation.
        """
        if not self.save_annotations:
            return
        # TODO: consider removing pandas dataframes from this function
        # fix coords to wsi base magnification
        left, top, _, _ = self._roicoords
        colmap = {
            "Identifier.CentroidX": "x",
            "Identifier.CentroidY": "y",
            "Classif.StandardClass": "StandardClass",
            "Classif.SuperClass": "SuperClass",
        }
        df = classif_df.loc[:, list(colmap.keys())].copy()
        df.rename(columns=colmap, inplace=True)
        # df.loc[:, ['x', 'y']] *= self.hres_mpp / self._slide.base_mpp
        df[["x", "y"]] = df[["x", "y"]].astype(float) * (
            self.hres_mpp / self._slide.base_mpp
        )
        df.loc[:, "x"] += left
        df.loc[:, "y"] += top

        self._save_nuclei_locs_hui_style(df)

    @collect_errors()
    def _maybe_save_roi_preds(self, rgb, preds):
        """Save masks, metadata, etc."""
        if self.save_wsi_mask:
            mask = preds["combined_mask"].copy()
            imwrite(opj(self._savedir, "roiMasks", self._roiname + ".png"), mask)
        self._maybe_visualize_roi(
            rgb, mask=preds["combined_mask"], sstroma=preds["sstroma"]
        )
        if self.save_nuclei_meta:
            preds["classif_df"].to_csv(
                opj(self._savedir, "nucleiMeta", self._roiname + ".csv"),
            )
        self._maybe_save_nuclei_props(rgb=rgb, preds=preds)
        self._maybe_save_nuclei_annotation(preds["classif_df"])

    @collect_errors()
    def _maybe_visualize_roi(self, rgb, mask, sstroma):
        """
        Plot and save roi visualization.
        """
        if not self._debug:
            return

        _, ax = plt.subplots(1, 2, figsize=(7.0 * 2, 7.0))
        ax[0].imshow(rgb)
        ax[0].imshow(
            np.ma.masked_array(sstroma, mask=~sstroma),
            alpha=0.3,
            cmap=ListedColormap([[0.01, 0.74, 0.25]]),
        )
        ax[1].imshow(
            gvcm(mask), cmap=self.VisConfigs.COMBINED_CMAP, interpolation="nearest"
        )
        plt.tight_layout(pad=0.4)
        plt.savefig(opj(self._savedir, "roiVis", self._roiname + ".png"))
        plt.close()

    @collect_errors()
    def _maybe_save_nuclei_props(self, rgb, preds):
        """Get and save nuclear morphometric features."""
        if not self.save_nuclei_props:
            return
        # self.logger.info(self._rmonitor + ": getting nuclei props ..")
        props = self.get_nuclei_props_df(rgb=np.uint8(rgb), preds=preds)
        props.to_csv(
            opj(self._savedir, "nucleiProps", self._roiname + ".csv"),
        )

    @staticmethod
    def _simplify_roi_preds(preds):
        """Keep only what's needed and simplify terminology."""
        keep_cols = [
            c for c in preds["classif_df"].columns if not c.startswith("Unconstrained.")
        ]
        return {
            "mask": preds["combined_mask"],
            "sstroma": preds["sstroma"],
            "cldf": preds["classif_df"].loc[:, keep_cols],
        }

    def _summarize_roi(self, preds):
        """
        Get metrics from roi.
        """
        rgsumm, metrics = self._get_region_metrics(
            preds["mask"][..., 0], sstroma=preds["sstroma"]
        )
        nusumm, nmetrics = self._get_nuclei_metrics(
            preds["cldf"], rstroma_count=rgsumm["pixelCount_AnyStroma"]
        )
        metrics.update(nmetrics)
        left, top, right, bottom = self._roicoords
        meta = {
            "slide_name": self._sldname,
            "roi_id": self._rid,
            "roi_name": self._roiname,
            "wsi_left": left,
            "wsi_top": top,
            "wsi_right": right,
            "wsi_bottom": bottom,
            "mpp": self.hres_mpp,
            "model_name": self._modelname,
            "metrics": metrics,
            "region_summary": rgsumm,
            "nuclei_summary": nusumm,
        }
        save_json(meta, path=opj(self._savedir, "roiMeta", self._roiname + ".json"))

    def _get_region_metrics(self, mask, sstroma):
        """
        Summarize region mask and some metrics.
        """
        # pixel summaries
        out = summarize_region_mask(mask, rcd=self.rcd)
        out["pixelCount_EXCLUDE"] -= self.n_edge_pixels_discarded
        # isolate salient tils
        salient_tils = mask == self.rcd["TILS"]
        salient_tils[~sstroma] = False
        p = "pixelCount"
        out.update(
            {
                f"{p}_SalientStroma": int(sstroma.sum()),
                f"{p}_SalientTILs": int(salient_tils.sum()),
            }
        )

        return self._region_metrics(out)

    def _region_metrics(self, out: dict, nrois=1):
        """
        Given pixel count summary, summarize region metrics
        """
        # summarize metrics
        p = "pixelCount"
        everything = nrois * ((self.roi_side_hres**2) - self.n_edge_pixels_discarded)
        junk = out[f"{p}_JUNK"] + out[f"{p}_WHITE"] + out[f"{p}_EXCLUDE"]
        nonjunk = everything - junk
        anystroma = out[f"{p}_STROMA"] + out[f"{p}_TILS"]
        out[f"{p}_AnyStroma"] = anystroma
        # Saliency score is higher when there's more tumor & salient stroma
        # Note that relying on stroma within x microns from tumor edge is not
        # enough as ROIs with scattered tumor nests that are spaced out would
        # get a high score even though there's little tumor or maybe even
        # a few misclassified pixels here and there. So we use both.
        sscore = out[f"{p}_SalientStroma"] / everything
        sscore *= out[f"{p}_TUMOR"] / everything
        metrics = {
            "SaliencyScore": sscore,
            "TissueRatio": nonjunk / everything,
            "TILs2AnyStromaRatio": _divnonan(out[f"{p}_TILS"], anystroma),
            "TILs2AllRatio": _divnonan(out[f"{p}_TILS"], nonjunk),
            "TILs2TumorRatio": _divnonan(out[f"{p}_TILS"], out[f"{p}_TUMOR"]),
            "SalientTILs2StromaRatio": _divnonan(
                out[f"{p}_SalientTILs"], out[f"{p}_SalientStroma"]
            ),
        }

        return out, metrics

    def _get_nuclei_metrics(self, cldf, rstroma_count):
        """
        Summarize nuclei mask and some metrics.
        """
        out = summarize_nuclei_mask(
            cldf.loc[:, "Classif.StandardClass"].to_dict(), ncd=self.ncd
        )

        return self._nuclei_metrics(out, rstroma_count)

    @staticmethod
    def _nuclei_metrics(out: dict, rst_count):
        """
        Given count summary, summarize nuclei metrics
        """
        nst = "nNuclei"
        tiln = out[f"{nst}_TILsCell"] + out[f"{nst}_ActiveTILsCell"]
        stroman = out[f"{nst}_StromalCellNOS"]
        allstroma = tiln + stroman
        metrics = {
            "nTILsCells2AnyStromaRegionArea": _divnonan(tiln, rst_count),
            "nTILsCells2nStromaCells": _divnonan(tiln, stroman),
            "nTILsCells2nAllStromaCells": _divnonan(tiln, allstroma),
            "nTILsCells2nAllCells": _divnonan(tiln, out[f"{nst}_all"]),
            "nTILsCells2nTumorCells": _divnonan(tiln, out[f"{nst}_CancerEpithelium"]),
        }

        return out, metrics

    def _save_nuclei_locs_hui_style(self, df):
        """Save nuclei centroids annotation HistomicsUI style."""
        anndoc = {
            "name": f"nucleiLocs_{self._roiname}",
            "description": "Nuclei centroids.",
            "elements": self._df_to_points_list_hui_style(df),
        }
        savename = opj(self._savedir, "annotations", f"nucleiLocs_{self._roiname}.json")
        save_json(anndoc, path=savename)

    def _df_to_points_list_hui_style(self, df):
        """
        Parse TILs score datafram to a list of Points dicts, in accordance
        with the HistomicsUI platform schema.

        This implementation is specifically made to avoid doing a python
        loop over each row, and instead relies on pandas ops which are more
        efficient.
        """
        # handle if no nuclei in ROI
        if df.shape[0] < 1:
            return []

        df.loc[:, "lineColor"] = df.loc[:, "StandardClass"].map(
            {
                cls: self._huicolor(col)
                for cls, col in self.VisConfigs.NUCLEUS_COLORS.items()
            }
        )
        records = (
            "{'type': 'point', "
            f"'group': '" + df.loc[:, "StandardClass"] + "', 'label': {"
            f"'value': '"
            + df.loc[:, "StandardClass"]
            + "'}, 'lineColor': '"
            + df.loc[:, "lineColor"]
            + "', 'lineWidth': 2, 'center': ["
            + df.loc[:, "x"].astype(int).astype(str)
            + ", "
            + df.loc[:, "y"].astype(int).astype(str)
            + ", 0]}"
        )
        records = records.apply(lambda x: ast.literal_eval(x)).to_list()

        return records

    @staticmethod
    def _huicolor(col):
        return f"rgb({','.join(str(int(j)) for j in col)})"
