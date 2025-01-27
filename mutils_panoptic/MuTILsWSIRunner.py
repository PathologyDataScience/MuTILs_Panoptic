import os
from os.path import join as opj
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from pandas import DataFrame, read_csv
from imageio import imwrite
import warnings
from glob import glob
import shutil
import logging
import pyvips
import ast
from histomicstk.preprocessing.color_deconvolution import (
    color_deconvolution_routine)

# histolab modules
from histolab.slide import Slide, SlideSet
from histolab.tile import Tile
from histolab.tiler import ScoreTiler
from histolab.types import CoordinatePair
from histolab.masks import BiggestTissueBoxMask
from histolab.filters.image_filters_functional \
    import rag_threshold

# mutils
from MuTILs_Panoptic.mutils_panoptic.MuTILsInference import \
    MutilsInferenceRunner
from MuTILs_Panoptic.utils.GeneralUtils import (
    load_json, save_json, CollectErrors, write_or_append_json_list,
    _divnonan, weighted_avg_and_std,
)
from MuTILs_Panoptic.utils.MiscRegionUtils import (
    get_objects_from_binmask, summarize_region_mask,
    summarize_nuclei_mask, numpy2vips,
)
from MuTILs_Panoptic.utils.RegionPlottingUtils import (
    get_visualization_ready_combined_mask as gvcm,
)

collect_errors = CollectErrors()


# =============================================================================


class MuTILsWSIRunner(MutilsInferenceRunner):
    """"""

    def __init__(
            self,
            # paths
            model_configs,
            model_paths: dict,
            slides_path: str,
            base_savedir: str,
            *,
            keep_slides: list = None,
            monitor="",
            # what (not) to save
            save_wsi_mask=True,
            save_annotations=False,
            save_nuclei_meta=True,
            save_nuclei_props=True,
            # roi size and scoring
            roi_side_hres=1024,
            discard_edge_hres=0,
            vlres_scorer_kws=None,
            roi_clust_mpp=20.0,  # 0.5x
            roi_kmeans_kvp=None,
            topk_rois=None,
            topk_rois_sampling_mode="weighted",
            # color normalization & augmentation
            cnorm=True,
            cnorm_kwargs=None,
            maskout_regions_for_cnorm=None,
            ntta=0, dltransforms=None,
            # misc params
            valid_extensions=None,
            logger=None,
            COHORT=None,
            N_SUBSETS=None,
            restrict_to_vta=False,
            # intra-tumoral stroma (saliency)
            filter_stromal_whitespace=False,
            min_tumor_for_saliency=4,
            max_salient_stroma_distance=64,
            topk_salient_rois=64,
            # parsing nuclei from inference
            no_watershed_nucleus_classes=None,
            min_nucl_size=5,
            max_nucl_size=90,
            nprops_kwargs=None,
            # for grand challenge platform
            grandch=False,
            gcpaths=None,
            # internal
            _debug=False,
            _reverse=False,
    ):
        super().__init__(
            model_configs=model_configs,
            roi_side_hres=roi_side_hres,
            discard_edge_hres=discard_edge_hres,
            cnorm=cnorm,
            cnorm_kwargs=cnorm_kwargs,
            ntta=ntta,
            dltransforms=dltransforms,
            no_watershed_nucleus_classes=no_watershed_nucleus_classes,
            maskout_regions_for_cnorm=maskout_regions_for_cnorm,
            filter_stromal_whitespace=filter_stromal_whitespace,
            min_tumor_for_saliency=min_tumor_for_saliency,
            max_salient_stroma_distance=max_salient_stroma_distance,
            min_nucl_size=min_nucl_size,
            max_nucl_size=max_nucl_size,
            nprops_kwargs=nprops_kwargs,
            _debug=_debug,
        )

        self.logger = logger or logging.getLogger(__name__)
        collect_errors.logger = self.logger
        collect_errors._debug = self._debug

        self.base_savedir = opj(base_savedir, 'perSlideResults')
        os.makedirs(self.base_savedir, exist_ok=True)
        self._monitor = monitor
        self.save_wsi_mask = save_wsi_mask
        self.save_annotations = save_annotations
        self.save_nuclei_meta = save_nuclei_meta
        self.save_nuclei_props = save_nuclei_props
        self.roi_clust_mpp = roi_clust_mpp
        self.topk_rois = topk_rois
        self.topk_salient_rois = topk_salient_rois
        if topk_rois is not None:
            assert topk_salient_rois <= topk_rois, (
                "The no. of salient ROIs used for final TILs scoring must be "
                "less than the total no. of rois that we do inference on!"
            )
        assert topk_rois_sampling_mode in ("stratified", "weighted", "sorted")
        self._topk_rois_sampling_mode = topk_rois_sampling_mode

        # grand-challenge platform
        self.grandch = grandch
        self.gcpaths = gcpaths
        self._cta2vta = None
        self._grand_challenge_sanity()

        # model ensembles
        assert all(os.path.isfile(j) for j in model_paths.values()), \
            "Some of the models weight files do not exist!"
        self.model_paths = model_paths
        self.n_models = len(model_paths)

        # vlres cellularity scorer & roi clustering
        self.vlres_scorer_kws = vlres_scorer_kws or {
            'check_tissue': True,
            'tissue_percent': 50,
            'pixel_overlap': int(2 * self.discard_edge_hres * self.h2vl),
        }
        self.vlres_scorer_kws.update({
            'scorer': (
                self.tile_scorer_by_tissue_ratio if self.topk_rois is None
                else self.tile_scorer_by_deconvolution
            ),
            'tile_size': (self.roi_side_vlres, self.roi_side_vlres),
            'n_tiles': 0,  # ALL tiles
            'mpp': self.vlres_mpp,
        })
        self.roi_kmeans_kvp = roi_kmeans_kvp or {
            'n_segments': 128, 'compactness': 10, 'threshold': 9}

        # slides to analyse
        self._reverse = _reverse
        self.slides = SlideSet(
            slides_path=slides_path,
            processed_path=self.base_savedir,
            valid_extensions=valid_extensions or [
                '.tif', '.tiff',  # tils challenge
                '.svs',  # TCGA
                '.scn',  # CPS cohorts
                '.ndpi',  # new cps scans
                '.mrxs',  # NHS breast
            ],
            keep_slides=keep_slides,
            reverse=self._reverse,
            slide_kwargs={'use_largeimage': True},
        )
        # internal properties that change with slide/roi/hpf (convenience)
        self._slide = None
        self._sldname = None
        self._slmonitor = None
        self._top_rois = None
        self._savedir = None
        self._modelname = None
        self._mrois = None
        self._mdmonitor = None
        self._rid = None
        self._rmonitor = None
        self._hpf = None
        self._hid = None
        self._hpfname = None

    def run_all_slides(self):
        """Run the MuTILs pipeline for all slides."""
        nslds = len(self.slides)
        for slidx, self._slide in enumerate(self.slides):
            self._sldname = self._slide.name
            self._slmonitor = (
                f"{self._monitor}slide {slidx + 1} of {nslds}: {self._sldname}"
            )
            collect_errors.monitor = self._slmonitor
            self.logger.info(f"")
            self.logger.info(f"*** {self._slmonitor} ***")
            self.logger.info(f"")
            self.run_slide()

    @collect_errors()
    def run_slide(self):
        """"""
        # when running slides in reverse order, strictly avoid running on
        # any slide if its director already exists to prevent conflicts
        self._savedir = opj(self.base_savedir, self._sldname)
        if self._reverse and os.path.exists(self._savedir):
            return

        # This must be OUTSIDE self.run_slide() for error collection
        self._set_sldmeta()
        self._run_slide()
        self.summarize_slide()
        self._maybe_concat_wsi_mask()
        self._save_slide_metrics()

    @collect_errors()
    def summarize_slide(self):
        """Get slide-level metrics."""
        metas = [
            load_json(path)
            for path in glob(opj(self._savedir, 'roiMeta', '*.json'))]
        # noinspection PyTypedDict
        self._sldmeta["metrics"] = {
            'weighted_by_rois': self._summarize_rois(metas),
            'unweighted_global': self._get_global_metrics(metas),
        }

    @collect_errors()
    def run_single_model(self):
        """Load & run a single model for all assigned rois."""
        # load model weights
        self.model_path = self.model_paths[self._modelname]
        self.load_model()
        # Iterate through one roi at a time
        nrois = len(self._mrois)
        for rno, self._rid in enumerate(self._mrois):

            if self._debug and rno > 1:
                break

            self._rmonitor = f"{self._mdmonitor}: roi {rno + 1} of {nrois}"
            collect_errors.monitor = self._rmonitor
            self.logger.info(self._rmonitor)
            self.run_roi()

    @collect_errors()
    def run_roi(self):
        """Run model over a single roi."""
        self._roiname = (
            f"{self._sldname}_roi-{self._rid}"
            f"{self._bounds2str(*self._roicoords)}"
        )
        # get and predict roi
        tile = self._slide.extract_tile(
            coords=self._top_rois[self._rid][1], mpp=self.hres_mpp,
            tile_size=(self.roi_side_hres, self.roi_side_hres),
        )
        rgb, preds = self._predict_roi(tile=tile)
        # summarize & save metadata
        self._summarize_roi(preds)

    def tile_scorer_by_tissue_ratio(self, tile: Tile):
        """"""
        try:
            ignore, _ = self._get_tile_ignore(
                tile, filter_tissue=True, get_lres=False)

            ratio = float(np.mean(1 - ignore))
            return (
                ratio if ratio > self.vlres_scorer_kws['tissue_percent'] * 0.01
                else 0.
            )

        except Exception as e:
            self.logger.debug(str(e.__repr__()))

            return 0.

    def tile_scorer_by_deconvolution(self, tile: Tile):
        """
        Simple scoring of a Tile by hematoxylin channel.
        """
        try:
            ignore, _ = self._get_tile_ignore(
                tile, filter_tissue=True, get_lres=False)
        except Exception as e:
            self.logger.debug(str(e.__repr__()))
            ignore = None
        try:
            stains, _, _ = color_deconvolution_routine(
                im_rgb=np.uint8(tile.image), mask_out=ignore,
                stain_unmixing_method='macenko_pca')
            stains = (255 - stains) / 255
            # score is maximized for intratumoral stroma (both htx & eosin)
            score = float(np.mean(stains[..., 0]) * np.mean(stains[..., 1]))
        except Exception as e:
            self.logger.debug(str(e.__repr__()))
            score = 0.

        return score

    # HELPERS -----------------------------------------------------------------

    def _save_slide_metrics(self):
        """
        Save slide level results.
        """
        self.logger.info(f"{self._slmonitor}: Done, saving results.")
        new_json = True
        if self.grandch:
            self._save_single_til_score()
            if self.gcpaths['result_file'] is not None:
                write_or_append_json_list(
                    self._sldmeta, path=self.gcpaths['result_file'])
                new_json = False
        if new_json:
            save_json(
                self._sldmeta,
                path=opj(self._savedir, self._slide.name + '.json'))

    @collect_errors()
    def _save_single_til_score(self):
        """
        Save single til score json for grand-challenge.
        """
        metric = 'Mean_nTILsCells2nAllStromaCells'
        if self._cta2vta is not None:
            metric = 'Calibrated' + metric
        where = (
            self.gcpaths['tilscore_file']
            if self.gcpaths['tilscore_file'] is not None
            else opj(self._savedir, 'til-score.json')
        )
        score = 100. * self._sldmeta['metrics']['weighted_by_rois'][metric]
        save_json(score, path=where)
        self._sldmeta['outputs'].append(str(where))

    @collect_errors()
    def _maybe_concat_wsi_mask(self):
        """
        concatenate roi combined masks into a WSI level prediction.
        """
        if not self.save_wsi_mask:
            return

        if self.grandch:
            where = self.gcpaths['wsi_mask'] or self._savedir
        else:
            where = self._savedir
        savename = opj(where, self._slide.name + '.tif')

        # if exists, dont overwrite!!
        if os.path.isfile(savename):
            self.logger.warning(f"{self._slmonitor}: WSI mask already exists!")
            return

        # get "empty" wsi mask (tissue mask is delineated)
        wsimask = self._init_wsi_mask()
        # Read tiles and append to mask
        self.logger.info(f"{self._slmonitor}: Insterting tiles to WSI mask.")
        mpp = self.lres_mpp if self.grandch else self.hres_mpp
        sf = self._slide.base_mpp / mpp
        mask_paths = glob(opj(self._savedir, 'roiMasks', '*.png'))
        for midx, mpath in enumerate(mask_paths):
            self._rid = int(mpath.split('_roi-')[-1].split('_')[0])
            left, top, _, _ = [int(j * sf) for j in self._roicoords]
            mtile = pyvips.Image.new_from_file(mpath, access="sequential")
            wsimask = wsimask.insert(mtile, left, top, expand=True,
                                     background=0)
        # save result
        self.logger.info(f"{self._slmonitor}: Saving WSI mask.")
        pixel_per_mm = 1 / (1e-3 * mpp)
        wsimask.tiffsave(
            savename,
            tile=True, tile_width=512, tile_height=512, pyramid=True,
            compression='lzw', Q=100,  # must be lzw to preserve pixel values!
            xres=pixel_per_mm, yres=pixel_per_mm,
        )
        self._sldmeta['outputs'].append(str(savename))
        # maybe cleanup. No need to keep both tile masks and WSI mask!
        if not self._debug:
            shutil.rmtree(opj(self._savedir, 'roiMasks'))

    def _init_wsi_mask(self):
        """
        Initialize WSI mask using pyvips.
        """
        # first get tissue mask
        self.logger.info(f"{self._slmonitor}: initializing WSI mask.")
        shape = list(self._slide.thumbnail.size)[::-1] + [1]
        # Make a vips image from the tissue mask, resize to hres mpp & save
        # See: https://github.com/libvips/pyvips/issues/167
        # and: https://github.com/libvips/pyvips/issues/69
        # and: https://github.com/libvips/pyvips/issues/164
        # and: https://github.com/libvips/pyvips/issues/40
        thumb_mask = np.zeros(shape, dtype=np.uint8)
        if self.grandch:
            # grand challenge mask is low res stromal regions
            sf1 = self._slide.base_mpp / self.lres_mpp
        else:
            # analysis is done at level of hpfs and mask includes nuclei
            sf1 = self._slide.base_mpp / self.hres_mpp
        wsimask = numpy2vips(thumb_mask)
        # resize to desired level
        to_width, to_height = [sf1 * j for j in self._slide.dimensions]
        sf2 = to_height / thumb_mask.shape[0]
        wsimask = wsimask.resize(sf2, kernel="nearest")

        return wsimask

    def _get_global_metrics(self, metas):
        """
        Get global slide-level metrics (i.e. NOT averaged from rois).
        """
        # regions
        pxc = 'pixelCount_'
        r_pixsums = self._ksums(
            metas, k0='region_summary',
            ks=[j for j in metas[0]['region_summary'] if j.startswith(pxc)],
        )
        _, metrics = self._region_metrics(r_pixsums, nrois=len(metas))
        # nuclei
        n_counts = self._ksums(
            metas, k0='nuclei_summary',
            ks=list(metas[0]['nuclei_summary'].keys())
        )
        _, nmetrics = self._nuclei_metrics(
            n_counts, rst_count=r_pixsums['pixelCount_AnyStroma'])
        metrics.update(nmetrics)

        # calibrate to vta range
        if self._cta2vta is not None:
            metrics.update(
                self._calibrate_metrics(metrics, pfx='CTA.Global_')
            )

        return metrics

    def _summarize_rois(self, metas):
        """
        Get avg and std of roi metadata, weighted by saliency.
        """
        metrics = DataFrame.from_records([j['metrics'] for j in metas])
        summ = {'n_rois': metrics.shape[0]}
        # restrict to topk salient rois
        metrics = metrics.sort_values('SaliencyScore', ascending=False)
        if self.topk_salient_rois is not None:
            metrics = metrics[:self.topk_salient_rois]
        # keep relevant columns
        saliency = np.float32(metrics.loc[:, 'SaliencyScore'])
        cols = [j for j in metrics.columns if j != 'SaliencyScore']
        metrics = metrics.loc[:, cols]
        # metrics are weighted by saliency (saliency itself is unweighted)
        means = {'SaliencyScore': float(np.mean(saliency))}
        stds = {'SaliencyScore': float(np.std(saliency))}
        max_saliency = saliency.max()
        if max_saliency > 0:
            saliency = saliency / max_saliency
        else:
            saliency = np.ones(saliency.size)
        for k in cols:
            wmn, wst = weighted_avg_and_std(metrics.loc[:, k].values, saliency)
            means[k], stds[k] = float(wmn), float(wst)
        # now collect
        summ.update({'n_salient_rois': metrics.shape[0]})
        summ.update({f'Mean_{k}': v for k, v in means.items()})
        summ.update({f'Std_{k}': v for k, v in stds.items()})

        # calibrate to vta range
        if self._cta2vta is not None:
            summ.update(
                self._calibrate_metrics(summ, pfx='CTA.SaliencyWtdMean_')
            )

        return summ

    @collect_errors()
    def _calibrate_metrics(self, cta: dict, pfx=''):
        """
        Calibrate computational to visual TILs assessment scores.
        """

        def strip(metric):
            return metric.replace('Mean_', '').replace('Std_', '')

        return {
            f"Calibrated{metric}":
                val * self._cta2vta[f"{pfx}{strip(metric)}"]["slope"]
            for metric, val in cta.items()
            if f"{pfx}{strip(metric)}" in self._cta2vta
        }

    @collect_errors()
    def _maybe_fix_wsi_tilslocs_gch_style(self):
        """
        Fix and save slide-level TILs location detections.
        """
        where = opj(self._savedir, 'tmp_tilsLocs.json')
        if not os.path.isfile(where):
            return

        tils_wsi = {
            'name': self._sldname,
            'version': {'major': 1, 'minor': 0},
            'type': 'Multiple points',
            'points': [],
        }
        tmplocs = load_json(where)
        for locs in tmplocs:
            tils_wsi['points'].extend(locs['tmp'])
        path = (
            opj(self._savedir, 'annotations', 'detected-lymphocytes.json')
            if self.gcpaths['detect_file'] is None
            else self.gcpaths['detect_file']
        )
        save_json(tils_wsi, path=path)
        self._sldmeta['outputs'].append(str(self.gcpaths['detect_file']))
        if not self._debug:
            os.remove(where)

    def _summarize_roi(self, preds):
        """
        Get metrics from roi.
        """
        rgsumm, metrics = self._get_region_metrics(
            preds['mask'][..., 0], sstroma=preds['sstroma'])
        nusumm, nmetrics = self._get_nuclei_metrics(
            preds['cldf'], rstroma_count=rgsumm['pixelCount_AnyStroma'])
        metrics.update(nmetrics)
        left, top, right, bottom = self._roicoords
        meta = {
            'slide_name': self._sldname,
            'roi_id': self._rid,
            'roi_name': self._roiname,
            'wsi_left': left,
            'wsi_top': top,
            'wsi_right': right,
            'wsi_bottom': bottom,
            'mpp': self.hres_mpp,
            'model_name': self._modelname,
            'metrics': metrics,
            'region_summary': rgsumm,
            'nuclei_summary': nusumm,
        }
        save_json(meta, path=opj(
            self._savedir, 'roiMeta', self._roiname + '.json'))

    def _get_region_metrics(self, mask, sstroma):
        """
        Summarize region mask and some metrics.
        """
        # pixel summaries
        out = summarize_region_mask(mask, rcd=self.rcd)
        out['pixelCount_EXCLUDE'] -= self.n_edge_pixels_discarded
        # isolate salient tils
        salient_tils = mask == self.rcd['TILS']
        salient_tils[~sstroma] = False
        p = 'pixelCount'
        out.update({
            f'{p}_SalientStroma': int(sstroma.sum()),
            f'{p}_SalientTILs': int(salient_tils.sum()),
        })

        return self._region_metrics(out)

    def _region_metrics(self, out: dict, nrois=1):
        """
        Given pixel count summary, summarize region metrics
        """
        # summarize metrics
        p = 'pixelCount'
        everything = nrois * (
                (self.roi_side_hres ** 2) - self.n_edge_pixels_discarded)
        junk = out[f'{p}_JUNK'] + out[f'{p}_WHITE'] + out[f'{p}_EXCLUDE']
        nonjunk = everything - junk
        anystroma = out[f'{p}_STROMA'] + out[f'{p}_TILS']
        out[f'{p}_AnyStroma'] = anystroma
        # Saliency score is higher when there's more tumor & salient stroma
        # Note that relying on stroma within x microns from tumor edge is not
        # enough as ROIs with scattered tumor nests that are spaced out would
        # get a high score even though there's little tumor or maybe even
        # a few misclassified pixels here and there. So we use both.
        sscore = out[f'{p}_SalientStroma'] / everything
        sscore *= out[f'{p}_TUMOR'] / everything
        metrics = {
            'SaliencyScore': sscore,
            'TissueRatio': nonjunk / everything,
            'TILs2AnyStromaRatio': _divnonan(out[f'{p}_TILS'], anystroma),
            'TILs2AllRatio': _divnonan(out[f'{p}_TILS'], nonjunk),
            'TILs2TumorRatio': _divnonan(out[f'{p}_TILS'], out[f'{p}_TUMOR']),
            'SalientTILs2StromaRatio': _divnonan(
                out[f'{p}_SalientTILs'], out[f'{p}_SalientStroma']),
        }

        return out, metrics

    def _get_nuclei_metrics(self, cldf, rstroma_count):
        """
        Summarize nuclei mask and some metrics.
        """
        out = summarize_nuclei_mask(
            cldf.loc[:, 'Classif.StandardClass'].to_dict(), ncd=self.ncd)

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
            'nTILsCells2AnyStromaRegionArea': _divnonan(tiln, rst_count),
            'nTILsCells2nStromaCells': _divnonan(tiln, stroman),
            'nTILsCells2nAllStromaCells': _divnonan(tiln, allstroma),
            'nTILsCells2nAllCells': _divnonan(tiln, out[f"{nst}_all"]),
            'nTILsCells2nTumorCells': _divnonan(
                tiln, out[f"{nst}_CancerEpithelium"]),
        }

        return out, metrics

    @property
    def _roicoords(self):
        return [int(j) for j in self._top_rois[self._rid][1]]

    def _predict_roi(self, tile: Tile):
        """Get MuTILS predictions for one ROI."""
        hres_ignore, lres_ignore = self._get_tile_ignore(tile)
        rgb = self.maybe_color_normalize(
            tile.image.convert('RGB'), mask_out=hres_ignore)
        inference = self.do_inference(rgb=rgb, lres_ignore=lres_ignore)
        preds = self.refactor_inference(inference, hres_ignore=hres_ignore)
        preds['sstroma'] = self.get_salient_stroma_mask(
            preds['combined_mask'][..., 0])
        self._maybe_save_roi_preds(rgb=rgb, preds=preds)
        preds = self._simplify_roi_preds(preds)

        return rgb, preds

    @collect_errors()
    def _maybe_save_roi_preds(self, rgb, preds):
        """Save masks, metadata, etc."""
        if self.save_wsi_mask:
            mask = self._maybe_binarize_roi_mask(preds['combined_mask'].copy())
            imwrite(
                opj(self._savedir, 'roiMasks', self._roiname + '.png'), mask
            )
        self._maybe_visualize_roi(
            rgb, mask=preds['combined_mask'], sstroma=preds['sstroma'])
        if self.save_nuclei_meta:
            preds['classif_df'].to_csv(
                opj(self._savedir, 'nucleiMeta', self._roiname + '.csv'),
            )
        self._maybe_save_nuclei_props(rgb=rgb, preds=preds)
        self._maybe_save_nuclei_annotation(preds['classif_df'])

    def _maybe_binarize_roi_mask(self, mask):
        """
        If grand challenge, binarize region stromal mask.
        """
        if not self.grandch:
            return mask

        # binarize regions (stroma vs else)
        shape = mask.shape[:2]
        mask = np.in1d(
            mask[..., 0], [self.rcd['STROMA'], self.rcd['TILS']])
        mask = Image.fromarray(mask.reshape(shape))
        # resize to lres
        mask = mask.resize(
            (self.roi_side_lres, self.roi_side_lres), Image.NEAREST)
        mask = np.array(mask, dtype=np.uint8)

        return mask

    @collect_errors()
    def _maybe_save_nuclei_annotation(self, classif_df):
        """
        Save nuclei locations annotation.
        """
        if not self.save_annotations:
            return

        # fix coords to wsi base magnification
        left, top, _, _ = self._roicoords
        colmap = {
            'Identifier.CentroidX': 'x',
            'Identifier.CentroidY': 'y',
            'Classif.StandardClass': 'StandardClass',
            'Classif.SuperClass': 'SuperClass',
        }
        df = classif_df.loc[:, list(colmap.keys())].copy()
        df.rename(columns=colmap, inplace=True)
        df.loc[:, ['x', 'y']] *= self.hres_mpp / self._slide.base_mpp
        df.loc[:, 'x'] += left
        df.loc[:, 'y'] += top
        # parse to preferred style
        if self.grandch:
            istils = df.loc[:, 'SuperClass'] == 'TILsSuperclass'
            self._save_tilslocs_gch_style(df.loc[istils, ['x', 'y']])
        else:
            self._save_nuclei_locs_hui_style(df)

    def _save_nuclei_locs_hui_style(self, df):
        """Save nuclei centroids annotation HistomicsUI style."""
        anndoc = {
            'name': f"nucleiLocs_{self._roiname}",
            'description': 'Nuclei centroids.',
            'elements': self._df_to_points_list_hui_style(df),
        }
        savename = opj(
            self._savedir, 'annotations', f'nucleiLocs_{self._roiname}.json')
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

        df.loc[:, 'lineColor'] = df.loc[:, 'StandardClass'].map({
            cls: self._huicolor(col)
            for cls, col in self.cfg.VisConfigs.NUCLEUS_COLORS.items()
        })
        records = (
                "{'type': 'point', " f"'group': '"
                + df.loc[:, 'StandardClass']
                + "', 'label': {" f"'value': '"
                + df.loc[:, 'StandardClass']
                + "'}, 'lineColor': '"
                + df.loc[:, 'lineColor']
                + "', 'lineWidth': 2, 'center': ["
                + df.loc[:, 'x'].astype(int).astype(str)
                + ", "
                + df.loc[:, 'y'].astype(int).astype(str)
                + ", 0]}"
        )
        records = records.apply(lambda x: ast.literal_eval(x)).to_list()

        return records

    def _save_tilslocs_gch_style(self, tils):
        """Save temporary TILs location annotation (fixed for wsi at end)."""
        tils = self._df_to_points_list_gch_style(tils)
        write_or_append_json_list(
            {'tmp': tils}, path=opj(self._savedir, 'tmp_tilsLocs.json'))

    @staticmethod
    def _df_to_points_list_gch_style(df):
        """
        Parse TILs score datafram to a list of Points dicts, in accordance
        with the grandchallenge platform schema:
          https://github.com/comic/grand-challenge.org/blob/ ...
          90d825737065aa2b4df4344d5fe4c0646472dc87/app/grandchallenge/ ...
          reader_studies/models.py#L979-L991

        This implementation is specifically made to avoid doing a python
        loop over each row, and instead relies on pandas ops which are more
        efficient.
        """
        # handle if no TILs in ROI
        if df.shape[0] < 1:
            return []

        # Multiple Points object is a list of Points objects
        df = df.astype(int).astype(str)
        df.loc[:, 'pre'] = "{'point':["
        df.loc[:, 'post'] = ", 0]}"
        records = df.loc[:, 'pre'] \
                  + df.loc[:, 'x'] + ", " + df.loc[:, 'y'] + df.loc[:, 'post']
        records = records.apply(lambda x: ast.literal_eval(x)).to_list()

        return records

    @collect_errors()
    def _maybe_save_nuclei_props(self, rgb, preds):
        """Get and save nuclear morphometric features."""
        if not self.save_nuclei_props:
            return
        self.logger.info(self._rmonitor + ": getting nuclei props ..")
        props = self.get_nuclei_props_df(rgb=np.uint8(rgb), preds=preds)
        props.to_csv(
            opj(self._savedir, 'nucleiProps', self._roiname + '.csv'),
        )

    @collect_errors()
    def _maybe_visualize_roi(self, rgb, mask, sstroma):
        """
        Plot and save roi visualization.
        """
        if not self._debug:
            return

        fig, ax = plt.subplots(1, 2, figsize=(7. * 2, 7.))
        ax[0].imshow(rgb)
        ax[0].imshow(
            np.ma.masked_array(sstroma, mask=~sstroma), alpha=0.3,
            cmap=ListedColormap([[0.01, 0.74, 0.25]]),
        )
        ax[1].imshow(
            gvcm(mask), cmap=self.cfg.VisConfigs.COMBINED_CMAP,
            interpolation='nearest')
        plt.tight_layout(pad=0.4)
        plt.savefig(opj(self._savedir, 'roiVis', self._roiname + '.png'))
        plt.close()

    @staticmethod
    def _simplify_roi_preds(preds):
        """Keep only what's needed and simplify terminology."""
        keep_cols = [
            c for c in preds['classif_df'].columns
            if not c.startswith('Unconstrained.')]
        return {
            'mask': preds['combined_mask'],
            'sstroma': preds['sstroma'],
            'cldf': preds['classif_df'].loc[:, keep_cols],
        }

    def _get_tile_ignore(
            self, tile: Tile, filter_tissue=False, get_lres=True):
        """Get region outside tissue (eg. marker pen & white space)"""
        tile._filter_tissue = filter_tissue
        hres_ignore = ~tile._tissue_mask
        hres_ignore, _ = get_objects_from_binmask(
            hres_ignore, minpixels=128, use_watershed=False)
        hres_ignore = Image.fromarray(hres_ignore)
        if get_lres:
            lres_ignore = hres_ignore.resize(
                (self.roi_side_lres, self.roi_side_lres), Image.NEAREST)
            lres_ignore = np.array(lres_ignore, dtype=bool)
        else:
            lres_ignore = None
        hres_ignore = np.array(hres_ignore, dtype=bool)

        return hres_ignore, lres_ignore

    @collect_errors()
    def _run_slide(self):
        """Run the MuTILs pipeline for one slide."""
        self._create_slide_dirs()
        model_rois = self._load_or_extract_roi_locs()
        self._maybe_save_roilocs_annotation(model_rois)
        mno = -1
        for self._modelname, self._mrois in model_rois.items():

            mno += 1
            if self._debug and mno > 0:
                break

            self._mdmonitor = f"{self._slmonitor}: {self._modelname}"
            collect_errors.monitor = self._mdmonitor
            self.logger.info(self._mdmonitor)
            self.run_single_model()

        collect_errors.monitor = self._slmonitor
        self._maybe_fix_wsi_tilslocs_gch_style()

    def _assign_rois_to_models(self):
        """
        Assign rois to trained model folds (as a form of ensembling).
        """
        self.logger.info(f"{self._slmonitor}: assign rois to models ..")
        # get topk rois, stratified by superpixel in vlres adjacency image
        df = self._assign_rois_to_rag()
        top_idxs = self._pick_topk_rois_from_rag(df)
        if self.grandch:
            # Stratified grid-like assignment of rois to models
            # -> Multiple models can be assigned to the same "superpixel"
            # This is a good arrangement when you don't care about "fusing"
            # the mask from multiple adjacent tiles and you only care about
            # each tile independently (eg. TIL score frome each tile)
            df.loc[top_idxs, 'model'] = [
                i % self.n_models for i in range(len(top_idxs))]
        else:
            # Smart assignment based on region adjacency to reduce edge artif.
            # (eg model overcalling tumor vs another overcalling stroma)
            # -> Each "superpixel" gets only one model assigned to it
            # This is a good arrangement when you care about fusing the mask
            # from multiple tiles to get whole-slide region polygons for
            # research assessment
            df.loc[top_idxs, 'model'] = df.loc[:, 'rag'] % self.n_models

        return df

    def _pick_topk_rois_from_rag(self, full_df):
        """Stratified sampling of cellular rois from different RAG regions."""
        # some sanity
        df = full_df.copy()
        df = df.loc[df.loc[:, 'rag'] > 0, :]
        df = df.loc[df.loc[:, 'score'] > 0, :]
        df.sort_values('score', axis=0, ascending=False, inplace=True)

        # maybe we want to analyze ALL rois
        if self.topk_rois is None:
            return list(df.index)

        if self._topk_rois_sampling_mode == "stratified":
            # stratified sampling, emphasizing both diversity & cellularity
            return self._stratified_topk_rois(df)

        elif self._topk_rois_sampling_mode == "weighted":
            # weighted sampling of rois by cellularity
            return self._weighted_sampling_topk_rois(df)

        elif self._topk_rois_sampling_mode == "sorted":
            # just get the top k rois by cellularity
            return list(df.index)[:self.topk_rois]

        else:
            sm = self._topk_rois_sampling_mode
            raise ValueError(f"Unknown self._topk_rois_sampling_mode: {sm}")

    def _weighted_sampling_topk_rois(self, df):
        """Wtd sampling by cellu. -- may pick many rois from same region."""
        probs = np.float32(df.loc[:, 'score'])
        probs = probs / np.sum(probs)
        sampled = np.random.choice(
            list(df.index), p=probs, replace=False,
            size=min(self.topk_rois, probs.shape[0]),
        )
        return sampled.tolist()

    def _stratified_topk_rois(self, df):
        """Stratified sampling -- emphasize variability."""
        nrags = len(np.unique(df.loc[:, 'rag']))
        leftover_idxs = list(df.index)
        top_idxs = []
        max_rois = min(self.topk_rois, df.shape[0])
        repeat = True
        while repeat:
            idxs = leftover_idxs.copy()
            done = set()
            for idx in idxs:
                row = df.loc[idx, :]
                if row['rag'] not in done:
                    top_idxs.append(idx)
                    done.add(row['rag'])
                    leftover_idxs.remove(idx)
                if len(top_idxs) == max_rois:
                    repeat = False
                    break
                if len(done) == nrags:
                    break
        df = df.loc[top_idxs, :]
        df.sort_values('score', axis=0, ascending=False, inplace=True)

        return list(df.index)

    def _assign_rois_to_rag(self):
        """
        Assign each roi to a region from the slide RAG mask.
        """
        # get the region adjacency labeled mask
        rag = self._get_slide_region_adjacency()
        # parse roi dataframe
        df = DataFrame(
            index=np.arange(len(self._top_rois)),
            columns=['xmin', 'ymin', 'xmax', 'ymax', 'rag', 'score', 'model'],
        )
        df.loc[:, 'model'] = -1
        sf = rag.shape[1] / self._slide.dimensions[0]
        for rid, (score, tr) in enumerate(self._top_rois):
            xmin = max(int(tr.x_ul * sf), 0)
            ymin = max(int(tr.y_ul * sf), 0)
            xmax = min(int(tr.x_br * sf), rag.shape[1])
            ymax = min(int(tr.y_br * sf), rag.shape[0])
            tally = rag[ymin:ymax, xmin:xmax].ravel()
            unique, count = np.unique(
                tally[tally >= 0], return_counts=True)
            # now assign
            df.loc[rid, 'xmin'] = tr.x_ul
            df.loc[rid, 'ymin'] = tr.y_ul
            df.loc[rid, 'xmax'] = tr.x_br
            df.loc[rid, 'ymax'] = tr.y_br
            try:
                df.loc[rid, 'rag'] = int(unique[np.argmax(count)])
            except ValueError:
                df.loc[rid, 'rag'] = 0
            df.loc[rid, 'score'] = score

        return df

    def _get_slide_region_adjacency(self):
        """
        Segment an image with K-means, build region adjacency graph based on
        on the segments, combine similar regions based on threshold value,
        and then output these resulting region segments. Eseentially, this
        breaks down the slide into very large "superpixel" chunks.
        """
        # get slide and mask at saliency mpp
        sf = self.roi_clust_mpp / self._slide.base_mpp
        rgb = self._slide.scaled_image(sf)
        mask = Image.fromarray(
            BiggestTissueBoxMask()._thumb_mask(self._slide))
        mask = np.uint8(mask.resize(rgb.size[:2]))
        # maybe cluster
        if self._topk_rois_sampling_mode == "stratified":
            self.logger.info(f"{self._slmonitor}: get wsi region adjacency")
            segments = rag_threshold(
                img=rgb, mask=mask, return_labels=True, **self.roi_kmeans_kvp)
        else:
            segments = mask

        self._maybe_vis_rag(segments)

        return segments

    @collect_errors()
    def _maybe_vis_rag(self, rag):
        """Maybe visualize region adjacency graph used.

        Note that we cannot save the RAG itself as we'd to convert to uint8
        which is lossy since there's usually a lot more than 255 regions.
        """
        if not self._debug:
            return

        plt.figure(figsize=(7, 7))
        plt.imshow(
            np.ma.masked_array(rag, rag == 0),
            cmap='tab20', interpolation='nearest')
        plt.tight_layout()
        plt.savefig(opj(self._savedir, f"{self._sldname}_RAGraph.png"))
        plt.close()

    @collect_errors()
    def _maybe_exclude_some_rois(self, roi_models: list):
        """If an roi has already been predicted, exclude it."""
        roi_models = np.array(roi_models)
        done = [
            int(j.split('_roi-')[1].split('_')[0])
            for j in os.listdir(opj(self._savedir, 'roiMeta'))
        ]
        roi_models[done] = -1
        return roi_models

    @collect_errors()
    def _maybe_save_roilocs_annotation(self, model_rois):
        """
        Save ROI bounding box annotation.
        """
        if not self.save_annotations:
            return

        savenames = (
            self._save_roilocs_gch_style() if self.grandch
            else self._save_roilocs_hui_style(model_rois)
        )
        self._sldmeta['outputs'].extend([str(j) for j in savenames])

    def _save_roilocs_gch_style(self):
        """
        Grand-challenge compatible roi locations annotation.
        """
        roiboxes = {
            'version': {'major': 1, 'minor': 0},
            'type': 'Multiple 2D bounding boxes',
            'boxes': [
                {
                    "name": "%.4f" % score,
                    "corners": [
                        [int(rloc.x_ul), int(rloc.y_ul), 0],
                        [int(rloc.x_br), int(rloc.y_ul), 0],
                        [int(rloc.x_br), int(rloc.y_br), 0],
                        [int(rloc.x_ul), int(rloc.y_br), 0],
                    ],
                }
                for (score, rloc) in self._top_rois
            ],
        }
        path = self.gcpaths['roilocs_out'] or opj(
            self._savedir, 'annotations', f'roisLocs.json')
        save_json(roiboxes, path=path)

        return [path]

    @staticmethod
    def _huicolor(col):
        return f"rgb({','.join(str(int(j)) for j in col)})"

    def _save_roilocs_hui_style(self, model_rois):
        """
        HistomicsUI compatible roi locations annotation.
        """
        colors = np.array(plt.get_cmap('tab10').colors)[:self.n_models] * 255
        colors = [self._huicolor(col) for col in colors]
        mid = -1
        savenames = []
        for modelname, mrois in model_rois.items():
            mid += 1
            anndoc = {
                'name': f"roiLocs_{modelname}",
                'description': 'Regions of interest locations.',
                'elements': [],
            }
            for rid in mrois:
                score, rloc = self._top_rois[rid]
                width = int(rloc.x_br - rloc.x_ul)
                height = int(rloc.y_br - rloc.y_ul)
                anndoc['elements'].append({
                    'type': 'rectangle',
                    'group': 'roi',
                    'label': {'value': score},
                    'lineColor': f"{colors[mid]}",
                    'lineWidth': 2,
                    'center': [
                        int(rloc.x_ul) + width // 2,
                        int(rloc.y_ul) + height // 2,
                        0
                    ],
                    'width': width,
                    'height': height,
                    'rotation': 0,
                })
            savename = opj(
                self._savedir, 'annotations', f'roisLocs_{modelname}.json')
            save_json(anndoc, path=savename)
            savenames.append(savename)

        return savenames

    def _load_or_extract_roi_locs(self):
        """Find large cellular regions for analysis."""
        self.logger.info(f"{self._slmonitor}: extract rois ..")
        report_path = opj(self._savedir, f"{self._sldname}_RoiLocs.csv")
        if os.path.isfile(report_path):
            roi_models = self._load_roi_assignment(report_path)
        else:
            roi_models = self._simple_score_rois_for_analysis(report_path)
        roi_models = self._maybe_exclude_some_rois(roi_models)
        model_rois = {
            mn: np.argwhere(roi_models == mid)[:, 0].tolist()
            for mid, mn in enumerate(list(self.model_paths.keys()))
        }

        return model_rois

    def _simple_score_rois_for_analysis(self, report_path):
        """
        Score roi cellularity at very low resolution.
        """
        roi_extractor = ScoreTiler(**self.vlres_scorer_kws)

        # score and maybe visualize roi locations
        self.logger.info(f"{self._slmonitor}: Scoring slide tiles ..")
        self._top_rois = roi_extractor.extract(
            slide=self._slide, save_tiles=False,
            monitor=self._slmonitor, logfreq=256)
        # maybe append prespecified roi locations (grand-challenge)
        self._maybe_append_roilocs_from_m2db()
        # assign to various models for ensembling
        roi_df = self._assign_rois_to_models()
        self._save_roi_assignment(report_path, roi_df=roi_df)
        # maybe visualize for sanity
        roi_models = np.array(roi_df.loc[:, 'model'], dtype=int)
        self._maybe_vis_roilocs(roi_extractor, models=roi_models)

        return roi_models

    @collect_errors()
    def _maybe_append_roilocs_from_m2db(self):
        """
        Maybe get ROI locations to analyse from prespecified file.
        This is used by grand challenge organizers to make sure some
        parts of the slide are analyzed to be able to assess the stromal
        mask.
        """
        if not self.grandch:
            return

        fpath = self.gcpaths['roilocs_in']
        if (fpath is None) or (not os.path.isfile(fpath)):
            return

        self.logger.info(
            f"{self._slmonitor}: Appending prespecified roi locations.")
        roiboxes = load_json(fpath)
        for box in roiboxes['boxes']:
            corners = box['corners']
            cpair = CoordinatePair(
                corners[0][0], corners[0][1],  # xmin, ymin
                corners[2][0], corners[2][1],  # xmax, ymax
            )
            self._top_rois.append((-1, cpair))

    @collect_errors()
    def _maybe_vis_roilocs(self, extractor: ScoreTiler, models):
        """
        Visualize ROI locations.
        """
        if self.grandch and not self._debug:
            return

        self.logger.info(f"{self._slmonitor}: visualizing roi locations.")
        unique_colors = np.array(plt.get_cmap('tab10').colors)
        colors = unique_colors[models + 1]
        colors = colors[models >= 0]
        colors = [tuple(j) for j in np.int32(colors * 255)]
        roiloc_vis = extractor.locate_tiles(
            slide=self._slide,
            tiles=[j for i, j in enumerate(self._top_rois) if models[i] >= 0],
            scale_factor=64, alpha=255, linewidth=2, outline=colors)
        savename = opj(self._savedir, f"{self._sldname}_RoiLocs.png")
        roiloc_vis.save(savename)
        self._sldmeta['outputs'].append(str(savename))

    @collect_errors()
    def _save_roi_assignment(self, report_path, roi_df):
        """Save roi locs and assignment to models."""
        self.logger.info(f"{self._slmonitor}: save roi locs ..")
        roi_df.to_csv(report_path)
        self._sldmeta['outputs'].append(str(report_path))

    def _load_roi_assignment(self, report_path):
        """Use already extracted roi locations from a previous partial run."""
        reportdf = read_csv(report_path)
        self._top_rois = []
        roi_models = []
        for _, row in reportdf.iterrows():
            cpair = CoordinatePair(
                row['xmin'], row['ymin'], row['xmax'], row['ymax'])
            self._top_rois.append((row['score'], cpair))
            roi_models.append(int(row['model']))

        return roi_models

    def _create_slide_dirs(self):
        """"""
        has_annots = self.save_annotations
        if self.grandch:
            has_annots = has_annots and any(
                self.gcpaths[k] is None
                for k in ('roilocs_out', 'detect_file')
            )
        for condition, dirname in (
                (True, 'roiMeta'),
                (self.save_wsi_mask, 'roiMasks'),
                (self._debug, 'roiVis'),
                (has_annots, 'annotations'),
                (self.save_nuclei_meta, 'nucleiMeta'),
                (self.save_nuclei_props, 'nucleiProps'),
        ):
            if condition:
                os.makedirs(opj(self._savedir, dirname), exist_ok=True)

    def _set_sldmeta(self):
        """This must be run FIRST since erros will be collected here."""
        collect_errors.reset()
        self._sldmeta = {
            'inputs': [self._slide._path],
            'meta': {
                'slide_name': self._sldname,
                'base_mpp': self._slide.base_mpp,
            },
            'metrics': {},
            'error_messages': collect_errors.msgs,
            'outputs': [],
        }

    def _grand_challenge_sanity(self):
        """
        Warnings, sanity checks etc for grand-challenge platform.
        """
        # maybe we're not on grand-challenge
        if not self.grandch:
            return

        # no need for nuclei metadata if grand challenge
        self.save_nuclei_meta = False
        self.save_nuclei_props = False

        # assign defaults
        in_paths = [
            'roilocs_in',  # needed ROI locations (input)
            'cta2vta',  # linear calibration from CTA to VTA
        ]
        out_paths = [
            'roilocs_out',  # slide ROI locations (output)
            'result_file',  # slide metrics
            'tilscore_file',  # slide float tils score
            'detect_file',  # slide tils detections
            'wsi_mask',  # slide mask
        ]
        inout_paths = in_paths + out_paths
        self.gcpaths = self.gcpaths or {}
        for k in inout_paths:
            if k not in self.gcpaths:
                self.gcpaths[k] = None

        # sanity
        if any(j is not None for j in ['roilocs_out', 'detect_file']):
            self.save_annotations = True
        if self.gcpaths['wsi_mask'] is not None:
            os.makedirs(self.gcpaths['wsi_mask'], exist_ok=True)
        if self.gcpaths['cta2vta'] is not None:
            self._cta2vta = load_json(self.gcpaths['cta2vta'])

        # warnings
        warnings.warn(
            "This assumes that only ONE slide is run at a time "
            "as the grand-challenge platform does and REQUIRES. "
            "The following files will be OVERWRITTEN for each "
            f"slide being run!: "
            f"{[v for k, v in self.gcpaths.items() if k in out_paths]}"
        )
        warnings.warn(
            "** IMPORTANT NOTE **: "
            "Gr. challenge expects a binary WSI mask (1=stroma, 0=else). "
            "This binarization completely ignores the concept of areas "
            "not analysed! Zero here could either mean the region was not "
            "analysed by the algorithm or it is analysed and determined "
            "not to be stroma! We discussed this with the challenge "
            "organizers and they said they still prefer a binary mask "
            "and they'd only calculate the DICE metrics for the regions "
            "within the slide that they know there is algorithmic output, "
            "by definition, since they'd prespecify ROI locations for "
            "segmentation."
        )


# =============================================================================


if __name__ == "__main__":

    from MuTILs_Panoptic.configs.MuTILsWSIRunConfigs import RunConfigs
    import argparse

    parser = argparse.ArgumentParser(description='Run MuTILsWSI model.')
    parser.add_argument('-s', '--subsets', type=int, default=[0], nargs='+')
    parser.add_argument('-r', '--reverse', type=int, default=0)
    ARGS = parser.parse_args()

    RunConfigs.initialize()

    for subset in ARGS.subsets:

        logging.info('Entered the for loop with param: %s out of %s subsets', subset, ARGS.subsets)

        monitor = (
                f"{'(DEBUG)' if RunConfigs.RUN_KWARGS['_debug'] else ''}"
                f"{RunConfigs.RUN_KWARGS['COHORT']}: SUBSET {subset} "
                f"{'(reverse)' if ARGS.reverse else ''}"
                ": "
            )

        SLIDENAMES = RunConfigs.SLIDENAMES[subset]

        if ARGS.reverse:
            SLIDENAMES.reverse()

        runner = MuTILsWSIRunner(
            **RunConfigs.RUN_KWARGS,
            monitor=monitor,
            keep_slides=SLIDENAMES,
            _reverse=bool(ARGS.reverse)
        )

        runner.run_all_slides()
