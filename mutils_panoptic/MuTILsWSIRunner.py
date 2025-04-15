import os
import sys
from os.path import join as opj
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from pandas import DataFrame, read_csv
import torch.multiprocessing as mp
import torch
from glob import glob
import shutil
import logging
import pyvips
import time
import argparse
import threading
import random

# histolab
from histolab.slide import SlideSet
from histolab.tile import Tile
from histolab.tiler import ScoreTiler
from histolab.types import CoordinatePair
from histolab.masks import BiggestTissueBoxMask
from histolab.filters.image_filters_functional import rag_threshold

# histomicstk
from histomicstk.preprocessing.color_deconvolution import color_deconvolution_routine

# mutils
from MuTILs_Panoptic.configs.MuTILsWSIRunConfigs import RunConfigs
from MuTILs_Panoptic.utils.MiscRegionUtils import get_objects_from_binmask, numpy2vips
from MuTILs_Panoptic.utils.MiscRegionUtils import load_trained_mutils_model
from MuTILs_Panoptic.mutils_panoptic.MuTILsInference import RoiProcessorConfig, RoiProcessor
from MuTILs_Panoptic.utils.GeneralUtils import (
    load_json, save_json, CollectErrors, _divnonan, weighted_avg_and_std
)

collect_errors = CollectErrors()

# ==================================================================================================
class MuTILsWSIRunner:
    """ Run the MuTILs pipeline for a set of slides. """

    def __init__(self, config):
        # paths
        self.model_paths = config.model_paths
        self.slides_path = config.slides_path
        self.base_savedir = config.base_savedir
        # what (not) to save
        self._monitor = config.monitor
        self.save_wsi_mask = config.save_wsi_mask
        self.save_annotations = config.save_annotations
        self.save_nuclei_meta = config.save_nuclei_meta
        self.save_nuclei_props = config.save_nuclei_props
        # slides to analyse
        # self._reverse = _reverse
        # TODO: Take SlideSet out from here!
        self.slides = SlideSet(
            slides_path=self.slides_path,
            processed_path=self.base_savedir,
            valid_extensions=config.valid_extensions,
            keep_slides=config.slide_names,
            reverse=False,
            slide_kwargs={'use_largeimage': True},
        )
        # roi size and scoring
        self.roi_side_hres = config.roi_side_hres
        self.discard_edge_hres = config.discard_edge_hres
        self.roi_clust_mpp = config.roi_clust_mpp
        self.topk_rois = config.topk_rois
        self.independent_tile_assignment = config.independent_tile_assignment
        # parsing nuclei from inference
        self.no_watershed_nucleus_classes = config.no_watershed_nucleus_classes
        self.min_nucl_size = config.min_nucl_size
        self.max_nucl_size = config.max_nucl_size
        self.nprops_kwargs = config.nprops_kwargs
        # color normalization & augmentation
        self.cnorm = config.cnorm
        self.cnorm_kwargs = config.cnorm_kwargs
        self.maskout_regions_for_cnorm = config.maskout_regions_for_cnorm
        # config shorthands
        self.mtp = config.mtp
        self.rcc = config.rcc
        self.rcd = config.rcd
        self.ncd = config.ncd
        self.no_watershed_lbls = config.no_watershed_lbls
        self.maskout_region_codes = config.maskout_region_codes
        self.hres_mpp = config.hres_mpp
        self.lres_mpp = config.lres_mpp
        self.vlres_mpp = config.vlres_mpp
        self.h2l = config.h2l
        self.h2vl = config.h2vl
        self.roi_side_lres = config.roi_side_lres
        self.roi_side_vlres = config.roi_side_vlres
        self.n_edge_pixels_discarded = config.n_edge_pixels_discarded
        # vlres cellularity scorer & roi clustering
        self.vlres_scorer_kws = config.vlres_scorer_kws
        self.vlres_scorer_kws.update({
            'scorer': (
                self.tile_scorer_by_tissue_ratio if self.topk_rois is None
                else self.tile_scorer_by_deconvolution
            ),
            'tile_size': (self.roi_side_vlres, self.roi_side_vlres),
            'n_tiles': 0,  # ALL tiles
            'mpp': self.vlres_mpp,
        })
        self.roi_kmeans_kvp = config.roi_kmeans_kvp
        self._topk_rois_sampling_mode = config.topk_rois_sampling_mode
        # test-time color augmentation (0 = no augmentation)
        self.ntta = config.ntta
        self.dltransforms = config.dltransforms
        # intra-tumoral stroma (saliency)
        self.filter_stromal_whitespace = config.filter_stromal_whitespace
        self.min_tumor_for_saliency = config.min_tumor_for_saliency
        self.max_salient_stroma_distance = config.max_salient_stroma_distance
        self.topk_salient_rois = config.topk_salient_rois
        # internal properties that change with slide/roi/hpf (convenience)
        self._slide = None
        self._sldname = None
        self._slmonitor = None
        self._top_rois = None
        self._savedir = None
        self._mrois = None
        self._mdmonitor = None
        self._rid = None
        self._rmonitor = None
        self._hpf = None
        self._hid = None
        self._hpfname = None
        self._debug = config._debug
        # misc params
        self.N_CPUs = config.N_CPUs
        self.logger = config.logger or logging.getLogger(__name__)
        collect_errors.logger = self.logger
        collect_errors._debug = self._debug
        # model ensembles
        self.model_paths = config.model_paths
        self.n_models = len(config.model_paths)

        self.check_gpus()

# ==================================================================================================

    def check_gpus(self):
        """Check if GPUs are available. Set self.use_gpus accordingly: True if at least 5 GPUs are
        available, False otherwise.
        """
        self.logger.info("--- Checking GPUs ------------------------------------")
        if torch.cuda.is_available() and (torch.cuda.device_count() > 4):
            self.logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
            self.logger.info(f"Running multiple models in parallel on multiple GPUs.")
            self.load_all_models()
            use_gpus = True
        elif torch.cuda.is_available() and (torch.cuda.device_count() < 5):
            self.logger.info(f"Number of GPUs ({torch.cuda.device_count()}) is less than the number of models (5)")
            self.logger.info(f"Running single model at once on a single GPU.")
            use_gpus = False # Don't get confused, a single GPU still can be used!
        else:
            self.logger.info("No GPU available, using CPU instead.")
            use_gpus = False
        self.logger.info("------------------------------------------------------")

        self.use_gpus = use_gpus

    def load_model(self, model_path: str=None, device_id: str=None) -> torch.nn.Module:
        """Load a trained model from a given path and move it to the given device.

        Parameters
        ----------
        model_path : str
            The path to the trained model.
        device_id : str
            The device id to move the model to.

        Returns
        -------
        torch.nn.Module
            The loaded model.
        """
        if model_path is None:
            raise ValueError("Model path is missing!")
        device = torch.device(f'cuda:{device_id}') if torch.cuda.is_available() else torch.device('cpu')
        model = load_trained_mutils_model(model_path, mtp=self.mtp)
        model.eval()
        model.to(device)
        self.logger.info(f"Model {os.path.basename(model_path)} loaded on device {device}")
        return model

    def load_all_models(self) -> None:
        """Load all models if there are enough GPUs available (5)."""
        self.models = {}
        i = 0
        for model_name, model_path in self.model_paths.items():
            device_id = str(i) if torch.cuda.is_available() else None
            i += 1
            model = self.load_model(model_path, device_id)
            self.models[model_name] = {"model": model, "device": device_id}

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
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger.info(f"Starting slide {self._sldname}: {formatted_time} UTC")
        start_time = time.time()
        # when running slides in reverse order, strictly avoid running on
        # any slide if its director already exists to prevent conflicts
        self._savedir = opj(self.base_savedir, self._sldname)

        # This must be OUTSIDE self.run_slide() for error collection
        self._set_sldmeta()
        self._run_slide()
        self.summarize_slide()
        self._maybe_concat_wsi_mask()
        self._save_slide_metrics()
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.logger.info(f"Finishing slide {self._sldname}: {formatted_time} UTC")
        self.logger.info(f"Runtime of slide {self._sldname}: {np.around(time.time() - start_time, decimals=2)} seconds")

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

    @staticmethod
    def process_chunk(chunk: list, chunk_id: int, config: RoiProcessorConfig):
        """Processes a chunk of ROIs.

        Parameters
        ----------
        chunk : list
            List of tuples, each containing the id number of a ROI and the model which dedicated to
            do the inference on it.
        config : RoiProcessorConfig
            The configuration object containing parameters for the RoiProcessor.
        """
        roi_processor = RoiProcessor(config)
        roi_processor.run(chunk, chunk_id)

    @staticmethod
    def start_process(chunk_id: int, chunk: list, config: RoiProcessorConfig, processes: list):
        """ Starts a new process for a chunk of ROIs.

        Parameters
        ----------
        chunk_id : int
            The id of the chunk.
        chunk : list
            List of tuples, each containing the id number of a ROI and the model which dedicated to
            do the inference on it.
        config : RoiProcessorConfig
            The configuration object containing parameters for the RoiProcessor.
        processes : list
            A list of processes.
        """
        p = mp.Process(
            target=MuTILsWSIRunner.process_chunk,
            args=(chunk, chunk_id),
            kwargs={"config": config}
        )
        p.start()
        processes.append(p)

    def run_parallel_models(self, model_rois: dict):
        """Runs multiple chunks of ROIs in parallel using multiprocessing.

        Parameters
        ----------
        model_rois : dict
            A dictionary containing the model names as keys and a list of ROI id numbers as values.
        """
        model_roi_pairs = [(roi_id, model_name) for model_name in model_rois.keys() \
                           for roi_id in model_rois[model_name]]
        random.seed(42)
        random.shuffle(model_roi_pairs)
        roi_chunks = [model_roi_pairs[i::self.N_CPUs] for i in range(self.N_CPUs)]

        if self._debug:
            self.N_CPUs = 2
            number_of_rois_to_process = 4
            roi_chunks = [chunk[:number_of_rois_to_process] for chunk in roi_chunks[:self.N_CPUs]]

        self.logger.info(f"Number of CPUs to use: {self.N_CPUs}")
        _number_of_rois_to_process = sum(len(chunk) for chunk in roi_chunks)
        self.logger.info(f"Number of ROIs to use: {_number_of_rois_to_process}")

        mp.set_start_method("spawn", force=True)

        threads = []
        processes = []

        config = RoiProcessorConfig(
                        _debug=self._debug,
                        _sldname=self._sldname,
                        models=self.models,
                        _top_rois=self._top_rois,
                        _slide=self._slide,
                        hres_mpp=self.hres_mpp,
                        roi_side_hres=self.roi_side_hres,
                        roi_side_lres=self.roi_side_lres,
                        cnorm=self.cnorm,
                        cnorm_kwargs=self.cnorm_kwargs,
                        ntta=self.ntta,
                        discard_edge_hres=self.discard_edge_hres,
                        filter_stromal_whitespace=self.filter_stromal_whitespace,
                        no_watershed_nucleus_classes=self.no_watershed_nucleus_classes,
                        maskout_regions_for_cnorm=self.maskout_regions_for_cnorm,
                        min_nucl_size=self.min_nucl_size,
                        max_nucl_size=self.max_nucl_size,
                        max_salient_stroma_distance=self.max_salient_stroma_distance,
                        min_tumor_for_saliency=self.min_tumor_for_saliency,
                        _savedir=self._savedir,
                        save_wsi_mask=self.save_wsi_mask,
                        save_nuclei_meta=self.save_nuclei_meta,
                        save_nuclei_props=self.save_nuclei_props,
                        nprops_kwargs=self.nprops_kwargs,
                        save_annotations=self.save_annotations
                    )

        # Start all process simultaneouly in separate threads
        for chunk_id, chunk in enumerate(roi_chunks):
            t = threading.Thread(target=self.start_process, args=(chunk_id, chunk, config, processes))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        try:
            while True:
                all_done = True
                for p in processes:
                    if p.exitcode is None:
                        all_done = False
                    elif p.exitcode != 0:
                        self.logger.error(f"Process {p.pid} failed with exit code {p.exitcode}")
                        # Terminate all other processes
                        for other_p in processes:
                            if other_p.is_alive():
                                self.logger.info(f"Terminating process {other_p.pid}")
                                other_p.terminate()
                        # Clean up and exit
                        for p in processes:
                            p.join()
                        self.logger.error(f"Process {p.pid} failed, shutting down the pipeline.")
                        self._maybe_merge_logs()
                        sys.exit(1)

                if all_done:
                    self.logger.info("All processes completed successfully.")
                    break

                time.sleep(0.1)  # Polling interval

        except KeyboardInterrupt:
            self.logger.error("Keyboard interrupt detected. Terminating all processes.")
            for p in processes:
                p.terminate()
            self._maybe_merge_logs()
            sys.exit(1)

        for p in processes:
            p.join()

    @collect_errors()
    def run_single_model(self, _modelname, _mrois):
        """Load & run a single model for all assigned rois."""
        # load model weights
        model_path = self.model_paths[_modelname]
        if torch.cuda.is_available():
            model = self.load_model(model_path, device_id=0)
            device = str(0)
        else:
            model = self.load_model(model_path)
            device = "cpu"
        models = {_modelname: {"model": model, "device": device}}
        # create roi processor
        config = RoiProcessorConfig(
                _debug=self._debug,
                _sldname=self._sldname,
                models=models,
                _top_rois=self._top_rois,
                _slide=self._slide,
                hres_mpp=self.hres_mpp,
                roi_side_hres=self.roi_side_hres,
                roi_side_lres=self.roi_side_lres,
                cnorm=self.cnorm,
                cnorm_kwargs=self.cnorm_kwargs,
                ntta=self.ntta,
                discard_edge_hres=self.discard_edge_hres,
                filter_stromal_whitespace=self.filter_stromal_whitespace,
                no_watershed_nucleus_classes=self.no_watershed_nucleus_classes,
                maskout_regions_for_cnorm=self.maskout_regions_for_cnorm,
                min_nucl_size=self.min_nucl_size,
                max_nucl_size=self.max_nucl_size,
                max_salient_stroma_distance=self.max_salient_stroma_distance,
                min_tumor_for_saliency=self.min_tumor_for_saliency,
                _savedir=self._savedir,
                save_wsi_mask=self.save_wsi_mask,
                save_nuclei_meta=self.save_nuclei_meta,
                save_nuclei_props=self.save_nuclei_props,
                nprops_kwargs=self.nprops_kwargs,
                save_annotations=self.save_annotations
            )
        roi_processor = RoiProcessor(config)
        # Iterate through one roi at a time
        nrois = len(_mrois)
        for rno, _rid in enumerate(_mrois):

            if self._debug:
                _rid = rno
                if rno > 1:
                    break

            self._rmonitor = f"{self._mdmonitor}: roi {rno + 1} of {nrois}"
            collect_errors.monitor = self._rmonitor
            self.logger.info(self._rmonitor)
            roi_id_and_model_name = [_rid, _modelname]
            roi_processor.logger = self.logger
            roi_processor.run_roi(roi_id_and_model_name)

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

    # HELPERS --------------------------------------------------------------------------------------
    @property
    def _roicoords(self):
        return [int(j) for j in self._top_rois[self._rid][1]]

    def _save_slide_metrics(self):
        """
        Save slide level results.
        """
        self.logger.info(f"{self._slmonitor}: Done, saving results.")
        new_json = True
        if new_json:
            save_json(
                self._sldmeta,
                path=opj(self._savedir, self._slide.name + '.json'))

    @collect_errors()
    def _maybe_concat_wsi_mask(self):
        """
        concatenate roi combined masks into a WSI level prediction.
        """
        if not self.save_wsi_mask:
            return

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
        sf = self._slide.base_mpp / self.hres_mpp
        mask_paths = glob(opj(self._savedir, 'roiMasks', '*.png'))
        for _, mpath in enumerate(mask_paths):
            self._rid = int(mpath.split('_roi-')[-1].split('_')[0])
            left, top, _, _ = [int(j * sf) for j in self._roicoords]
            mtile = pyvips.Image.new_from_file(mpath, access="sequential")
            wsimask = wsimask.insert(mtile, left, top, expand=True,
                                     background=0)
        # save result
        self.logger.info(f"{self._slmonitor}: Saving WSI mask.")
        pixel_per_mm = 1 / (1e-3 * self.hres_mpp)
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

        return summ

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
        if self.use_gpus:
            self.run_parallel_models(model_rois)
            self._maybe_merge_logs()
        else:
            mno = -1
            for _modelname, _mrois in model_rois.items():
                mno += 1
                if self._debug and mno > 1:
                    break

                self._mdmonitor = f"{self._slmonitor}: {_modelname}"
                collect_errors.monitor = self._mdmonitor
                self.logger.info(self._mdmonitor)
                self.run_single_model(_modelname, _mrois)

        collect_errors.monitor = self._slmonitor

    def _assign_rois_to_models(self):
        """
        Assign rois to trained model folds (as a form of ensembling).
        """
        self.logger.info(f"{self._slmonitor}: assign rois to models ..")
        # get topk rois, stratified by superpixel in vlres adjacency image
        df = self._assign_rois_to_rag()
        top_idxs = self._pick_topk_rois_from_rag(df)
        if self.independent_tile_assignment:
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

        savenames = (self._save_roilocs_hui_style(model_rois))
        self._sldmeta['outputs'].extend([str(j) for j in savenames])

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
        # assign to various models for ensembling
        roi_df = self._assign_rois_to_models()
        self._save_roi_assignment(report_path, roi_df=roi_df)
        # maybe visualize for sanity
        roi_models = np.array(roi_df.loc[:, 'model'], dtype=int)
        self._maybe_vis_roilocs(roi_extractor, models=roi_models)

        return roi_models

    @collect_errors()
    def _maybe_vis_roilocs(self, extractor: ScoreTiler, models):
        """
        Visualize ROI locations.
        """
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

    @staticmethod
    def _ksums(records, ks: list, k0=None):
        """Get sum of some value (eg pixels) from multiple rois."""
        whats = records if k0 is None else [j[k0] for j in records]
        whats = DataFrame.from_records(
            whats,
            exclude=[k for k, v in whats[0].items() if isinstance(v, dict)]
        ).loc[:, ks]

        return whats.sum(0).to_dict()

    @staticmethod
    def _get_latest_logfile(directory, pattern: str="MuTILsWSIRunner_2*"):
        """
        Returns the latest (most recently modified) file in the directory.

        Args:
            directory (str): Path to the directory.
            pattern (str): Pattern to match files.

        Returns:
            str: Path to the latest file, or None if no files are found.
        """
        files = glob(os.path.join(directory, pattern))
        if not files:
            return None

        latest_file = max(files, key=os.path.getmtime)
        return latest_file

    def _maybe_merge_logs(self):
        """Merge logs from parallel processes."""
        if not self.use_gpus:
            return

        logdir = '/home/output/LOGS'
        latest_logfile = self._get_latest_logfile(logdir)
        chunk_logs = glob(opj(logdir, 'MuTILsWSIRunner_chunk*'))

        with open(latest_logfile, 'a') as outfile:
            for log in chunk_logs:
                with open(log, 'r') as infile:
                    content = infile.read()
                    outfile.write(content)

        # Delete the chunk logs
        for log in chunk_logs:
            try:
                os.remove(log)
            except Exception as e:
                self.logger.info(f"Failed to delete {log}: {e}")
# ==================================================================================================


if __name__ == "__main__":

    start = time.time()

    # Get the configuration
    runconfig = RunConfigs()
    config = runconfig.get_config()

    config.logger.info("------------------------------------------------------")
    config.logger.info("          *** STARTING MUTILSWSIRUNNER ***            ")
    config.logger.info("------------------------------------------------------")

    # Set up the MuTILsWSIRunner
    runner = MuTILsWSIRunner(config)

    runner.run_all_slides()

    logging.info(f"Total runtime of MuTILsWSI: {np.around(time.time() - start, decimals=2)} seconds")