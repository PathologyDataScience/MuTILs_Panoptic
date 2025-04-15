import os
from os.path import join as opj
import yaml
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from MuTILs_Panoptic.utils.TorchUtils import transform_dlinput
from MuTILs_Panoptic.utils.MiscRegionUtils import get_configured_logger
import MuTILs_Panoptic.configs.panoptic_model_configs as model_configs

configuration_file = os.path.abspath(__file__).split('.')[0] + '.yaml'

class Logger:

    @staticmethod
    def set_up_logger(log_dir: str) -> object:
        """ Set up the logger for the project.

        Args:
            log_dir (str): Path to the directory where the logs will be saved.

        Returns:
            object: The logger object."""
        if not os.path.exists(log_dir):
            raise FileNotFoundError('Folder does not exist')

        LOGDIR = opj(log_dir, 'LOGS')
        os.makedirs(LOGDIR, exist_ok=True)

        logger = get_configured_logger(logdir=LOGDIR, prefix='MuTILsWSIRunner', tofile=True)
        return logger

class ConfigParser:

    @staticmethod
    def parse_config(config_file: str) -> dict:
        """Load configuration from a YAML file.

        Args:
            config_file (str): Path to the configuration file.

        Returns:
            dict: The parsed configuration as a dictionary.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        try:
            with open(config_file, 'r') as ymlfile:
                return yaml.load(ymlfile, Loader=yaml.FullLoader)
        except FileNotFoundError as fnfe:
            raise fnfe
        except yaml.YAMLError as ye:
            raise ye

class RunConfigs:

    def __init__(self):
        """Initialize the configuration parameters for the pipeline.
        """
        # Load configuration parameters from the YAML file
        RUN_KWARGS = ConfigParser.parse_config(configuration_file)

        # Set further configuration parameters
        RUN_KWARGS['logger'] = Logger.set_up_logger(RUN_KWARGS['base_savedir'])
        RUN_KWARGS['slide_names'] = self.get_slide_names(RUN_KWARGS)

        self.config = Config(**RUN_KWARGS)

    @staticmethod
    def get_slide_names(run_kwargs: dict) -> list:
        """Get slide names.

        Args:
            run_kwargs (dict): Configuration dictionary.

        Returns:
            list: List of slide names.
        """
        all_slidenames = os.listdir(run_kwargs['slides_path'])
        all_slidenames.sort()

        return all_slidenames

    def get_config(self):
        """Get the configuration object.

        Returns:
            Config: The configuration object.
        """
        return self.config


def default_model_paths() -> Dict[str, str]:
    return {
        "mutils_06022021-fold1": "/home/models/fold_1/mutils_06022021_fold1.pt",
        "mutils_06022021-fold2": "/home/models/fold_2/mutils_06022021_fold2.pt",
        "mutils_06022021-fold3": "/home/models/fold_3/mutils_06022021_fold3.pt",
        "mutils_06022021-fold4": "/home/models/fold_4/mutils_06022021_fold4.pt",
        "mutils_06022021-fold5": "/home/models/fold_5/mutils_06022021_fold5.pt"
    }

def default_cnorm_kwargs() -> Dict[str, Any]:
    return {
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

def default_nprops_kwargs() -> Dict[str, Any]:
    return {
        'fsd_bnd_pts': 128,
        'fsd_freq_bins': 6,
        'cyto_width': 8,
        'num_glcm_levels': 32,
        'morphometry_features_flag': True,
        'fsd_features_flag': True,
        'intensity_features_flag': True,
        'gradient_features_flag': True,
        'haralick_features_flag':True
    }

def default_roi_kmeans_kvp() -> Dict[str, Any]:
    return {
            'n_segments': 128,
            'compactness': 10,
            'threshold': 9
    }

@dataclass
class Config:
    """Configuration class for the MuTILsWSIRunner."""
    slides_path: str = '/home/input'
    base_savedir: str = '/home/output'
    model_paths: Dict[str, str] = field(default_factory=default_model_paths)
    monitor: str = ''

    save_wsi_mask: bool = False
    save_annotations: bool = False
    save_nuclei_meta: bool = True
    save_nuclei_props: bool = True

    roi_side_hres: int = 1024
    discard_edge_hres: int = 0
    roi_clust_mpp: float = 20.0

    topk_rois: Optional[int] = None
    topk_salient_rois: Optional[int] = None

    vlres_scorer_kws: Dict[str, Any] = None
    roi_kmeans_kvp: Optional[Dict[str, Any]] = field(default_factory=default_roi_kmeans_kvp)

    topk_rois_sampling_mode: str = "stratified"
    independent_tile_assignment: bool = True

    cnorm: bool = True
    cnorm_kwargs: Dict[str, Any] = field(default_factory=default_cnorm_kwargs)
    maskout_regions_for_cnorm: List[str] = field(default_factory=lambda: ['BLOOD', 'WHITE', 'EXCLUDE'])

    ntta: int = 0
    dltransforms: Optional[List[Dict[str, Any]]] = None

    valid_extensions: List[str] = field(default_factory=lambda: ['.svs', '.tif', '.tiff', '.ndpi', '.mrxs', '.scn'])

    filter_stromal_whitespace: bool = False
    min_tumor_for_saliency: int = 4
    max_salient_stroma_distance: int = 64

    no_watershed_nucleus_classes: List[str] = field(default_factory=lambda: ['StromalCellNOS', 'ActiveStromalCellNOS'])

    min_nucl_size: int = 5
    max_nucl_size: int = 90
    nprops_kwargs: Optional[Dict[str, Any]] = field(default_factory=default_nprops_kwargs)

    mtp: Any = field(default_factory=lambda: model_configs.MuTILsParams)
    rcc: Any = field(default_factory=lambda: model_configs.RegionCellCombination)
    rcd: Any = field(default_factory=lambda: model_configs.RegionCellCombination.REGION_CODES)
    ncd: Any = field(default_factory=lambda: model_configs.RegionCellCombination.NUCLEUS_CODES)
    no_watershed_lbls: Dict = field(init=False)
    maskout_region_codes: List = field(init=False)

    hres_mpp: float = field(init=False)
    lres_mpp: float = field(init=False)
    vlres_mpp: float = field(init=False)
    h2l: float = field(init=False)
    h2vl: float = field(init=False)
    roi_side_lres: int = field(init=False)
    roi_side_vlres: int = field(init=False)
    n_edge_pixels_discarded: float = field(init=False)

    N_CPUs: int = 1
    _debug: bool = False

    logger: Optional[Any] = field(default=None)
    slide_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        os.makedirs(self.base_savedir, exist_ok=True)
        self.base_savedir = opj(self.base_savedir, 'perSlideResults')

        assert all(os.path.isfile(j) for j in self.model_paths.values()), \
            "Some of the models weight files do not exist!"

        assert self.topk_rois_sampling_mode in ("stratified", "weighted", "sorted")

        if self.topk_rois is not None:
            assert self.topk_salient_rois <= self.topk_rois, (
                "The no. of salient ROIs used for final TILs scoring must be "
                "less than the total no. of rois that we do inference on!"
            )

        self.no_watershed_lbls = {
            self.ncd[cls] for cls in self.no_watershed_nucleus_classes}

        self.maskout_region_codes = [
            self.rcd[reg] for reg in self.maskout_regions_for_cnorm]

        self.hres_mpp = self.mtp.model_params['hpf_mpp']
        self.lres_mpp = self.mtp.model_params['roi_mpp']
        self.vlres_mpp = 2 * self.lres_mpp
        self.h2l = self.hres_mpp / self.lres_mpp
        self.h2vl = self.hres_mpp / self.vlres_mpp
        self.roi_side_lres = int(self.h2l * self.roi_side_hres)
        self.roi_side_vlres = int(self.h2vl * self.roi_side_hres)
        self.n_edge_pixels_discarded = 4 * self.discard_edge_hres * (
            self.roi_side_hres - self.discard_edge_hres)

        self.mtp.model_params.update({
            'training': False,
            'roi_side': self.roi_side_lres,
            'hpf_side': self.roi_side_hres,  # predict all nuclei in roi
            'topk_hpf': 1,
        })

        self.vlres_scorer_kws = self.vlres_scorer_kws or {
            'check_tissue': True,
            'tissue_percent': 50,
            'pixel_overlap': int(2 * self.discard_edge_hres * self.h2vl),
        }

        self.dltransforms = self.dltransforms or transform_dlinput(
            tlist=['augment_stain'],
            make_tensor=False,
            augment_stain_sigma1=0.75,
            augment_stain_sigma2=0.75,
        )

        if self._debug:
            self.monitor = '[DEBUG]'

