import os
from os.path import join as opj
from pandas import read_csv
import yaml

# Configure logger. This must be done BEFORE importing histolab modules
from MuTILs_Panoptic.utils.MiscRegionUtils import get_configured_logger, load_region_configs
from MuTILs_Panoptic.utils.GeneralUtils import splitlist

HOME = os.path.expanduser('~')
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG_FILE = opj(FILE_PATH, 'config.yaml')

class BaseConfigs:

    def __init__(self, config_file) -> None:
        self.logger = self.set_up_logger()
        self.config_dict = self.load_config(config_file)
        self.set_config_values()

    def set_up_logger(self) -> object:
        '''
        Set up the logger for the class.

        Returns:
            object: The logger object.
        '''
        LOGDIR = opj(self.OUTPUT_PATH, 'LOGS')
        os.makedirs(LOGDIR, exist_ok=True)
        logger = get_configured_logger(
            logdir=LOGDIR, prefix='MuTILsWSIRunner', tofile=True
        )

        return logger


    def load_config(self, file_path: str) -> dict:
        """
        Load configuration from a YAML file.

        Parameters:
            file_path (str): The path to the YAML configuration file.

        Returns:
            dict: The loaded configuration as a dictionary.

        Raises:
            FileNotFoundError: If the configuration file is not found.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        try:
            with open(file_path, 'r') as ymlfile:
                return yaml.load(ymlfile, Loader=yaml.FullLoader)
        except FileNotFoundError as e:
            self.logger.error(f"Configuration file not found: {file_path}")
            raise e
        except yaml.YAMLError as exc:
            self.logger.error(f"Error parsing YAML file: {exc}")
            raise exc

    def set_config_values(self):
        '''
        Set the configuration values from the config file to the class attributes.

        Raises:
            KeyError: If any of the required configuration values are not set in the config file.
        '''
        required_keys = ['DEBUG', 'MODELNAME', 'COHORT', 'N_SUBSETS', 'BASEPATH',
                         'INPUT_PATH', 'OUTPUT_PATH']
        for key in required_keys:
            if key not in self.config_dict:
                self.logger.error(f"{key} not set in config file")
                raise KeyError(f"{key} not set in config file")

        self.DEBUG = self.config_dict['DEBUG']
        self.MODELNAME = self.config_dict['MODELNAME']
        self.COHORT  = self.config_dict['COHORT']
        self.N_SUBSETS = self.config_dict['N_SUBSETS']
        # self.N_SUBSETS = 8 if not self.DEBUG else 1
        # self.N_SUBSETS = 16 if not self.DEBUG else 1
        self.BASEPATH = self.config_dict['BASEPATH']
        self.INPUT_PATH = self.config_dict['INPUT_PATH']
        self.OUTPUT_PATH = self.config_dict['OUTPUT_PATH']
        self.BASEMODELPATH = self.config_dict['BASEMODELPATH']

class GrandChallengeConfigs(BaseConfigs):

    def __init__(self, config_file) -> None:
        super().__init__(config_file)

    # TODO: Make proper class inheritance
    WORKPATH = opj(HOME, 'Desktop', 'TILS_CHALLENGE')
    INPUT_PATH = opj(WORKPATH, '1_INPUT', BaseConfigs.COHORT)
    OUTPUT_PATH = opj(WORKPATH, '2_OUTPUT', BaseConfigs.COHORT)
    BASEMODELPATH = opj(WORKPATH, '0_MODELS', BaseConfigs.MODELNAME)

    restrict_to_vta = False

    save_wsi_mask = True
    save_annotations = True
    save_nuclei_meta = False
    save_nuclei_props = False

    grandch = True

    # uncomment below to run on GC platform which runs on one slide at a time
    # so the slides will overwrite these files
    gcpaths = {
        'roilocs_in': opj(INPUT_PATH, "regions-of-interest.json"),
        'cta2vta': opj(WORKPATH, '0_MODELS', 'Calibrations.json'),
        'roilocs_out': opj(OUTPUT_PATH, 'regions-of-interest.json'),
        'result_file': opj(OUTPUT_PATH, 'results.json'),
        'tilscore_file': opj(OUTPUT_PATH, 'til-score.json'),
        'detect_file': opj(OUTPUT_PATH, 'detected-lymphocytes.json'),
        'wsi_mask': opj(OUTPUT_PATH, 'images', 'segmented-stroma'),
    }
    # gcpaths = None  # each slide has its own folder (no overwrite)

    topk_rois = 300
    topk_rois_sampling_mode = "weighted"
    topk_salient_rois = 300
    vlres_scorer_kws = {
        'check_tissue': True,
        'tissue_percent': 25,
        'pixel_overlap': 0,
    }


class FullcTMEConfigs(BaseConfigs):

    def __init__(self, config_file) -> None:
        super().__init__(config_file)

    # TODO: Make proper class inheritance
    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
    REPO_PATH = os.path.dirname(CURRENT_PATH)

    INPUT_PATH = opj(REPO_PATH,'data/input', BaseConfigs.COHORT)
    OUTPUT_PATH = opj(REPO_PATH,'data/output', BaseConfigs.COHORT)
    BASEMODELPATH = opj(
        BaseConfigs.BASEPATH, 'results', 'mutils', 'models', BaseConfigs.MODELNAME
    )

    restrict_to_vta = False

    save_wsi_mask = True
    save_annotations = False
    save_nuclei_meta = True
    save_nuclei_props = True

    grandch = False
    gcpaths = None
    topk_rois = None
    topk_rois_sampling_mode = "stratified"
    topk_salient_rois = None
    vlres_scorer_kws = None


class NiceExamplesForMuTILsPreprintConfigs(BaseConfigs):

    def __init__(self, config_file) -> None:
        super().__init__(config_file)

    # TODO: Make proper class inheritance
    INPUT_PATH = opj('/input', BaseConfigs.COHORT)
    OUTPUT_PATH = opj('/output', BaseConfigs.COHORT)
    BASEMODELPATH = opj(
        BaseConfigs.BASEPATH, 'results', 'mutils', 'models', BaseConfigs.MODELNAME
    )

    restrict_to_vta = False

    save_wsi_mask = True
    save_annotations = False
    save_nuclei_meta = False
    save_nuclei_props = False

    grandch = False
    gcpaths = None
    topk_rois = None
    topk_rois_sampling_mode = "stratified"
    topk_salient_rois = None
    vlres_scorer_kws = None


class RunConfigs:

    # IMPORTANT: THIS SWITCHES THE ANALYSIS TYPE
    # cfg = NiceExamplesForMuTILsPreprintConfigs
    # cfg = GrandChallengeConfigs
    cfg = FullcTMEConfigs(BASE_CONFIG_FILE)

    # model weights
    MODEL_PATHS = {}
    for f in range(1, 6):
        MODEL_PATHS[f'{cfg.MODELNAME}-fold{f}'] = opj(
            cfg.BASEMODELPATH, f'fold_{f}', f'{cfg.MODELNAME}_fold{f}.pt'
        )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Subdividing cohort to run on multiple docker instances

    ALL_SLIDENAMES = os.listdir(cfg.INPUT_PATH)

    if BaseConfigs.COHORT.startswith('TCGA') and cfg.restrict_to_vta:
        # RESTRICT TO ROBERTO SALGADO ASSESSED SLIDES
        SLIDENAMES = read_csv(opj(
            cfg.BASEPATH, 'data', 'tcga-clinical',
            'PRIVATE_RSalgado_TCGA_TILScores.csv'
        )).iloc[:, 0].to_list()
        SLIDENAMES = [j[:12] for j in SLIDENAMES]
        SLIDENAMES = [j for j in ALL_SLIDENAMES if j[:12] in SLIDENAMES]

    elif BaseConfigs.COHORT.endswith('CPS2') and cfg.restrict_to_vta:
        # RESTRICT TO TED ASSESSED SLIDES (CPS2)
        acs_vta = read_csv(
            opj(cfg.BASEPATH, 'data', 'acs-clinical',
                'CPSII_BRCA_FacilityIDs_20210331.csv'),
            index_col=0)
        acs_vta.rename(columns={'TILS_STR': 'vta'}, inplace=True)
        acs_vta = acs_vta.loc[:, 'vta'].map(lambda x: float(x) / 100).dropna()
        SLIDENAMES = list(acs_vta.index)
        SLIDENAMES = [j for j in ALL_SLIDENAMES if j.split('_')[0] in SLIDENAMES]

    else:
        SLIDENAMES = ALL_SLIDENAMES
        SLIDENAMES.sort()
        SLIDENAMES = splitlist(
            SLIDENAMES, len(SLIDENAMES) // cfg.N_SUBSETS
        )

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    RUN_KWARGS = dict(

        # paths & slides
        model_configs=load_region_configs(
            opj(cfg.BASEMODELPATH, 'region_model_configs.py'), warn=False
        ),
        model_paths=MODEL_PATHS,
        slides_path=cfg.INPUT_PATH,
        base_savedir=cfg.OUTPUT_PATH,

        # size params
        # roi_side_hres=512 if BaseConfigs.DEBUG else 1024,
        roi_side_hres=1024,
        discard_edge_hres=0,  # keep 0 -> slow + can't get the gap to be exact
        logger=cfg.logger,

        # Defined in cfg
        save_wsi_mask=cfg.save_wsi_mask,
        save_annotations=cfg.save_annotations,
        save_nuclei_meta=cfg.save_nuclei_meta,
        save_nuclei_props=cfg.save_nuclei_props,
        grandch=cfg.grandch,
        gcpaths=cfg.gcpaths,
        topk_rois=cfg.topk_rois,
        topk_rois_sampling_mode=cfg.topk_rois_sampling_mode,
        topk_salient_rois=cfg.topk_salient_rois,
        vlres_scorer_kws=cfg.vlres_scorer_kws,

        _debug=cfg.DEBUG,
    )
